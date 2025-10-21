# neuralhash_utils.py

from facenet_pytorch import MTCNN
import onnxruntime as ort
import numpy as np
import pickle
from PIL import Image
import torch
import os
from torchvision.transforms import ToPILImage

# Initialize MTCNN to keep all detected faces so we can select the largest
mtcnn = MTCNN(image_size=112, margin=0, keep_all=True) 

providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if ort.get_device() == "GPU"
    else ["CPUExecutionProvider"]
)

 # ArcFace typically uses 112x112
ONNX_MODEL_PATH = "../models/R100@Glint360k.onnx"

# Load ONNX model
resnet = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

# Extract input/output names dynamically
input_name = resnet.get_inputs()[0].name
output_name = resnet.get_outputs()[0].name


# Helper to convert tensor to a savable PIL Image
to_pil = ToPILImage()


def load_pca_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)['pca_model']

#def load_hyperplanes(path):
 #   with open(path, "rb") as f:
  #      f.seek(32)  # Skip the first 32 bytes of metadata
   #     arr = np.fromfile(f, dtype=np.int8, count=128*96)
    #arr = arr.reshape(96, 128).astype(np.float32)
    #return arr

def load_hyperplanes(path):
    file_size = os.path.getsize(path)
    dtype = np.float32
    bytes_per_elem = np.dtype(dtype).itemsize  # 4 bytes for float32
    expected_bytes = 128 * 96 * bytes_per_elem
    header_bytes = 32

    with open(path, "rb") as f:
        # If file has extra bytes (likely header)
        if file_size > expected_bytes:
            f.seek(header_bytes)
            arr = np.fromfile(f, dtype=dtype, count=128 * 96)
        else:
            arr = np.fromfile(f, dtype=dtype)

    arr = arr.reshape(96, 128)
    return arr

def load_hyperplanes_512(path):
    with open(path, "rb") as f:
            arr = np.fromfile(f, dtype=np.float32)
    arr = arr.reshape(96, 512)
    return arr

def get_embedding(image_path):
    """
    Generates a 512-d normalized embedding for a single detected face.
    """
    # --- Step 1: Load image ---
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert('RGB')

    # --- Step 2: Detect faces using MTCNN ---
    boxes, _ = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        raise ValueError("No face detected")

    # --- Step 3: Extract aligned face tensor(s) ---
    face_tensors = mtcnn(img)

    if isinstance(face_tensors, torch.Tensor):
        face_tensors = [face_tensors]  # handle single tensor output

    # --- Step 4: Select largest detected face ---
    if len(face_tensors) > 1:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        face = face_tensors[np.argmax(areas)]
    else:
        face = face_tensors[0]

    # --- Step 5: Prepare input for ONNX model ---
    face_np = face.unsqueeze(0).cpu().numpy().astype(np.float32)  # (1, 3, 112, 112)
    if face_np.ndim == 5:  # Fix extra dimension issue
        face_np = np.squeeze(face_np, axis=1)

    # --- Step 6: Run ONNX inference ---
    emb_512 = resnet.run([output_name], {input_name: face_np})[0].squeeze()

    # --- Step 7: Normalize embedding ---
    emb_512 = emb_512 / np.linalg.norm(emb_512)

    return emb_512


def hash_embedding(emb_512, pca, hyperplanes):
    """
    Takes a 512-d embedding and converts it into a 96-bit NeuralHash.
    """
    emb_128 = pca.transform([emb_512])[0]
    emb_128 /= np.linalg.norm(emb_128)
    bits = (np.dot(hyperplanes, emb_128) > 0).astype(np.uint8)
    return bits

def hash_embedding_512(emb_512, hyperplanes):
    """
    Takes a 512-d embedding and converts it into a 96-bit NeuralHash.
    """
    emb_512 /= np.linalg.norm(emb_512)
    bits = (np.dot(hyperplanes, emb_512) > 0).astype(np.uint8)
    return bits

def bits_to_hex(bits):
    return ''.join(f"{int(''.join(map(str, bits[i:i+8])),2):02x}" for i in range(0, len(bits), 8))


def calculate_hamming_distance(bits1, bits2):
    return sum(int(b1 != b2) for b1, b2 in zip(bits1, bits2))  # ensure Python int



def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two embedding vectors.
    Returns a value between -1 (opposite) and 1 (identical).
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Avoid division by zero
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("One or both vectors have zero magnitude.")
    
    return np.dot(vec1, vec2) / (norm1 * norm2)
