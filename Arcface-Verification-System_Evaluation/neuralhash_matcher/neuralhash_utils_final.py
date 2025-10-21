# neuralhash_utils.py

from facenet_pytorch import MTCNN
import onnxruntime as ort
import numpy as np
import pickle
from PIL import Image
import torch
import os
from torchvision.transforms import ToPILImage

# ----------------- MTCNN Initialization -----------------
mtcnn = MTCNN(image_size=112, margin=0, keep_all=True)

providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if ort.get_device() == "GPU"
    else ["CPUExecutionProvider"]
)

ONNX_MODEL_PATH = "../models/R100@Glint360k.onnx"
resnet = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

input_name = resnet.get_inputs()[0].name
output_name = resnet.get_outputs()[0].name

to_pil = ToPILImage()

# ----------------- Model Loading -----------------
def load_pca_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)['pca_model']

def load_hyperplanes(path):
    file_size = os.path.getsize(path)
    dtype = np.float32
    bytes_per_elem = np.dtype(dtype).itemsize
    expected_bytes = 128 * 96 * bytes_per_elem
    header_bytes = 32

    with open(path, "rb") as f:
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

# ----------------- Embedding -----------------
def get_embedding(image_path):
    """
    Generates a 512-d normalized embedding for the largest detected face.
    Discards all other faces if more than one is detected.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert('RGB')
    boxes, _ = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        raise ValueError("No face detected")

    # Get all faces (since keep_all=True)
    faces = mtcnn(img)

    # --- Handle multiple faces ---
    if faces is None:
        raise ValueError("No faces returned by MTCNN")

    # If multiple faces, pick the largest one
    if faces.ndim == 4 and len(boxes) > 1:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        largest_idx = np.argmax(areas)
        face = faces[largest_idx]
    elif faces.ndim == 4:  # single face but still batched
        face = faces[0]
    else:
        face = faces

    # --- Prepare for ONNX inference ---
    face_np = face.unsqueeze(0).cpu().numpy().astype(np.float32)
    emb_512 = resnet.run([output_name], {input_name: face_np})[0].squeeze()
    emb_512 /= np.linalg.norm(emb_512)

    return emb_512

# ----------------- NeuralHash -----------------
def hash_embedding(emb_512, pca, hyperplanes):
    emb_128 = pca.transform([emb_512])[0]
    emb_128 /= np.linalg.norm(emb_128)
    bits = (np.dot(hyperplanes, emb_128) > 0).astype(np.uint8)
    return bits

def hash_embedding_512(emb_512, hyperplanes):
    emb_512 /= np.linalg.norm(emb_512)
    bits = (np.dot(hyperplanes, emb_512) > 0).astype(np.uint8)
    return bits

def bits_to_hex(bits):
    return ''.join(f"{int(''.join(map(str, bits[i:i+8])), 2):02x}" for i in range(0, len(bits), 8))

def calculate_hamming_distance(bits1, bits2):
    return sum(int(b1 != b2) for b1, b2 in zip(bits1, bits2))

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("One or both vectors have zero magnitude.")
    return np.dot(vec1, vec2) / (norm1 * norm2)
