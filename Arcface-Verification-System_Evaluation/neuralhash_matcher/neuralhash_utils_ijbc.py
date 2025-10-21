# neuralhash_utils.py

from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pickle
from PIL import Image
from torchvision.transforms import ToPILImage

# Initialize MTCNN to keep all detected faces so we can select the largest
mtcnn = MTCNN(image_size=160, margin=0, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Helper to convert tensor to a savable PIL Image
to_pil = ToPILImage()

def load_pca_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)['pca_model']


def load_hyperplanes(path):
    arr = np.fromfile(path, dtype=np.int8).reshape(-1, 128).astype(np.float32)[:96]
    return arr


def get_embedding(image_path):
    """
    Generates a 512-d feature vector for a single face in an image.
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        return None # Return None if image is not found

    face_tensors = mtcnn(img)

    if face_tensors is None:
        return None # Return None if no face is detected

    # ... (Your logic to select the largest face if multiple are detected) ...
    if face_tensors.dim() > 3 and face_tensors.shape[0] > 1:
        boxes, _ = mtcnn.detect(img)
        box_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        largest_box_index = np.argmax(box_areas)
        face_tensor = face_tensors[largest_box_index]
    else:
        face_tensor = face_tensors[0] if face_tensors.dim() > 3 else face_tensors

    # This is the key part: return the 512-d embedding BEFORE hashing
    emb_512 = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy().squeeze()
    return emb_512


def hash_embedding(emb_512, pca, hyperplanes):
    """
    Takes a 512-d embedding and converts it into a 96-bit NeuralHash.
    """
    emb_128 = pca.transform([emb_512])[0]
    emb_128 /= np.linalg.norm(emb_128)
    bits = (np.dot(hyperplanes, emb_128) > 0).astype(np.uint8)
    return bits

# ... (keep your other utility functions like calculate_hamming_distance) ...


def bits_to_hex(bits):
    return ''.join(f"{int(''.join(map(str, bits[i:i+8])),2):02x}" for i in range(0,len(bits),8))

def calculate_hamming_distance(bits1, bits2):
    return sum(int(b1 != b2) for b1, b2 in zip(bits1, bits2))  # ensure Python int
