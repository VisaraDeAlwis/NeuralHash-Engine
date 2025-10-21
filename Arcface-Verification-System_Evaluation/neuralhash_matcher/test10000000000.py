from PIL import Image
from tqdm import tqdm
import csv
import numpy as np
import time
import json
from sklearn.metrics import accuracy_score
from neuralhash_utils_R100 import (
    load_pca_model, load_hyperplanes,
    get_embedding, hash_embedding, calculate_hamming_distance
)

# ----------------- Constants -----------------
LFW_DIR = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled"
PAIRS_FILE = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\LFW\pairs_new.csv"
PCA_MODEL_PATH = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\models\pca_512_to_128.pkl"
HYPERPLANE_PATH = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\models\neuralhash_128x96_seed1.dat"
HASH_CACHE_FILE = "neuralhash_cacherf.json"

# ----------------- Utility Functions -----------------
def get_image_path(lfw_dir, person, image_num):
    """Builds full path for a given person and image number."""
    return os.path.join(lfw_dir, person, f"{person}_{int(image_num):04d}.jpg")

def calculate_accuracy(threshold, distances, labels):
    """Calculates accuracy for a given threshold."""
    predictions = (distances <= threshold).astype(int)
    return accuracy_score(labels, predictions)

# ----------------- Load Models -----------------
try:
    pca_model = load_pca_model(PCA_MODEL_PATH)
    hyperplanes = load_hyperplanes(HYPERPLANE_PATH)
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)


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

# ----------------- Test Embedding -----------------
emb1 = get_embedding(r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg")

emb2 = get_embedding(r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled\Annette_Lu\Annette_Lu_0001.jpg")


cosiine = cosine_similarity(emb1,emb2)
print(cosiine)


nhash1 = hash_embedding(emb1, pca_model, hyperplanes)
nhash2 = hash_embedding(emb2,pca_model,hyperplanes)
print("nhash1", nhash1)
print("Nhash2", nhash2)
x=calculate_hamming_distance(nhash1,nhash2)
print("Hamm",x)
print("✅ Test passed.\n")