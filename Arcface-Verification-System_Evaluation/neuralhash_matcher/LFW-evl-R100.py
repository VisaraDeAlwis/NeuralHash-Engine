import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv
import time
import json
from sklearn.metrics import accuracy_score
from neuralhash_utils_R100 import (
    load_pca_model, load_hyperplanes,
    get_embedding, hash_embedding, calculate_hamming_distance, cosine_similarity,
    load_hyperplanes_512, hash_embedding_512
)

# ----------------- Constants -----------------
LFW_DIR = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled"
PAIRS_FILE = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\LFW\pairs_new.csv"
PCA_MODEL_PATH = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\models\pca_512_to_128.pkl"
HYPERPLANE_PATH = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\models\neuralhash_128x96_seed1.dat"
HYPERPLANE_PATH_128 = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\Dat FIle\my_seed128.dat"
HYPERPLANE_PATH_512 = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\Dat FIle\my_seed512.dat"
HASH_CACHE_FILE = "neuralhash_cacherf.json"
EMBED_CACHE_FILE = "lfw_embeddings_cache.json"

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
    hyperplanes128 = load_hyperplanes(HYPERPLANE_PATH_128)
    hyperplanes512 = load_hyperplanes_512(HYPERPLANE_PATH_512)
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

# ----------------- Step 1: Load or Generate Embedding Cache -----------------
if os.path.exists(EMBED_CACHE_FILE):
    print(f"📂 Loading cached embeddings from {EMBED_CACHE_FILE}...")
    with open(EMBED_CACHE_FILE, "r") as f:
        embedding_cache = json.load(f)
    print(f"✅ Loaded {len(embedding_cache)} embeddings.\n")
else:
    print("⚙️ No embedding cache found. Generating embeddings for all unique images...\n")
    embedding_cache = {}

    with open(PAIRS_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in tqdm(reader, desc="Generating Embeddings"):
            if len(row) == 4 and all(row):
                person1, image_num1, person2, image_num2 = row
            elif len(row) == 3 and all(row):
                person1, image_num1, image_num2 = row
                person2 = person1
            else:
                continue

            path1 = get_image_path(LFW_DIR, person1, image_num1)
            path2 = get_image_path(LFW_DIR, person2, image_num2)

            for path in [path1, path2]:
                if path not in embedding_cache and os.path.exists(path):
                    try:
                        emb = get_embedding(path)
                        embedding_cache[path] = emb.tolist()  # Convert numpy → list for JSON
                    except Exception as e:
                        print(f"⚠️ Error processing {path}: {e}")

    with open(EMBED_CACHE_FILE, "w") as f:
        json.dump(embedding_cache, f)
    print(f"💾 Saved {len(embedding_cache)} embeddings to {EMBED_CACHE_FILE}\n")

# ----------------- Step 2: Evaluate Using Cached Embeddings -----------------
all_pairs_data = []
start_time = time.time()

with open(PAIRS_FILE, "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header

    for row in tqdm(reader, desc="Evaluating Pairs"):
        if len(row) == 4 and all(row):
            person1, image_num1, person2, image_num2 = row
            is_match = False
        elif len(row) == 3 and all(row):
            person1, image_num1, image_num2 = row
            person2 = person1
            is_match = True
        else:
            continue

        path1 = get_image_path(LFW_DIR, person1, image_num1)
        path2 = get_image_path(LFW_DIR, person2, image_num2)

        if path1 not in embedding_cache or path2 not in embedding_cache:
            continue

        emb1 = np.array(embedding_cache[path1])
        emb2 = np.array(embedding_cache[path2])

        hash1 = hash_embedding(emb1, pca_model, hyperplanes)
        hash1_128 = hash_embedding(emb1, pca_model, hyperplanes128)
        hash1_512 = hash_embedding_512(emb1, hyperplanes512)

        hash2 = hash_embedding(emb2, pca_model, hyperplanes)
        hash2_128 = hash_embedding(emb2, pca_model, hyperplanes128)
        hash2_512 = hash_embedding_512(emb2, hyperplanes512)

        dist = calculate_hamming_distance(hash1, hash2)
        dist_128 = calculate_hamming_distance(hash1_128, hash2_128)
        dist_512 = calculate_hamming_distance(hash1_512, hash2_512)
        label = 1 if is_match else 0

        print(f"\nCosine Similarity: {cosine_similarity(emb1, emb2)} - {label}")
        print(f"Distance hash1_Apl: {dist} - {label}")
        print(f"Distance hash1_128: {dist_128} - {label}")
        print(f"Distance hash1_512: {dist_512} - {label}")

        all_pairs_data.append({'distance': dist, 'label': label})

end_time = time.time()
print("✅ Evaluation completed successfully.\n")

# ----------------- Step 3: 10-Fold Cross-Validation -----------------
accuracies = []
if all_pairs_data:
    fold_size = len(all_pairs_data) // 10
    print("\nStep 3: Starting 10-Fold Cross-Validation...")
    for i in tqdm(range(10), desc="Processing Folds"):
        start, end = i * fold_size, (i + 1) * fold_size
        test_data = all_pairs_data[start:end]
        train_data = all_pairs_data[:start] + all_pairs_data[end:]

        train_distances = np.array([p['distance'] for p in train_data])
        train_labels = np.array([p['label'] for p in train_data])
        test_distances = np.array([p['distance'] for p in test_data])
        test_labels = np.array([p['label'] for p in test_data])

        best_fold_accuracy = 0
        best_fold_threshold = 0
        for threshold in range(len(hyperplanes) + 1):
            acc = calculate_accuracy(threshold, train_distances, train_labels)
            if acc > best_fold_accuracy:
                best_fold_accuracy = acc
                best_fold_threshold = threshold

        fold_accuracy = calculate_accuracy(best_fold_threshold, test_distances, test_labels)
        accuracies.append(fold_accuracy)

if accuracies:
    mean_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies)

    print("\n--- LFW 10-Fold Cross-Validation Results ---")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation: (+/-) {std_dev:.4f}")
    print("\nIndividual Fold Accuracies:")
    for idx, acc in enumerate(accuracies):
        print(f"  Fold {idx + 1}: {acc:.4f}")

    avg_time_per_hash = (end_time - start_time) / len(all_pairs_data)
    print(f"⏱️ Average time per hash generation: {avg_time_per_hash:.4f} seconds")
    print("--------------------------------------------")
else:
    print("No pairs were processed to calculate accuracy.")
