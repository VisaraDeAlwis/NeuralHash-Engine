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
    get_embedding, hash_embedding, calculate_hamming_distance,cosine_similarity
)

# ----------------- Constants -----------------
LFW_DIR = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled"
PAIRS_FILE = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\LFW\pairs_new.csv"
PCA_MODEL_PATH = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\models\pca_512_to_128.pkl"
HYPERPLANE_PATH = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\models\neuralhash_128x96_seed1.dat"
HYPERPLANE_PATH_128 =r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\Dat FIle\my_seed128.dat"
HYPERPLANE_PATH_512 = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\Dat FIle\my_seed512.dat"
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
    hyperplanes128 = load_hyperplanes(HYPERPLANE_PATH_128)
    #hyperplane512 = load_hyperplanes(HYPERPLANE_PATH_512)
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

# ----------------- Test Embedding -----------------
emb = get_embedding(r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg")
nhash = hash_embedding(emb, pca_model, hyperplanes)
nhash128 = hash_embedding(emb, pca_model, hyperplanes128)
print("Test hash:", nhash)
print("✅ Test passed.\n")

all_pairs_data =[]
print("Step 1: Pre-processing all pairs to generate hashes and distances...")

# ----------------- Step 1: Load or Generate Hash Cache -----------------
if os.path.exists(HASH_CACHE_FILE):
    print(f"📂 Loading cached hashes from {HASH_CACHE_FILE}...")
    with open(HASH_CACHE_FILE, "r") as f:
        hash_cache = json.load(f)
    print(f"✅ Loaded {len(hash_cache)} cached image hashes.\n")
else:
    print("⚙️ No cache file found. Will generate hashes for new images.\n")
    start_time = time.time()

    with open(PAIRS_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in tqdm(reader, desc="Processing Pairs"):
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

            try:
                emb1 = get_embedding(path1)
                hash1 = hash_embedding(emb1, pca_model, hyperplanes)
                hash1_128 =hash_embedding(emb1,pca_model,hyperplanes128)
                emb2 = get_embedding(path2)
                hash2 =hash_embedding(emb2, pca_model, hyperplanes)
                hash2_128 =hash_embedding(emb2, pca_model, hyperplanes128)
            except ValueError:
                continue
            dist = calculate_hamming_distance(hash1, hash2)
            dist_128 = calculate_hamming_distance(hash1_128, hash2_128)
            label = 1 if is_match else 0
           # print(f" Distance - {dist},cosine_similarity- {cosine_similarity(emb1,emb2)} - {label}")
            print(f" Distance hash1- {dist} - {label}")
            print(f" Distance hash1_128- {dist_128} - {label}")
            all_pairs_data.append({'distance': dist, 'label': label})

end_time = time.time()
print("Process Completed Successfully")

# ----------------- Step 5: Time Measurement -----------------
print("🎯 Processing completed successfully.\n")

# ----------------- Step 6: 10-Fold Cross Validation -----------------

# --- Step 2: Perform 10-Fold Cross-Validation ---
accuracies = []
# Ensure we have data to process
if all_pairs_data:
    fold_size = len(all_pairs_data) // 10
    print("\nStep 2: Starting 10-Fold Cross-Validation...")
    for i in tqdm(range(10), desc="Processing Folds"):
        # --- 2a: Split data into training (9 folds) and testing (1 fold) ---
        start, end = i * fold_size, (i + 1) * fold_size
        
        test_data = all_pairs_data[start:end]
        train_data = all_pairs_data[:start] + all_pairs_data[end:]

        # Extract distances and labels for each set
        train_distances = np.array([p['distance'] for p in train_data])
        train_labels = np.array([p['label'] for p in train_data])
        test_distances = np.array([p['distance'] for p in test_data])
        test_labels = np.array([p['label'] for p in test_data])
        
        # --- 2b: Find the best threshold ONLY on the training data for this fold ---
        best_fold_accuracy = 0
        best_fold_threshold = 0
        for threshold in range(len(hyperplanes) + 1):
            acc = calculate_accuracy(threshold, train_distances, train_labels)
            if acc > best_fold_accuracy:
                best_fold_accuracy = acc
                best_fold_threshold = threshold
                
        # --- 2c: Apply the best threshold to the unseen test data ---
        fold_accuracy = calculate_accuracy(best_fold_threshold, test_distances, test_labels)
        accuracies.append(fold_accuracy)

# --- Step 3: Calculate and display the final results ---
if accuracies:
    mean_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies)

    print("\n--- LFW 10-Fold Cross-Validation Results ---")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation: (+/-) {std_dev:.4f}")
    print("\nIndividual Fold Accuracies:")
    for idx, acc in enumerate(accuracies):
        print(f"  Fold {idx+1}: {acc:.4f}")
    
    avg_time_per_hash = (end_time - start_time) / len(all_pairs_data) 
    print(f"⏱️ Average time per hash generation: {avg_time_per_hash:.4f} seconds")
    
    print("--------------------------------------------")
else:
    print("No pairs were processed to calculate accuracy.")

