import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv
import time
import json
from sklearn.metrics import accuracy_score
# Make sure your utility file is accessible
from neuralhash_utils_final import (
    load_pca_model, load_hyperplanes,
    get_embedding, hash_embedding, calculate_hamming_distance,
    cosine_similarity, load_hyperplanes_512, hash_embedding_512
)

# ----------------- Constants -----------------
LFW_DIR = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\LFW\lfw-deepfunneled\lfw-deepfunneled"
PAIRS_FILE = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\LFW\pairs_new.csv"
PCA_MODEL_PATH = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\models\pca_512_to_128.pkl"
HYPERPLANE_PATH = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\models\neuralhash_128x96_seed1.dat"
HYPERPLANE_PATH_128 = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\Dat FIle\my_seed128.dat"
HYPERPLANE_PATH_512 = r"D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\Dat FIle\my_seed512.dat"
EMBEDDING_CACHE_FILE = "embeddings_cache.json"

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

# ----------------- Stage 1: Generate and Cache Embeddings -----------------
print("\n--- Stage 1: Caching Embeddings ---")
embedding_cache = {}
if os.path.exists(EMBEDDING_CACHE_FILE):
    print(f"📂 Loading existing embedding cache from {EMBEDDING_CACHE_FILE}...")
    with open(EMBEDDING_CACHE_FILE, "r") as f:
        embedding_cache = json.load(f)
    print(f"✅ Loaded {len(embedding_cache)} cached embeddings.")

# Find all unique image paths required for the evaluation
unique_image_paths = set()
print("🔍 Scanning pairs file for unique images...")
with open(PAIRS_FILE, "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if len(row) == 4 and all(row):
            person1, image_num1, person2, image_num2 = row
            unique_image_paths.add(get_image_path(LFW_DIR, person1, image_num1))
            unique_image_paths.add(get_image_path(LFW_DIR, person2, image_num2))
        elif len(row) == 3 and all(row):
            person1, image_num1, image_num2 = row
            unique_image_paths.add(get_image_path(LFW_DIR, person1, image_num1))
            unique_image_paths.add(get_image_path(LFW_DIR, person1, image_num2))

# Generate embeddings for images not already in the cache
new_embeddings_generated = 0
paths_to_process = [p for p in unique_image_paths if p not in embedding_cache]

if paths_to_process:
    print(f"⚙️ Generating new embeddings for {len(paths_to_process)} images...")
    for path in tqdm(paths_to_process, desc="Generating Embeddings"):
        try:
            if os.path.exists(path):
                emb = get_embedding(path)
                # Convert numpy array to list for JSON serialization
                embedding_cache[path] = emb.tolist()
                new_embeddings_generated += 1
            else:
                print(f"⚠️ Warning: Image path not found: {path}")
        except Exception as e:
            print(f"❌ Error generating embedding for {path}: {e}")

    # Save the updated cache to disk
    if new_embeddings_generated > 0:
        print(f"💾 Saving {new_embeddings_generated} new embeddings to {EMBEDDING_CACHE_FILE}...")
        with open(EMBEDDING_CACHE_FILE, "w") as f:
            json.dump(embedding_cache, f, indent=4)
        print("✅ Cache updated successfully.")
else:
    print("✅ All required embeddings are already cached.")


# ----------------- Stage 2: Process Pairs Using Cached Embeddings -----------------
print("\n--- Stage 2: Processing Pairs and Evaluating ---")
all_pairs_data = []
all_pairs_data_128 =[]
all_pairs_data_512 = []
start_time = time.time()

with open(PAIRS_FILE, "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header

    for row in tqdm(reader, desc="Processing Pairs with Cached Embeddings"):
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

        # Retrieve embeddings from cache
        emb1_list = embedding_cache.get(path1)
        emb2_list = embedding_cache.get(path2)

        if emb1_list is None or emb2_list is None:
            print(f"⚠️ Warning: Embedding not found in cache for pair: ({path1}, {path2}). Skipping.")
            continue

        # Convert lists back to numpy arrays
        emb1 = np.array(emb1_list)
        emb2 = np.array(emb2_list)

        # --- Hash Calculation ---
        hash1 = hash_embedding(emb1, pca_model, hyperplanes)
        hash1_128 = hash_embedding(emb1, pca_model, hyperplanes128)
        hash1_512 = hash_embedding_512(emb1, hyperplanes512)

        hash2 = hash_embedding(emb2, pca_model, hyperplanes)
        hash2_128 = hash_embedding(emb2, pca_model, hyperplanes128)
        hash2_512 = hash_embedding_512(emb2, hyperplanes512)
        
        # --- Distance Calculation ---
        dist = calculate_hamming_distance(hash1, hash2)
        dist_128 = calculate_hamming_distance(hash1_128, hash2_128)
        dist_512 = calculate_hamming_distance(hash1_512, hash2_512)
        cos_sim = cosine_similarity(emb1, emb2)
        label = 1 if is_match else 0
        
        # You can uncomment these lines for debugging a few pairs
        # print(f"\nCosine Similarity: {cos_sim:.4f} - Label: {label}")
        # print(f"Hamming Distance (Apple): {dist} - Label: {label}")
        # print(f"Hamming Distance (128-bit): {dist_128} - Label: {label}")
        # print(f"Hamming Distance (512-bit): {dist_512} - Label: {label}")

        # Storing data for Apple's hash for the 10-fold validation
        all_pairs_data.append({'distance': dist, 'label': label})
        all_pairs_data_128.append({'distance': dist_128, 'label': label})
        all_pairs_data_512.append({'distance': dist_512, 'label': label})

end_time = time.time()
processing_duration = end_time - start_time
print("🎯 Pair processing completed successfully.\n")


# ----------------- Stage 3: 10-Fold Cross Validation -----------------
#Accuracy score for Original Seed.dat file with ResNet100

accuracies = []
if all_pairs_data:
    fold_size = len(all_pairs_data) // 10
    print("--- Stage 3: Starting 10-Fold Cross-Validation ---")
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
        # Iterate through possible thresholds (0 to 96 for a 96-bit hash)
        for threshold in range(len(hyperplanes) + 1):
            acc = calculate_accuracy(threshold, train_distances, train_labels)
            if acc > best_fold_accuracy:
                best_fold_accuracy = acc
                best_fold_threshold = threshold
                
        fold_accuracy = calculate_accuracy(best_fold_threshold, test_distances, test_labels)
        accuracies.append(fold_accuracy)

# --- Final Results ---
if accuracies:
    mean_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies)

    print("\n--- LFW 10-Fold Cross-Validation Results (Using Apple NeuralHash) ---")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation: (+/-) {std_dev:.4f}")
    print("\nIndividual Fold Accuracies:")
    for idx, acc in enumerate(accuracies):
        print(f"  Fold {idx+1}: {acc:.4f}")
    
    avg_time_per_pair = processing_duration / len(all_pairs_data) if all_pairs_data else 0
    print(f"\n⏱️  Average processing time per pair (excluding embedding): {avg_time_per_pair:.6f} seconds")
    
    print("----------------------------------------------------------------------")
else:
    print("No pairs were processed to calculate accuracy.")

#for 128 generated dat file
print("Evaluations for 128*96 generated dat")


accuracies = []
if all_pairs_data_128:
    fold_size = len(all_pairs_data_128) // 10
    print("--- Stage 3: Starting 10-Fold Cross-Validation ---")
    for i in tqdm(range(10), desc="Processing Folds"):
        start, end = i * fold_size, (i + 1) * fold_size

        test_data = all_pairs_data_128[start:end]
        train_data = all_pairs_data_128[:start] + all_pairs_data_128[end:]

        train_distances = np.array([p['distance'] for p in train_data])
        train_labels = np.array([p['label'] for p in train_data])
        test_distances = np.array([p['distance'] for p in test_data])
        test_labels = np.array([p['label'] for p in test_data])
        
        best_fold_accuracy = 0
        best_fold_threshold = 0
        # Iterate through possible thresholds (0 to 96 for a 96-bit hash)
        for threshold in range(len(hyperplanes) + 1):
            acc = calculate_accuracy(threshold, train_distances, train_labels)
            if acc > best_fold_accuracy:
                best_fold_accuracy = acc
                best_fold_threshold = threshold
                
        fold_accuracy = calculate_accuracy(best_fold_threshold, test_distances, test_labels)
        accuracies.append(fold_accuracy)

# --- Final Results ---
if accuracies:
    mean_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies)

    print("\n--- LFW 10-Fold Cross-Validation Results (Using Apple NeuralHash) ---")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation: (+/-) {std_dev:.4f}")
    print("\nIndividual Fold Accuracies:")
    for idx, acc in enumerate(accuracies):
        print(f"  Fold {idx+1}: {acc:.4f}")
    
    avg_time_per_pair = processing_duration / len(all_pairs_data) if all_pairs_data else 0
    print(f"\n⏱️  Average processing time per pair (excluding embedding): {avg_time_per_pair:.6f} seconds")
    
    print("----------------------------------------------------------------------")
else:
    print("No pairs were processed to calculate accuracy.")

#for generated 512*96 dat


print("Evaluations for 512*96 generated dat")

accuracies = []
if all_pairs_data_512:
    fold_size = len(all_pairs_data_512) // 10
    print("--- Stage 3: Starting 10-Fold Cross-Validation ---")
    for i in tqdm(range(10), desc="Processing Folds"):
        start, end = i * fold_size, (i + 1) * fold_size

        test_data = all_pairs_data_512[start:end]
        train_data = all_pairs_data_512[:start] + all_pairs_data_512[end:]

        train_distances = np.array([p['distance'] for p in train_data])
        train_labels = np.array([p['label'] for p in train_data])
        test_distances = np.array([p['distance'] for p in test_data])
        test_labels = np.array([p['label'] for p in test_data])
        
        best_fold_accuracy = 0
        best_fold_threshold = 0
        # Iterate through possible thresholds (0 to 96 for a 96-bit hash)
        for threshold in range(len(hyperplanes) + 1):
            acc = calculate_accuracy(threshold, train_distances, train_labels)
            if acc > best_fold_accuracy:
                best_fold_accuracy = acc
                best_fold_threshold = threshold
                
        fold_accuracy = calculate_accuracy(best_fold_threshold, test_distances, test_labels)
        accuracies.append(fold_accuracy)

# --- Final Results ---
if accuracies:
    mean_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies)

    print("\n--- LFW 10-Fold Cross-Validation Results (Using Apple NeuralHash) ---")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation: (+/-) {std_dev:.4f}")
    print("\nIndividual Fold Accuracies:")
    for idx, acc in enumerate(accuracies):
        print(f"  Fold {idx+1}: {acc:.4f}")
    
    avg_time_per_pair = processing_duration / len(all_pairs_data) if all_pairs_data else 0
    print(f"\n⏱️  Average processing time per pair (excluding embedding): {avg_time_per_pair:.6f} seconds")
    
    print("----------------------------------------------------------------------")
else:
    print("No pairs were processed to calculate accuracy.")
