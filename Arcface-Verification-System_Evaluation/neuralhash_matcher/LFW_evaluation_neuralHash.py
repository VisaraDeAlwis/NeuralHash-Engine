import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv
import json
from sklearn.metrics import accuracy_score
from neuralhash_utils import load_pca_model, load_hyperplanes, generate_neuralhash, calculate_hamming_distance

# --- UTILITY FUNCTIONS (Unchanged) ---
def get_image_path(lfw_dir, person, image_num):
    """Constructs the full path for a given LFW image."""
    return os.path.join(lfw_dir, person, f"{person}_{int(image_num):04d}.jpg")

def calculate_accuracy(threshold, distances, labels):
    """Calculates accuracy for a given threshold."""
    predictions = (distances <= threshold).astype(int)
    return accuracy_score(labels, predictions)

# --- MAIN EVALUATION LOGIC ---

# Define constants
LFW_DIR = "D://FYP//Madusha_ArcFace//Arcface-Verification-System//datasets//LFW//lfw-deepfunneled//lfw-deepfunneled"
PAIRS_FILE = "D://FYP//Madusha_ArcFace//Arcface-Verification-System//datasets//LFW//pairs_new.csv"
PCA_MODEL_PATH = 'D://FYP\Madusha_ArcFace//Arcface-Verification-System//models//pca_512_to_128.pkl'
HYPERPLANE_PATH = 'D://FYP//Madusha_ArcFace//Arcface-Verification-System//models//neuralhash_128x96_seed1.dat'
# --- NEW: Define path for the hash cache file ---
HASH_CACHE_FILE = 'neuralhashes_cache.json'

# --- Load Models ---
try:
    pca_model = load_pca_model(PCA_MODEL_PATH)
    hyperplanes = load_hyperplanes(HYPERPLANE_PATH)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# --- Step 1: Generate or Load NeuralHashes ---
hashes = {}
if os.path.exists(HASH_CACHE_FILE):
    print(f"Loading cached hashes from {HASH_CACHE_FILE}...")
    with open(HASH_CACHE_FILE, 'r') as f:
        hashes = json.load(f)
    print("Hashes loaded successfully.")
else:
    print("No cache file found. Generating hashes for all unique images...")
    # First, find all unique images that need to be processed
    unique_images = set()
    with open(PAIRS_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            if len(row) == 4 and all(row):
                unique_images.add(get_image_path(LFW_DIR, row[0], row[1]))
                unique_images.add(get_image_path(LFW_DIR, row[2], row[3]))
            elif len(row) == 3 and all(row):
                unique_images.add(get_image_path(LFW_DIR, row[0], row[1]))
                unique_images.add(get_image_path(LFW_DIR, row[0], row[2]))
    
    # Now, generate hash for each unique image
    for image_path in tqdm(sorted(list(unique_images)), desc="Generating Hashes"):
        try:
            bits = generate_neuralhash(image_path, pca_model, hyperplanes)
            # JSON can't store numpy arrays, so convert to a standard list
            hashes[image_path] = bits.tolist()
        except ValueError:
            # If a face can't be found, store None so we know to skip it later
            hashes[image_path] = None
    
    # Save the generated hashes to the cache file
    print(f"Saving hashes to {HASH_CACHE_FILE}...")
    with open(HASH_CACHE_FILE, 'w') as f:
        json.dump(hashes, f)
    print("Cache file created successfully.")

# --- Step 2: Pre-calculate all distances using the hashes ---
all_pairs_data = []
print("\nStep 2: Calculating distances using hashes...")
with open(PAIRS_FILE, 'r') as f:
    reader = csv.reader(f)
    next(reader)

    for row in tqdm(reader, desc="Calculating Distances"):
        is_match = False
        if len(row) == 4 and all(row):
            path1 = get_image_path(LFW_DIR, row[0], row[1])
            path2 = get_image_path(LFW_DIR, row[2], row[3])
            is_match = False
        elif len(row) == 3 and all(row):
            path1 = get_image_path(LFW_DIR, row[0], row[1])
            path2 = get_image_path(LFW_DIR, row[0], row[2])
            is_match = True
        else:
            continue
        
        # Get hashes from our dictionary
        bits1 = hashes.get(path1)
        bits2 = hashes.get(path2)
        
        if bits1 is not None and bits2 is not None:
            # Convert back to numpy array for calculation
            dist = calculate_hamming_distance(np.array(bits1), np.array(bits2))
            label = 1 if is_match else 0
            all_pairs_data.append({'distance': dist, 'label': label})

print(f"\nSuccessfully calculated distances for {len(all_pairs_data)} pairs.")

# --- Step 3: Perform 10-Fold Cross-Validation ---
accuracies = []
if all_pairs_data:
    fold_size = len(all_pairs_data) // 10
    print("\nStep 3: Starting 10-Fold Cross-Validation...")
    for i in tqdm(range(10), desc="Processing Folds"):
        # Split data into training and testing sets
        start, end = i * fold_size, (i + 1) * fold_size
        test_data = all_pairs_data[start:end]
        train_data = all_pairs_data[:start] + all_pairs_data[end:]

        train_distances = np.array([p['distance'] for p in train_data])
        train_labels = np.array([p['label'] for p in train_data])
        test_distances = np.array([p['distance'] for p in test_data])
        test_labels = np.array([p['label'] for p in test_data])
        
        # Find the best threshold ONLY on the training data for this fold
        best_fold_accuracy = 0
        best_fold_threshold = 0
        for threshold in range(len(hyperplanes) + 1):
            acc = calculate_accuracy(threshold, train_distances, train_labels)
            if acc > best_fold_accuracy:
                best_fold_accuracy = acc
                best_fold_threshold = threshold
        print(f"  Fold {i+1}: Best Threshold = {best_fold_threshold}, Training Accuracy = {best_fold_accuracy:.4f}")
                
        # Apply the best threshold to the unseen test data
        fold_accuracy = calculate_accuracy(best_fold_threshold, test_distances, test_labels)
        accuracies.append(fold_accuracy)

# --- Step 4: Calculate and display the final results ---
if accuracies:
    mean_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies)

    print("\n--- LFW 10-Fold Cross-Validation Results ---")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation: (+/-) {std_dev:.4f}")
    print("\nIndividual Fold Accuracies:")
    for idx, acc in enumerate(accuracies):
        print(f"  Fold {idx+1}: {acc:.4f}")
    print("--------------------------------------------")
else:
    print("No pairs were processed to calculate accuracy.")