import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv
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
LFW_DIR = "D://FYP//Madusha_ArcFace_Evaluation//Arcface-Verification-System_Evaluation//datasets//LFW//lfw-deepfunneled//lfw-deepfunneled"
PAIRS_FILE = "D://FYP//Madusha_ArcFace_Evaluation//Arcface-Verification-System_Evaluation//datasets//LFW//pairs_new.csv"
PCA_MODEL_PATH = 'D://FYP//Madusha_ArcFace_Evaluation//Arcface-Verification-System_Evaluation//models//pca_512_to_128.pkl'
HYPERPLANE_PATH = 'D://FYP//Madusha_ArcFace_Evaluation//Arcface-Verification-System_Evaluation//models//neuralhash_128x96_seed1.dat'

# --- Load Models ---
try:
    pca_model = load_pca_model(PCA_MODEL_PATH)
    hyperplanes = load_hyperplanes(HYPERPLANE_PATH)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# --- Step 1: Pre-calculate all distances to avoid redundant processing ---
all_pairs_data = []
print("Step 1: Pre-processing all pairs to generate hashes and distances...")
with open(PAIRS_FILE, 'r') as f:
    reader = csv.reader(f)
    next(reader) # Skip header

    for row in tqdm(reader, desc="Generating Hashes"):
        # Simplified parsing logic for the cleaned file
        if len(row) == 4 and all(row):
            person1, img_num1, person2, img_num2 = row
            is_match = False
        elif len(row) == 3 and all(row):
            person1, img_num1, img_num2 = row
            person2 = person1
            is_match = True
        else:
            continue

        path1 = get_image_path(LFW_DIR, person1, img_num1)
        path2 = get_image_path(LFW_DIR, person2, img_num2)

        try:
            bits1 = generate_neuralhash(path1, pca_model, hyperplanes)
            bits2 = generate_neuralhash(path2, pca_model, hyperplanes)
        except ValueError:
            continue
        
        dist = calculate_hamming_distance(bits1, bits2)
        label = 1 if is_match else 0
        all_pairs_data.append({'distance': dist, 'label': label})

print(f"\nSuccessfully pre-processed {len(all_pairs_data)} pairs.")

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
    print("--------------------------------------------")
else:
    print("No pairs were processed to calculate accuracy.")