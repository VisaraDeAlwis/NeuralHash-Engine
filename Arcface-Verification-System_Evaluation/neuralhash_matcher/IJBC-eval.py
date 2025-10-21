import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv
import collections
from sklearn.metrics import accuracy_score
from neuralhash_utils_ijbc import load_pca_model, load_hyperplanes, get_embedding, hash_embedding, calculate_hamming_distance

# --- UTILITY FUNCTIONS (Unchanged) ---

def get_image_path(lfw_dir, person, image_num):
    """Constructs the full path for a given LFW image."""
    return os.path.join(lfw_dir, person, f"{person}_{int(image_num):04d}.jpg")

def calculate_accuracy(threshold, distances, labels):
    """Calculates accuracy for a given threshold."""
    predictions = (distances <= threshold).astype(int)
    return accuracy_score(labels, predictions)

def calculate_tar_far(threshold, distances, labels):
    """Calculates TAR and FAR for a given threshold."""
    predictions = (distances <= threshold)
    
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    
    genuine_pairs = np.sum(labels == 1)
    imposter_pairs = np.sum(labels == 0)
    
    tar = true_positives / genuine_pairs if genuine_pairs > 0 else 0
    far = false_positives / imposter_pairs if imposter_pairs > 0 else 0
    
    return tar, far

# --- MAIN EVALUATION LOGIC ---

# Define constants
IJB_DIR = "D:\FYP\Madusha_ArcFace_Evaluation\Arcface-Verification-System_Evaluation\datasets\ijb-testsuite\ijb\IJBC"
META_DIR = os.path.join(IJB_DIR, "meta")
MASTER_ROSTER_FILE = os.path.join(IJB_DIR, "meta", "ijbc_face_tid_mid.txt")
PAIR_LABEL_FILE = os.path.join(IJB_DIR, "meta", "ijbc_template_pair_label.txt")

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

#Build the Template Map ---
print("Step 1: Building template map from master roster file...")
template_map = collections.defaultdict(list)
with open(MASTER_ROSTER_FILE, 'r') as f:
    for line in f:
        # The file is space-separated: filepath mid tid sid
        filename, subject_id, template_id = line.strip().split()
        # IMPORTANT: Verify this 'loose_crop' path matches your folder structure
        full_image_path = os.path.join(IJB_DIR,"loose_crop", filename) 
        template_map[int(template_id)].append(full_image_path)
print(f"Found {len(template_map)} templates.")


# --- Step 2: Generate Averaged Template Embeddings ---
print("\nStep 2: Generating features for each template...")
template_embeddings = {}
for template_id, image_paths in tqdm(template_map.items(), desc="Processing Templates"):
    embeddings_for_template = [emb for path in image_paths if (emb := get_embedding(path)) is not None]
            
    if embeddings_for_template:
        avg_embedding = np.mean(embeddings_for_template, axis=0)
        template_embeddings[template_id] = avg_embedding

# --- Step 3: Hash the Template Embeddings ---
print("\nStep 3: Hashing the averaged template embeddings...")
template_hashes = {}
for template_id, avg_embedding in tqdm(template_embeddings.items(), desc="Hashing Templates"):
    template_hashes[template_id] = hash_embedding(avg_embedding, pca_model, hyperplanes)

# --- Step 4: Perform 1:1 Verification ---
print("\nStep 4: Performing 1:1 verification...")
distances = []
labels = []
with open(PAIR_LABEL_FILE, 'r') as f:
    for line in tqdm(f, desc="Comparing Pairs"):
        # The file is space-separated: tid1 tid2 label
        tid1_str, tid2_str, label_str = line.strip().split()
        tid1, tid2, label = int(tid1_str), int(tid2_str), int(label_str)
        
        hash1 = template_hashes.get(tid1)
        hash2 = template_hashes.get(tid2)
        
        if hash1 is not None and hash2 is not None:
            dist = calculate_hamming_distance(hash1, hash2)
            distances.append(dist)
            labels.append(label)

# --- Step 5: Calculate and Report TAR @ FAR ---
print("\nStep 5: Calculating TAR at various FARs...")
if distances:
    distances = np.array(distances)
    labels = np.array(labels)
    
    # Calculate TAR/FAR for all possible thresholds (0-96 for Hamming distance)
    all_rates = []
    for threshold in range(len(hyperplanes) + 1):
        tar, far = calculate_tar_far(threshold, distances, labels)
        all_rates.append((far, tar))
    
    # Sort rates by False Accept Rate
    all_rates.sort(key=lambda x: x[0])
    
    fars, tars = zip(*all_rates)

    # Find the TAR for standard FAR points
    far_targets = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    results = {}
    for far_target in far_targets:
        # Find the highest TAR where the FAR is at or below the target
        valid_indices = np.where(np.array(fars) <= far_target)[0]
        if len(valid_indices) > 0:
            best_tar = np.max(np.array(tars)[valid_indices])
            results[far_target] = best_tar
        else:
            results[far_target] = 0.0 # No threshold achieves this low FAR

    print("\n--- IJB-B 1:1 Verification Results ---")
    for far_val, tar_val in results.items():
        print(f"TAR @ FAR={far_val:.0e}: {tar_val:.4f}")
    print("--------------------------------------")
else:
    print("No pairs were processed.")
