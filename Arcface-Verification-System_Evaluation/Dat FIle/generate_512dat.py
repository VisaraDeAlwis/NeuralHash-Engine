import numpy as np

def generate_hyperplanes(seed=42):
    np.random.seed(seed)
    # 96 hyperplanes in 512D space
    hyperplanes = np.random.randn(96, 512).astype(np.float32)
    
    # (optional) Normalize each hyperplane vector
    hyperplanes /= np.linalg.norm(hyperplanes, axis=1, keepdims=True)
    return hyperplanes

def save_hyperplanes(hyperplanes, path="my_seed512.dat"):
    hyperplanes.astype(np.float32).tofile(path)
    print(f"Saved {hyperplanes.shape} hyperplanes to {path}")

H = generate_hyperplanes().T  # shape (512, 96)
H.astype(np.float32).tofile("my_seed512.dat")
print("✅ Hyperplanes saved to 'my_seed.dat'")
