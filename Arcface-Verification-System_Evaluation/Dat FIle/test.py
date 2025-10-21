import numpy as np
def load_hyperplanes(path):
    with open(path, "rb") as f:
        f.seek(32)  # Skip the first 32 bytes of metadata
        arr = np.fromfile(f, dtype=np.int8, count=128*96)
    arr = arr.reshape(96, 128).astype(np.float32)
    return arr

a= load_hyperplanes("..//models//neuralhash_128x96_seed1.dat")
print(a)