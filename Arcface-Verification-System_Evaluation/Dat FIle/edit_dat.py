import numpy as np

# Load binary file (float32)
data = np.fromfile("..//models//neuralhash_128x96_seed1.dat", dtype=np.float32)

print("Total elements:", len(data))

# Save all values to a text file (one per line)
np.savetxt("neuralhash_128x96_seed1_values_float32.txt", data)

print("✅ All values exported to 'neuralhash_128x96_seed1_values_float32.txt'")
