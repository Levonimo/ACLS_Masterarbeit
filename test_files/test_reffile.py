import numpy as np
from fastdtw import fastdtw as dtw
from time import time

# Load folder of parent directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from component.fun_Warping import correlation_optimized_warping
# Load the warped and unwarped chromatograms
# The files are saved as numpy arrays with the chromatograms as values and the names as keys
unwarped = np.load('./output/unwarped_chromatograms.npy', allow_pickle=True).item()
unwarped = {k: unwarped[k] for k in list(unwarped.keys())[0:60]}
# print(unwarped)
names = np.load('./output/selected_target.npy', allow_pickle=True)
names = names[0:60]

compare_chromatograms = np.zeros((len(unwarped), len(unwarped)))

time_start = time()
# Warp each chromatograms with each other and compare the results
unwarped_values = list(unwarped.values())
for idx_i, i in enumerate(unwarped_values):
    for idx_j, j in enumerate(unwarped_values):
        if idx_i != idx_j:
            # Warp the chromatograms
            warped, _, _ = correlation_optimized_warping(i, j)
            # Compare the chromatograms
            distance, _ = dtw(i, warped)
            # Store the distance in the matrix
            compare_chromatograms[idx_i, idx_j] = distance
time_end = time()
print(f"Time taken for warping and comparison: {time_end - time_start:.2f} seconds")

# Plot the comparison as a heatmap
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(compare_chromatograms, ax=ax, xticklabels=names, yticklabels=names, cmap='viridis')
ax.set_title('Comparison of Unwarped Chromatograms')
plt.show()

