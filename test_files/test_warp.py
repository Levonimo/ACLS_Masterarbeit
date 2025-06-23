import numpy as np
from fastdtw import fastdtw as dtw

# Load folder of parent directory
import sys
sys.path.append('..')
# Load the warped and unwarped chromatograms
# The files are saved as numpy arrays with the chromatograms as values and the names as keys

warped = np.load('./output/warped_chromatograms.npy', allow_pickle=True).item()
# only chromatograms from 30th to 60th key, key is the name of the chromatogram as string
warped = {k: warped[k] for k in list(warped.keys())[0:60]}

# print(warped)
unwarped = np.load('./output/unwarped_chromatograms.npy', allow_pickle=True).item()
# only chromatograms from 30th to 60th key, key is the name of the chromatogram as string
unwarped = {k: unwarped[k] for k in list(unwarped.keys())[0:60]}


# print(unwarped)
names = np.load('./output/selected_target.npy', allow_pickle=True)
# only names from 30th to 60th key, key is the name of the chromatogram as string
names = names[0:60]

# Compare each chromatogram with each other
def compare_chromatograms(chromatograms):
    """
    Compare each chromatogram with each other: Calculate the euclidean distance between each chromatogram and 
    return a matrix with the distances.
    """
    # set base of chromatogramm on 0
    
    chromatograms = [chromatogram - np.min(chromatogram) for chromatogram in chromatograms]


    n = len(chromatograms)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # distances[i, j] = np.linalg.norm(chromatograms[i] - chromatograms[j]) # Euclidean distance
            # distances[i, j] = np.sum(np.abs(chromatograms[i] - chromatograms[j])) # Manhattan distance
            distances[i, j], _ = dtw(chromatograms[i], chromatograms[j])  # Dynamic Time Warping distance
            #distances[i, j] = np.sum((chromatograms[i] - chromatograms[j])**2) # Squared Euclidean distance
            #distances[i, j] = 1 - np.corrcoef(chromatograms[i].flatten(), chromatograms[j].flatten())[0, 1] # Correlation distance
    return distances

# Compare the warped chromatograms
warped_distances = compare_chromatograms(list(warped.values()))

# Compare the unwarped chromatograms
unwarped_distances = compare_chromatograms(list(unwarped.values()))


difference = np.abs(warped_distances - unwarped_distances)

# Plot the comparison as a hetmap
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(12, 6))
sns.heatmap(warped_distances, ax=ax[0], xticklabels=names, yticklabels=names, cmap='viridis')
ax[0].set_title('Warped Chromatograms')
sns.heatmap(unwarped_distances, ax=ax[1], xticklabels=names, yticklabels=names, cmap='viridis')
ax[1].set_title('Unwarped Chromatograms')
sns.heatmap(difference, ax=ax[2], xticklabels=names, yticklabels=names, cmap='viridis')
ax[2].set_title('Difference')

plt.show()
    