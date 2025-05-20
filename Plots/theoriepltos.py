import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# Daten generieren
X,y = make_blobs(n_samples=300,centers=[[1, 0.], [5, 5], [10, 0]], cluster_std=[0.5, 1.1, 0.7],  random_state=42)

# Funktionen zur Berechnung der Metriken
def compute_dunn_index(X, labels):
    unique_clusters = np.unique(labels)
    intra_dists = []
    inter_dists = []

    for i in unique_clusters:
        cluster_i = X[labels == i]
        if len(cluster_i) > 1:
            intra = np.max(pairwise_distances(cluster_i))
            intra_dists.append(intra)
        else:
            intra_dists.append(0)

    for i in unique_clusters:
        for j in unique_clusters:
            if i < j:
                cluster_i = X[labels == i]
                cluster_j = X[labels == j]
                inter = np.min(pairwise_distances(cluster_i, cluster_j))
                inter_dists.append(inter)

    return np.min(inter_dists) / np.max(intra_dists)

# Werte vorbereiten
k_values = np.arange(2, 9)
dunn_values = []
gap_values = []




# Für Gap simulieren wir einfache Beispielwerte
simulated_gap = [0.25, 0.38, 0.50, 0.55, 0.52, 0.48, 0.45]

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    dunn = compute_dunn_index(X, kmeans.labels_)
    dunn_values.append(dunn)

# Plot 1: Dunn-Index Illustration
# Set a consistent figure size for both plots
figsize = (8, 5)

# Plot 1: Dunn-Index Illustration
fig1, ax1 = plt.subplots(figsize=figsize)
scatter = ax1.scatter(X[:, 0], X[:, 1], cmap='tab10', s=30)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()

# Maxima bestimmen
optimal_k_dunn = k_values[np.argmax(dunn_values)]
optimal_k_gap = k_values[np.argmax(simulated_gap)]

# Plot 2
fig, ax = plt.subplots(figsize=figsize)
ax.plot(k_values, dunn_values, marker='o', label='Dunn-Index')
ax.plot(k_values, simulated_gap, marker='s', label='Gap-Statistics')
ax.axvline(optimal_k_dunn, color='tab:blue', linestyle='--', alpha=0.6)
ax.axvline(optimal_k_gap, color='tab:orange', linestyle='--', alpha=0.6)
ax.scatter(optimal_k_dunn, max(dunn_values), color='tab:blue', zorder=5)
ax.scatter(optimal_k_gap, max(simulated_gap), color='tab:orange', zorder=5)
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Value')
ax.legend()
ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()
