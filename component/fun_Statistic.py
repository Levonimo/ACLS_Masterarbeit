import os
os.environ["OMP_NUM_THREADS"] = "1"


import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
from scipy.spatial.distance import euclidean
from itertools import combinations



def preprocess_data(df: pd.DataFrame) -> tuple:
    """Extract numerical PCA values from a DataFrame.

    -------
    Parameter:
        df : pd.DataFrame --> Table containing PCA columns

    Output:
        pc_data : np.ndarray --> Numerical PCA array
        sample_names : list --> Names of each sample
    """
    # Select only numerical PCA columns (assuming they start with 'PC')
    pc_data = df.select_dtypes(exclude=['category']).values
    sample_names = df.index.tolist()  # Use index as sample names
    return pc_data, sample_names

def compute_cophenetic_correlation(pc_data):
    """Calculate the cophenetic correlation coefficient.

    -------
    Parameter:
        pc_data : np.array --> PCA score matrix

    Output:
        coph_corr : float --> Cophenetic correlation coefficient
    """
    dist_matrix = pdist(pc_data, metric='euclidean')
    linkage_matrix = linkage(dist_matrix, method='ward')
    coph_corr, _ = cophenet(linkage_matrix, dist_matrix)
    return coph_corr

def gap_statistic(pc_data, n_refs=10, max_clusters=15):
    """Compute optimal cluster count using the Gap statistic.

    -------
    Parameter:
        pc_data : np.array --> PCA score matrix
        n_refs : int --> Number of reference reference sets
        max_clusters : int --> Maximum cluster count

    Output:
        optimal_clusters : int --> Estimated best cluster number
    """
    eps = 1e-10
    gaps = np.zeros(max_clusters)
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(pc_data)
        # Replace inertia with epsilon if it's zero
        inertia = kmeans.inertia_ if kmeans.inertia_ > 0 else eps
        ref_disps = np.zeros(n_refs)
        
        for i in range(n_refs-1):
            random_ref = np.random.uniform(pc_data.min(axis=0), pc_data.max(axis=0), size=pc_data.shape)
            ref_kmeans = KMeans(n_clusters=k, random_state=42).fit(random_ref)
            # Replace reference inertia with epsilon if it's zero
            ref_disp = ref_kmeans.inertia_ if ref_kmeans.inertia_ > 0 else eps
            ref_disps[i] = ref_disp
        
        gaps[k - 1] = np.mean(np.log(ref_disps)) - np.log(inertia)

    optimal_clusters = np.argmax(gaps) + 1  # Add 1 because index starts from 0
    return optimal_clusters

def compute_cluster_stability(pc_data, optimal_clusters, num_bootstraps=100):
    """Measure cluster stability using bootstrapping.

    -------
    Parameter:
        pc_data : np.array --> PCA score matrix
        optimal_clusters : int --> Chosen cluster count
        num_bootstraps : int --> Number of bootstrap iterations

    Output:
        stability_score : float --> Averaged variance of bootstrap labels
    """
    if optimal_clusters < 2:
        optimal_clusters = 2
        Warning(f"Optimal clusters set to {optimal_clusters} as it was less than 2.")
    bootstrap_labels = []
    for _ in range(num_bootstraps):
        sample_data = resample(pc_data, random_state=42)
        linkage_matrix = linkage(pdist(sample_data), method='ward')
        cluster_labels = fcluster(linkage_matrix, t=optimal_clusters, criterion='maxclust')
        bootstrap_labels.append(cluster_labels)

    return np.mean(np.var(bootstrap_labels, axis=0))

def compute_silhouette_score(pc_data, optimal_clusters):
    """Compute the silhouette score for hierarchical clusters.

    -------
    Parameter:
        pc_data : np.array --> PCA score matrix
        optimal_clusters : int --> Number of clusters

    Output:
        score : float --> Silhouette score
    """
    linkage_matrix = linkage(pdist(pc_data), method='ward')
    cluster_labels = fcluster(linkage_matrix, t=optimal_clusters, criterion='maxclust')
    return silhouette_score(pc_data, cluster_labels)

def compute_dunn_index(pc_data, min_clusters=2, max_clusters=15):
    """Find optimal cluster count using the Dunn index.

    -------
    Parameter:
        pc_data : np.array --> PCA score matrix
        min_clusters : int --> Minimum number of clusters
        max_clusters : int --> Maximum number of clusters

    Output:
        (optimal_clusters, best_dunn) : tuple --> Best cluster count and index
    """
    best_dunn = -np.inf
    optimal_clusters = min_clusters

    for k in range(min_clusters, max_clusters + 1):
        # Compute cluster labels using hierarchical clustering
        linkage_matrix = linkage(pdist(pc_data), method='ward')
        cluster_labels = fcluster(linkage_matrix, t=k, criterion='maxclust')
        unique_clusters = np.unique(cluster_labels)
        intra_dists = []
        inter_dists = []

        # Compute maximum intra-cluster distances for each cluster
        for clust in unique_clusters:
            cluster_points = pc_data[cluster_labels == clust]
            if cluster_points.shape[0] < 2:
                intra_dists.append(0)
            else:
                distances = [euclidean(x, y) for x, y in combinations(cluster_points, 2)]
                intra_dists.append(np.max(distances))

        # Compute minimum inter-cluster distances for each pair of clusters
        for (cl1, cl2) in combinations(unique_clusters, 2):
            cluster1 = pc_data[cluster_labels == cl1]
            cluster2 = pc_data[cluster_labels == cl2]
            inter_dists.append(np.min([euclidean(x, y) for x in cluster1 for y in cluster2]))

        max_intra = np.max(intra_dists)
        # Prevent division by zero
        if max_intra == 0:
            current_dunn = float('inf')
        else:
            current_dunn = np.min(inter_dists) / max_intra

        if current_dunn > best_dunn:
            best_dunn = current_dunn
            optimal_clusters = k

    return optimal_clusters, best_dunn

def analyze_clustering(df):
    """
    Runs all clustering significance checks and prints results.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing PCA scores.
    """
    # Preprocess data
    pc_data, sample_names = preprocess_data(df)

    # Compute statistics
    cophenetic_corr = compute_cophenetic_correlation(pc_data)
    optimal_clusters = gap_statistic(pc_data)
    stability_score = compute_cluster_stability(pc_data, optimal_clusters)
    silhouette = compute_silhouette_score(pc_data, optimal_clusters)
    optimal_clusters_dunn, dunn_index = compute_dunn_index(pc_data)

    # Print results
    print(f"Cophenetic Correlation: {cophenetic_corr:.4f}")
    print(f"Optimal Clusters (Gap Statistic): {optimal_clusters}")
    print(f"Cluster Stability Score: {stability_score:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Dunn Index: {dunn_index:.4f} (Optimal Clusters: {optimal_clusters_dunn})")

# # If this file is run directly, you can load test data here
# if __name__ == "__main__":
#     # Example test data
#     data = {
#         "PC1": np.random.randn(10),
#         "PC2": np.random.randn(10),
#         "PC3": np.random.randn(10),
#         "PC4": np.random.randn(10),
#         "PC5": np.random.randn(10),
#         "Group 1": ["A1"] * 10,
#         "Group 2": list(range(1, 11)),
#         "Group 3": ["SGO", "SOL", "SGL", "SOO", "SGL", "SGO", "SOL", "SGL", "SOO", "SGO"]
#     }
#     df = pd.DataFrame(data)
#     df.set_index(pd.Index(["002_A1_2_SOO", "003_A1_3_SGO", "004_A1_4_SOL", "005_A1_5_SGL",
#                            "006_A1_6_SGL", "007_A1_7_SGO", "008_A1_8_SOL", "009_A1_9_SGL",
#                            "010_A1_10_SOO", "011_A1_11_SGO"]), inplace=True)

#     # Run clustering analysis
#     analyze_clustering(df)
