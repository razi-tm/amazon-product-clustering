import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Example:
# Assume `embeddings` is a NumPy array of shape (n_samples, n_features)
# and `product_ids` is a list of corresponding product identifiers.


def hierarchical_clustering(
    embeddings: np.ndarray,
    product_ids: list,
    distance_thresholds: list = [0.5, 1.0, 1.5],
    linkage_method: str = 'ward',
    plot_dendrogram: bool = True
):
    """
    Perform hierarchical/agglomerative clustering on product embeddings.

    Args:
        embeddings (np.ndarray): Embedding matrix.
        product_ids (list): List of product IDs of length n_samples.
        distance_thresholds (list): Distances at which to cut dendrogram.
        linkage_method (str): Linkage method ('ward', 'average', 'complete').
        plot_dendrogram (bool): If True, plots the dendrogram.

    Returns:
        dict: Mapping from threshold to clusters (dict mapping cluster label to product IDs).
    """
    # Compute linkage matrix
    Z = linkage(embeddings, method=linkage_method)

    if plot_dendrogram:
        plt.figure(figsize=(10, 6))
        dendrogram(Z, no_labels=True, color_threshold=0)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.savefig("hierarchical_clustering_dendrogram.png", dpi=300)
        print("Saved hierarchical_clustering_dendrogram.png")

    clusters_by_threshold = {}
    for thresh in distance_thresholds:
        # fcluster assigns cluster labels given a distance cutoff
        labels = fcluster(Z, t=thresh, criterion='distance')
        clusters = {}
        for pid, lab in zip(product_ids, labels):
            clusters.setdefault(lab, []).append(pid)
        clusters_by_threshold[thresh] = clusters

    return clusters_by_threshold


# Usage example:
if __name__ == '__main__':
    # Load or compute embeddings and product IDs
    text_embeddings = np.load('text_embeddings.npy')
    image_embeddings = np.load('image_embeddings.npy')
    combined_embeddings = np.hstack([text_embeddings, image_embeddings])

    product_ids = pd.read_csv('cellphones_subset.csv')
    
    # thresholds: adjust for "identical", "similar", "related"
    thresholds = [0.3, 1.0, 2.0]
    clusters_dict = hierarchical_clustering(
        combined_embeddings,
        product_ids,
        distance_thresholds=thresholds,
        linkage_method='ward',
        plot_dendrogram=True
    )

    # Inspect clusters at each level
    for thresh, clusters in clusters_dict.items():
        print(f"\nClusters at distance ≤ {thresh}: \nTotal clusters: {len(clusters)}")
        # Print a sample
        for lab, items in list(clusters.items())[:5]:
            print(f"Cluster {lab} ({len(items)} items): {items[:5]}...")
