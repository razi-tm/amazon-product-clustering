
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import umap.umap_ as umap
import hdbscan
import networkx as nx


# ========== Step 1: Load Data and Embeddings ==========

df = pd.read_csv("cellphones_subset.csv")
text_embeddings = np.load("text_embeddings.npy")
image_embeddings = np.load("image_embeddings.npy")

# Normalize embeddings
text_embeddings = normalize(text_embeddings)
image_embeddings = normalize(image_embeddings)

# ========== Step 2: Identity Clustering (Text Only) ==========

print("Running Identity Clustering (Text)...")
kmeans = KMeans(n_clusters=20, random_state=42)
df["identity_cluster"] = kmeans.fit_predict(text_embeddings)

# ========== Step 3: Similarity Clustering (Text + Image) ==========

print("Running Similarity Clustering (Multimodal)...")
combined_embeddings = np.hstack([text_embeddings, image_embeddings])

# Reduce dimensions with UMAP
reducer = umap.UMAP(n_components=50, random_state=42)
embedding_umap = reducer.fit_transform(combined_embeddings)

# Cluster with HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
df["similarity_cluster"] = clusterer.fit_predict(embedding_umap)

# ========== Step 4: Complementary Product Clustering (Graph) ==========

print("Building Graph from 'also_buy' and 'also_view'...")
G = nx.Graph()

asin_to_idx = {asin: idx for idx, asin in enumerate(df["asin"])}

for idx, row in df.iterrows():
    source = row["asin"]
    if not isinstance(source, str):
        continue
    for target_list in ["also_buy", "also_view"]:
        try:
            neighbors = eval(row[target_list]) if isinstance(row[target_list], str) else row[target_list]
            for target in neighbors:
                if target in asin_to_idx:
                    G.add_edge(source, target)
        except:
            continue

# Community detection
print("Detecting communities...")
from networkx.algorithms.community import label_propagation_communities

communities = label_propagation_communities(G)
asin_to_community = {}
for i, com in enumerate(communities):
    for asin in com:
        asin_to_community[asin] = i

df["complementary_cluster"] = df["asin"].map(asin_to_community)

# ========== Step 5: Summary ==========

print("\nCluster counts:")
print("Identity Clusters:", df["identity_cluster"].nunique())
print("Similarity Clusters:", df["similarity_cluster"].nunique())
print("Complementary Clusters:", df["complementary_cluster"].nunique())

df.to_csv("clustered_products.csv", index=False)
print("\nSaved clustered dataset to 'clustered_products.csv'.")
