import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import umap.umap_ as umap


# Load clustered product metadata
df = pd.read_csv("clustered_products.csv")

# Load embeddings
text_embeddings = np.load("text_embeddings.npy")
image_embeddings = np.load("image_embeddings.npy")

# Combine for similarity clustering
from sklearn.preprocessing import normalize
combined_embeddings = np.hstack([normalize(text_embeddings), normalize(image_embeddings)])


def plot_similarity_clusters(df, combined_embeddings):
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_umap = reducer.fit_transform(combined_embeddings)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=embedding_umap[:, 0], y=embedding_umap[:, 1],
        hue=df["similarity_cluster"],
        palette="tab20",
        s=10, linewidth=0
    )
    plt.title("Similarity Clustering (UMAP + HDBSCAN)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster")
    plt.tight_layout()
    plt.savefig("similarity_clusters.png", dpi=300)
    print("Saved similarity_clusters.png")

plot_similarity_clusters(df, combined_embeddings)


def plot_graph_clusters(df, sample_size=200):
    G = nx.Graph()
    asin_to_cluster = dict(zip(df["asin"], df["complementary_cluster"]))

    for _, row in df.iterrows():
        source = row["asin"]
        if not isinstance(source, str):
            continue
        for field in ["also_buy", "also_view"]:
            try:
                neighbors = eval(row[field]) if isinstance(row[field], str) else row[field]
                for target in neighbors:
                    if target in asin_to_cluster:
                        G.add_edge(source, target)
            except:
                continue

    sampled_nodes = list(asin_to_cluster.keys())[:sample_size]
    sampled_subgraph = G.subgraph(sampled_nodes)

    pos = nx.spring_layout(sampled_subgraph, seed=42)
    communities = [asin_to_cluster[n] for n in sampled_subgraph.nodes()]
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(sampled_subgraph, pos, node_size=40, node_color=communities, cmap=plt.cm.Set3)
    nx.draw_networkx_edges(sampled_subgraph, pos, alpha=0.3)
    plt.title("Complementary Product Graph Clustering")
    plt.axis('off')
    plt.savefig("complementary_clusters.png", dpi=300)
    print("Saved complementary_clusters.png")

plot_graph_clusters(df)
