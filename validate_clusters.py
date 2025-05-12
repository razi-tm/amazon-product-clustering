import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter

def load_data(csv_path):
    return pd.read_csv(csv_path)

def check_image_existence(df, image_dir):
    df["image_exists"] = df["asin"].apply(lambda x: os.path.isfile(os.path.join(image_dir, f"{x}.jpg")))
    print("\nImage File Availability:")
    print(df["image_exists"].value_counts())
    return df

def print_cluster_distribution(df):
    for col in ["identity_cluster", "similarity_cluster", "complementary_cluster"]:
        print(f"\nCluster distribution for: {col}")
        print(df[col].value_counts(dropna=False).sort_index())

def print_cluster_examples(df, cluster_col, cluster_id, n=5, image_dir="images"):
    cluster_df = df[df[cluster_col] == cluster_id]
    if cluster_df.empty:
        print(f"No examples found for cluster {cluster_id} in {cluster_col}.")
        return

    sample = cluster_df.sample(n=min(n, len(cluster_df)))
    print(f"\nExamples from {cluster_col} = {cluster_id}:")
    for _, row in sample.iterrows():
        print(f"[{row['asin']}] {row.get('title', 'No Title')}")
        print(f"Image: {os.path.join(image_dir, f'{row['asin']}.jpg')} (exists: {os.path.isfile(os.path.join(image_dir, f'{row['asin']}.jpg'))})\n")

def plot_missing_assignments(df):
    cols = ["identity_cluster", "similarity_cluster", "complementary_cluster"]
    missing_counts = {col: df[col].isna().sum() for col in cols}
    plt.bar(missing_counts.keys(), missing_counts.values())
    plt.title("Missing Cluster Assignments")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("missing_assignments.png")
    print("Saved missing cluster assignments plot to 'missing_assignments.png'")

def run_all_validation(csv_path, image_dir="images"):
    df = load_data(csv_path)
    df = check_image_existence(df, image_dir)
    print_cluster_distribution(df)
    plot_missing_assignments(df)
    # Print examples from cluster 0 of each type (can be changed)
    print_cluster_examples(df, "identity_cluster", 0)
    print_cluster_examples(df, "similarity_cluster", 0)
    print_cluster_examples(df, "complementary_cluster", 0)

if __name__ == "__main__":
    run_all_validation("clustered_products.csv")