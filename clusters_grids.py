import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# CONFIGURATION
CSV_PATH        = "clustered_products.csv"   # your clustered CSV
IMAGES_DIR      = "images"                   # folder with <ASIN>.jpg files
OUTPUT_DIR      = "cluster_grids"            # where to save grid images
SAMPLES_PER_CLUSTER = 5                      # max images per cluster row

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # construct full image paths and filter only existing ones
    df["image_path"] = df["asin"].apply(lambda a: os.path.join(IMAGES_DIR, f"{a}.jpg"))
    df = df[df["image_path"].apply(os.path.exists)]
    return df

def plot_and_save_grid(df, cluster_col, cluster_id, samples_per_cluster=SAMPLES_PER_CLUSTER):
    sub = df[df[cluster_col] == cluster_id].head(samples_per_cluster)
    if sub.empty:
        return  # nothing to plot

    # create figure with one column per sample
    n = len(sub)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 4), squeeze=False)
    fig.suptitle(f"{cluster_col} = {cluster_id}", fontsize=14)

    for ax, (_, row) in zip(axes[0], sub.iterrows()):
        img = Image.open(row["image_path"])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(row["asin"], fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    out_fname = f"{cluster_col}_cluster_{cluster_id}.png"
    fig.savefig(os.path.join(f"{OUTPUT_DIR}/{cluster_col}", out_fname), dpi=150)
    plt.close(fig)

def main():
    cluster_columns = ["identity_cluster", "similarity_cluster", "complementary_cluster"]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for folder in cluster_columns:
        os.makedirs(f"{OUTPUT_DIR}/{folder}", exist_ok=True)

    df = load_data(CSV_PATH)

    for cluster_col in cluster_columns:
        # get unique IDs, skip NaNs
        ids = sorted(df[cluster_col].dropna().unique(), key=lambda x: int(x))
        print(f"Rendering {len(ids)} grids for '{cluster_col}'...")
        for cid in ids:
            plot_and_save_grid(df, cluster_col, cid)

    print(f"All grids saved under: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
