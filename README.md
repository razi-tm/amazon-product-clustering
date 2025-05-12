# 📦 Amazon Product Clustering Pipeline

This project performs **unsupervised clustering of Amazon products** using textual metadata, image features, and co-purchase/view graphs. It is tailored for the **"Cell Phones and Accessories"** category and supports three major clustering strategies:

* **Identity Clustering (Text-Based)** — groups based on product descriptions/titles.
* **Similarity Clustering (Multimodal)** — combines both text and image features.
* **Complementary Product Clustering** — detects related products based on graph links from `also_buy` and `also_view`.

Visualizations of clustered products (image grids) are also generated for fast inspection.

---

## Data
You can download Amazon Data (2018) for cell phones and accessories from: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Cell_Phones_and_Accessories.json.gz

---

## 📂 Project Structure

```
🔹 clustered_products.csv                     # Final dataset with cluster labels
🔹 product_clustering.py                      # Preprocessing + Embedding Generator
🔹 amazon_clustering_pipeline.py              # Clustering logic (Text, Image, Graph)
🔹 amazon_clustering_pipeline.py              # Clustering logic (Text, Image, Graph)
🔹 validate_clusters.py                       # validating results
🔹 plot_clusters.py                           # plot generator for each cluster
🔹 meta_Cell_Phones_and_Accessories.json      # Raw metadata (from Amazon)
🔹 cellphones_subset.csv                      # Cleaned dataset with key fields
🔹 images/                                    # Downloaded product images
🔹 cluster_grids/                             # Image grids for cluster preview
🔹 text_embeddings.npy                        # SentenceTransformer embeddings
🔹 image_embeddings.npy                       # CLIP image embeddings
🔹 README.md                                  # Project overview and usage
```

---

## 📦 Setup & Requirements

Make sure you have:

* Python 3.8+
* A GPU (optional but recommended for CLIP embeddings)

Install dependencies:

```bash
pip install -r requirements.txt
```

<details>
<summary>📄 <code>requirements.txt</code> (example)</summary>

```text
pandas
numpy
scikit-learn
umap-learn
hdbscan
networkx
matplotlib
sentence-transformers
torch
tqdm
Pillow
clip @ git+https://github.com/openai/CLIP.git
```

</details>

---

## 🔧 Step-by-Step Pipeline

### 1. 📄 Preprocess Metadata + Download Images + Generate Embeddings

```bash
python product_clustering.py
```

This will:

* Parse and clean product metadata (`meta_*.json` → `cellphones_subset.csv`)
* Download images to `images/`
* Generate and save:

  * `text_embeddings.npy` using Sentence-BERT (`all-MiniLM-L6-v2`)
  * `image_embeddings.npy` using CLIP (`ViT-B/32`)

---

### 2. 🧐 Perform Clustering

```bash
python amazon_clustering_pipeline.py
```

This script:

* Loads and normalizes embeddings
* Applies:

  * **KMeans** (Text Only — Identity Clustering)
  * **UMAP + HDBSCAN** (Text + Image — Similarity Clustering)
  * **Graph Clustering** via NetworkX (Complementary Product Clustering)
* Outputs `clustered_products.csv` with:

  * `identity_cluster`
  * `similarity_cluster`
  * `complementary_cluster`

---

### 3. 🧐 Validate Results

```bash
python validate_clusters.py
```

---

### 4. 🖼 Visualize Cluster Image Grids

```bash
python visualize_clusters_grids.py
```

This generates image grids for each cluster into `cluster_grids/`:

```
cluster_grids/
🔹 identity_cluster_0.png
🔹 similarity_cluster_2.png
🔹 complementary_cluster_5.png
└── ...
```

Each image contains up to 5 representative products in that cluster:

![Example Identity Cluster](cluster_grids/identity_cluster_0.png)

---

## 🔢 Example: CSV Output

| asin      | title                       | image\_url | identity\_cluster | similarity\_cluster | complementary\_cluster |
| --------- | --------------------------- | ---------- | ----------------- | ------------------- | ---------------------- |
| B00012345 | Apple iPhone XR - 64GB      | ...        | 0                 | 2                   | 5                      |
| B00067890 | OtterBox Case for iPhone XR | ...        | 3                 | 2                   | 5                      |

---

## 📌 Notes

* You can tune `n_clusters` (KMeans) or `min_cluster_size` (HDBSCAN) for different clustering granularity.
* Image quality and missing downloads may impact visual clustering—these are skipped if unavailable.
* You may also explore **embedding visualization** with t-SNE or UMAP in future work.

---

## 🧠 Future Work

* Fine-tune models on e-commerce domain-specific data.
* Add interactive HTML visualizations for clusters.
* Implement multilingual support for product text.
* Use category hierarchy more deeply in clustering.

---

## 🛠 Author & Credits

Developed by \[Your Name or Team]
Inspired by OpenAI CLIP, Amazon Metadata, and SBERT

---

## 📃 License

MIT License. Use freely with attribution.
