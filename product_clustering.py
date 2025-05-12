import json
import pandas as pd
import os
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from PIL import Image
import clip

def preproccess_csv():
    # Configure output
    pd.set_option('display.max_colwidth', 100)

    # Create output folder
    os.makedirs("images", exist_ok=True)

    # Load and Parse Amazon Metadata (first 10,000 items)
    MAX_PRODUCTS = 10000
    metadata_path = "meta_Cell_Phones_and_Accessories.json"

    products = []

    with open(metadata_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= MAX_PRODUCTS:
                break
            try:
                entry = json.loads(line)
                
                # Safe category parsing
                cat = entry.get("category", [])
                last_level = cat[-1] if cat and isinstance(cat[-1], list) else []
                category_str = " > ".join(last_level) if last_level else None

                # Safe image URL parsing
                image_list = entry.get("imageURLHighRes")
                image_url = image_list[0] if image_list and len(image_list) > 0 else None

                products.append({
                    "asin": entry.get("asin"),
                    "title": entry.get("title", "").strip(),
                    "description": " ".join(entry.get("description", [])),
                    "brand": entry.get("brand"),
                    "category": category_str,
                    "image_url": image_url,
                    "also_buy": entry.get("also_buy", []),
                    "also_view": entry.get("also_view", [])
                })
            except Exception as e:
                print(f"Error parsing line {i}: {e}")
                print("Entry:", entry)

    df = pd.DataFrame(products)

    # Clean and Filter
    df = df[df["title"].notnull()]
    df = df[df["image_url"].notnull()]
    df.reset_index(drop=True, inplace=True)

    print("Sample Data:")
    print(df.head(3).to_string())

    # Save Cleaned CSV
    df.to_csv("cellphones_subset.csv", index=False)
    print(f"Saved cleaned subset: {len(df)} products")

# Download Images
def download_images(csv_path):
    df = pd.read_csv(csv_path)
    print("Downloading images...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        asin = row['asin']
        url = row['image_url']
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                with open(f"images/{asin}.jpg", "wb") as f:
                    f.write(response.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

def text_embedding(csv_path):
    df = pd.read_csv(csv_path)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    def combine_text(row):
        return f"{row['title']} {row['description']}"

    df["text"] = df.apply(combine_text, axis=1)
    text_embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
    np.save("text_embeddings.npy", text_embeddings)

def image_embedding(csv_path):
    df = pd.read_csv(csv_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    image_embeddings = []

    for asin in tqdm(df["asin"]):
        try:
            img = Image.open(f"images/{asin}.jpg").convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = clip_model.encode_image(img_tensor)
            image_embeddings.append(embedding.cpu().numpy())
        except Exception as e:
            image_embeddings.append(np.zeros((1, 512)))  # placeholder
            print(f"Image error for {asin}: {e}")

    image_embeddings = np.vstack(image_embeddings)
    np.save("image_embeddings.npy", image_embeddings)


def main(csv_path="cellphones_subset.csv"):
    preproccess_csv()
    download_images(csv_path)
    text_embedding(csv_path)
    image_embedding(csv_path)

if __name__ == "__main__":
    main()
