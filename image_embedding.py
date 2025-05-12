import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import clip

df = pd.read_csv("cellphones_subset.csv")

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
