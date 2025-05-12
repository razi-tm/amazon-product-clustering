from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

df = pd.read_csv("cellphones_subset.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

def combine_text(row):
    return f"{row['title']} {row['description']}"

df["text"] = df.apply(combine_text, axis=1)
text_embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
np.save("text_embeddings.npy", text_embeddings)
