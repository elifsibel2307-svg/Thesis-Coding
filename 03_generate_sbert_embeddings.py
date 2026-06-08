import os
import pandas as pd
from sentence_transformers import SentenceTransformer

input_file = "data/processed/descriptions_for_embeddings.csv"
output_file = "data/processed/sbert_embeddings.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError("Run Part 1 first: descriptions_for_embeddings.csv is missing.")

df = pd.read_csv(input_file)

print("Loaded descriptions:", df.shape)

model = SentenceTransformer("all-mpnet-base-v2")

embeddings = model.encode(
    df["description"].fillna("").tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

embedding_columns = [f"emb_{i}" for i in range(embeddings.shape[1])]
embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns)

embeddings_df.insert(0, "company_id", df["company_id"])
embeddings_df.insert(1, "company_name", df["company_name"])

embeddings_df.to_csv(output_file, index=False)

print("Saved embeddings:", embeddings_df.shape)
print("Output file:", output_file)
