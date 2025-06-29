from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss

# Loading the dataset
df = pd.read_csv(r'data\pinterest_finalised.csv')

# Cleaning the dataset
# - Remove rows with missing values in 'title', 'description', or 'repin_count'
df.dropna(subset=['title', 'description', 'repin_count'], inplace=True)

# - Convert 'repin_count' to numeric, forcing errors to NaN, then drop those rows
df['full_text'] = df['title'] + ' ' + df['description']

# - Convert 'Repin Count' to numeric, forcing errors to NaN
df["repin_count"] = pd.to_numeric(df["repin_count"], errors="coerce")

# - Save the cleaned DataFrame to a new CSV file
df.to_csv('data/pinterest_cleaned.csv', index=False)

# Create Model and Generate Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['full_text'].tolist(), show_progress_bar=True, normalize_embeddings=True)

embeddings = np.array(embeddings, dtype=np.float32)

# Create FAISS index
dimennsion = embeddings.shape[1]
index = faiss.IndexFlatL2(dimennsion)
index.add(embeddings)

#Save the FAISS index to a file
faiss.write_index(index, 'data/faiss_index.index')
