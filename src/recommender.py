import numpy as np
from utils import get_model, get_faiss_index, get_dataframe

def get_user_recommendations(user_liked_texts, model, index, df, top_k=5):

    # Generate embeddings for the user's liked texts
    user_embeddings = model.encode(user_liked_texts, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
    
    user_profile = np.mean(user_embeddings, axis=0).reshape(1, -1)

    # Search the FAISS index for the most similar items
    distances, indices = index.search(user_profile, top_k)
    results = df.iloc[indices[0]].copy()
    results['similarity'] = 1 - distances[0]  # Convert distance to similarity
    return results


model = get_model()
index = get_faiss_index()
df = get_dataframe()
liked_pins = [
    "A beautiful sunset over the mountains",
    "A delicious recipe for chocolate cake",
    "A stunning landscape photograph of a forest"
]

user_recs = get_user_recommendations(liked_pins, model, index, df)
print(user_recs[['title', 'description', 'similarity']])


