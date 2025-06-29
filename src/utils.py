from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss

def get_model():
    """
    Load and return the SentenceTransformer model.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_faiss_index():
    """
    Load and return the FAISS index.
    """
    return faiss.read_index('data/faiss_index.index')

def get_dataframe():
    """
    Load and return the cleaned DataFrame.
    """
    return pd.read_csv('data/pinterest_cleaned.csv')