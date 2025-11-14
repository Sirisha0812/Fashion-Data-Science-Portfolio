"""
Build FAISS index for fast similarity search.
Run this after downloading embeddings from Kaggle.
"""

import numpy as np
import faiss
import pandas as pd
from pathlib import Path
import pickle

from config import *

def build_faiss_index(embeddings, index_type='IndexFlatL2'):
    """
    Build FAISS index from embeddings.
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        index_type: Type of FAISS index ('IndexFlatL2' or 'IndexIVFFlat')
    
    Returns:
        FAISS index
    """
    dimension = embeddings.shape[1]
    embeddings = embeddings.astype('float32')
    
    print(f"Building FAISS index...")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Dimension: {dimension}")
    
    if index_type == 'IndexFlatL2':
        # Exact search (slower but accurate)
        index = faiss.IndexFlatL2(dimension)
    elif index_type == 'IndexIVFFlat':
        # Approximate search (faster for large datasets)
        nlist = 100  # Number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        # Train the index
        print("   Training index...")
        index.train(embeddings)
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Add embeddings to index
    print("   Adding embeddings to index...")
    index.add(embeddings)
    
    print(f"Index built successfully!")
    print(f"   Total vectors: {index.ntotal}")
    
    return index

def save_index(index, filepath):
    """Save FAISS index to disk."""
    # Convert Path to string if needed
    filepath_str = str(filepath)
    faiss.write_index(index, filepath_str)
    print(f"Index saved to: {filepath_str}")

def load_index(filepath):
    """Load FAISS index from disk."""
    # Convert Path to string if needed
    filepath_str = str(filepath)
    index = faiss.read_index(filepath_str)
    print(f"Index loaded from: {filepath_str}")
    print(f"   Total vectors: {index.ntotal}")
    return index

def search_similar(index, query_embedding, k=10):
    """
    Search for similar items.
    
    Args:
        index: FAISS index
        query_embedding: numpy array of shape (1, dimension)
        k: Number of results to return
    
    Returns:
        distances: Array of distances
        indices: Array of indices
    """
    query_embedding = query_embedding.astype('float32')
    
    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Search
    distances, indices = index.search(query_embedding, k)
    
    return distances[0], indices[0]

if __name__ == "__main__":
    # Load embeddings
    embeddings_path = EMBEDDINGS_DIR / "text_embeddings.npy"
    
    if not embeddings_path.exists():
        print(f"ERROR: Embeddings not found at: {embeddings_path}")
        print(f"\nPlease download embeddings from Kaggle first:")
        print(f"   1. Run TF-IDF extraction in Kaggle")
        print(f"   2. Download text_embeddings.npy")
        print(f"   3. Save to: {embeddings_path}")
    else:
        print("Loading embeddings...")
        embeddings = np.load(embeddings_path)
        print(f"Loaded embeddings: {embeddings.shape}")
        
        # Build index
        index = build_faiss_index(embeddings)
        
        # Save index
        index_path = EMBEDDINGS_DIR / "faiss_index.bin"
        save_index(index, index_path)
        
        # Test search
        print("\nTesting search...")
        test_query = embeddings[0:1]  # Use first embedding as test
        distances, indices = search_similar(index, test_query, k=5)
        print(f"   Found {len(indices)} similar items")
        print(f"   Distances: {distances[:3]}")
        print(f"   Indices: {indices[:3]}")
        
        print("\nFAISS index ready for recommendations!")

