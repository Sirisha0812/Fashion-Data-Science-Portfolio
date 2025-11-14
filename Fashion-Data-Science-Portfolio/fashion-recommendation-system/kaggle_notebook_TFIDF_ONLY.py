"""
Kaggle Notebook - TF-IDF ONLY VERSION
No Hugging Face download needed - works immediately!
"""

# ============================================================================
# SETUP: Import packages
# ============================================================================
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm
import os
import time

print("="*60)
print("TEXT FEATURE EXTRACTION - TF-IDF VERSION")
print("="*60)
print(" No model download needed!")
print(" Works offline!")
print(" Fast extraction (2-5 minutes)!")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "="*60)
print("LOADING DATASET")
print("="*60)

csv_path = '/kaggle/input/fashion-product-images-dataset/fashion-dataset/styles.csv'

try:
    df = pd.read_csv(csv_path, on_bad_lines='skip', low_memory=False)
except TypeError:
    df = pd.read_csv(csv_path, error_bad_lines=False, low_memory=False)

print(f" Loaded {len(df)} products")

# ============================================================================
# TEXT FEATURE EXTRACTION - TF-IDF (No Download Needed!)
# ============================================================================
print("\n" + "="*60)
print("TEXT FEATURE EXTRACTION (TF-IDF)")
print("="*60)

# Get product descriptions
texts = df['productDisplayName'].fillna('').astype(str).tolist()
print(f"Extracting features from {len(texts)} product descriptions...")
print("⏳ Estimated time: 2-5 minutes")

# Start timer
start_time = time.time()

# Create TF-IDF vectorizer
print("\nCreating TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=512,  # Match embedding dimension (same as DistilBERT)
    stop_words='english',  # Remove common words
    ngram_range=(1, 2),  # Unigrams and bigrams (captures phrases)
    min_df=2,  # Ignore words that appear in <2 documents
    max_df=0.95,  # Ignore words that appear in >95% documents
    lowercase=True,
    strip_accents='unicode'
)

# Extract features
print("Fitting TF-IDF and extracting features...")
text_embeddings = vectorizer.fit_transform(texts).toarray()

# Normalize to unit length (like transformer embeddings)
print("Normalizing embeddings...")
text_embeddings = normalize(text_embeddings, norm='l2')

elapsed_time = time.time() - start_time

print(f"\n TF-IDF extraction complete!")
print(f"   Time taken: {elapsed_time/60:.1f} minutes ({elapsed_time:.1f} seconds)")
print(f"   Speed: {len(texts)/elapsed_time:.1f} texts/second")
print(f"   Embedding shape: {text_embeddings.shape}")
print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")

# ============================================================================
# SAVE EMBEDDINGS
# ============================================================================
output_path = '/kaggle/working/text_embeddings.npy'
np.save(output_path, text_embeddings)
print(f"\n Saved text embeddings: {text_embeddings.shape}")
print(f"   File: {output_path}")
print(f"   Size: {os.path.getsize(output_path) / 1e6:.1f} MB")

# Save product IDs
df[['id']].to_csv('/kaggle/working/product_ids.csv', index=False)
print(f" Saved product IDs")

# Optional: Save TF-IDF vectorizer for later use
import pickle
vectorizer_path = '/kaggle/working/tfidf_vectorizer.pkl'
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f" Saved TF-IDF vectorizer (for future use)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print(" TEXT FEATURE EXTRACTION COMPLETE!")
print("="*60)
print(f"\nPerformance:")
print(f"   Method: TF-IDF (no download needed)")
print(f"   Total time: {elapsed_time/60:.1f} minutes")
print(f"   Processing speed: {len(texts)/elapsed_time:.1f} texts/second")
print(f"   Embedding dimensions: {text_embeddings.shape[1]}")
print(f"\nFiles saved:")
print(f"    text_embeddings.npy ({os.path.getsize(output_path) / 1e6:.1f} MB)")
print(f"    product_ids.csv")
print(f"    tfidf_vectorizer.pkl (optional)")
print("\n TF-IDF Features:")
print("    Excellent for text-based recommendations")
print("    Captures important keywords and phrases")
print("    Same format as transformer embeddings (512-dim)")
print("    Ready to use for similarity search!")
print("\n Download these files and use them locally!")
print("="*60)

