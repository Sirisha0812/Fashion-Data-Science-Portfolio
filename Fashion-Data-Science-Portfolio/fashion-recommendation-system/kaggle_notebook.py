"""
Kaggle Notebook - OPTIMIZED VERSION
Faster text extraction with optimizations!
"""

# ============================================================================
# SETUP: Import packages
# ============================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING:  WARNING: No GPU detected! Using CPU (will be slow)")

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
# TEXT FEATURE EXTRACTION - OPTIMIZED VERSION
# ============================================================================
print("\n" + "="*60)
print("TEXT FEATURE EXTRACTION (OPTIMIZED)")
print("="*60)

class OptimizedTextFeatureExtractor:
    def __init__(self, device='cuda', batch_size=128):
        self.device = device
        self.batch_size = batch_size
        self.model_name = 'distilbert-base-uncased'
        
        print(f"Loading {self.model_name}...")
        print(f"Batch size: {batch_size}")
        
        # Load tokenizer and model with timeout and error handling
        try:
            # Set timeout to fail faster
            import os
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '10'
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=False,
                timeout=10
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=False,
                timeout=10
            )
            self.model.eval()
            self.model.to(device)
        except Exception as e:
            # Re-raise to be caught by outer try-except
            raise Exception(f"Failed to load model: {e}")
        
        # Optimize model for inference
        if device.type == 'cuda':
            # Use torch.compile for faster inference (PyTorch 2.0+)
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print(" Model optimized with torch.compile")
            except:
                print("WARNING:  torch.compile not available, using standard model")
        
        print(" Model loaded and optimized!")
    
    def extract_batch(self, texts):
        """Extract features with optimized batching."""
        all_features = []
        
        # Pre-tokenize all texts for better batching
        print("Pre-processing texts...")
        processed_texts = [str(t) for t in texts]
        
        # Process in larger batches
        num_batches = (len(processed_texts) + self.batch_size - 1) // self.batch_size
        print(f"Processing {len(processed_texts)} texts in {num_batches} batches...")
        
        for i in tqdm(range(0, len(processed_texts), self.batch_size), desc="Extracting features"):
            batch_texts = processed_texts[i:i+self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features (no gradients)
            with torch.no_grad():
                # Use inference mode for faster processing
                with torch.inference_mode():
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Normalize
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1
            features = features / norms
            all_features.extend(features)
        
        return np.array(all_features)

# Determine optimal batch size based on GPU memory
if torch.cuda.is_available():
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory_gb >= 16:
        batch_size = 128  # Large GPU
    elif gpu_memory_gb >= 8:
        batch_size = 64   # Medium GPU
    else:
        batch_size = 32   # Small GPU
    print(f"\nDetected GPU memory: {gpu_memory_gb:.1f} GB")
    print(f"Using batch size: {batch_size}")
else:
    batch_size = 16  # CPU
    print(f"\nWARNING:  Using CPU - batch size: {batch_size}")

# Try to initialize extractor (with fallback to TF-IDF)
print("\nInitializing text feature extractor...")
text_extractor = None
use_tfidf = False

try:
    text_extractor = OptimizedTextFeatureExtractor(device=device, batch_size=batch_size)
    print(" Using DistilBERT (transformer model)")
except Exception as e:
    print(f"\nWARNING:  Could not load DistilBERT: {e}")
    print("   This is likely a network issue (can't download from Hugging Face)")
    print("\n🔄 Falling back to TF-IDF (no download needed!)")
    use_tfidf = True

# Get product descriptions
texts = df['productDisplayName'].fillna('').astype(str).tolist()
print(f"\nExtracting features from {len(texts)} product descriptions...")

# Extract features
import time
start_time = time.time()

if use_tfidf:
    # Use TF-IDF fallback
    print("Using TF-IDF feature extraction (fast, no download needed)...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    
    vectorizer = TfidfVectorizer(
        max_features=512,  # Match embedding dimension
        stop_words='english',
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,
        max_df=0.95
    )
    
    print("Fitting TF-IDF vectorizer...")
    text_embeddings = vectorizer.fit_transform(texts).toarray()
    
    # Normalize to unit length (like transformer embeddings)
    text_embeddings = normalize(text_embeddings, norm='l2')
    
    print(" TF-IDF extraction complete!")
else:
    # Use transformer model
    print("⏳ Estimated time: 5-10 minutes on GPU, 15-30 minutes on CPU")
    text_embeddings = text_extractor.extract_batch(texts)

elapsed_time = time.time() - start_time
print(f"\n Extraction complete!")
print(f"   Method: {'TF-IDF' if use_tfidf else 'DistilBERT'}")
print(f"   Time taken: {elapsed_time/60:.1f} minutes")
print(f"   Speed: {len(texts)/elapsed_time:.1f} texts/second")

# Save embeddings
output_path = '/kaggle/working/text_embeddings.npy'
np.save(output_path, text_embeddings)
print(f"\n Saved text embeddings: {text_embeddings.shape}")
print(f"   File: {output_path}")
print(f"   Size: {os.path.getsize(output_path) / 1e6:.1f} MB")

# Save product IDs
df[['id']].to_csv('/kaggle/working/product_ids.csv', index=False)
print(f" Saved product IDs")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print(" TEXT FEATURE EXTRACTION COMPLETE!")
print("="*60)
print(f"\nPerformance:")
print(f"   Total time: {elapsed_time/60:.1f} minutes")
print(f"   Processing speed: {len(texts)/elapsed_time:.1f} texts/second")
print(f"   Device used: {device}")
print(f"\nFiles saved:")
print(f"    text_embeddings.npy ({os.path.getsize(output_path) / 1e6:.1f} MB)")
print(f"    product_ids.csv")
print("\n Download these files and use them locally!")
print("="*60)

