"""
Complete script to build the recommendation system.
Run this after downloading embeddings from Kaggle.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from similarity_search import build_faiss_index, save_index
from recommendation_engine import FashionRecommendationEngine
import numpy as np
import pandas as pd

def main():
    """Build the complete recommendation system."""
    print("="*70)
    print("BUILDING FASHION RECOMMENDATION SYSTEM")
    print("="*70)
    
    # Step 1: Check if embeddings exist
    from config import EMBEDDINGS_DIR
    embeddings_path = EMBEDDINGS_DIR / "text_embeddings.npy"
    
    if not embeddings_path.exists():
        print(f"\nERROR: Embeddings not found at: {embeddings_path}")
        print("\nPlease download embeddings from Kaggle first:")
        print("   1. Run kaggle_notebook_TFIDF_ONLY.py in Kaggle")
        print("   2. Download text_embeddings.npy")
        print("   3. Save to: data/embeddings/text_embeddings.npy")
        return
    
    # Step 2: Load embeddings
    print("\n" + "="*70)
    print("STEP 1: Loading Embeddings")
    print("="*70)
    embeddings = np.load(embeddings_path)
    print(f"Loaded embeddings: {embeddings.shape}")
    
    # Step 3: Build FAISS index
    print("\n" + "="*70)
    print("STEP 2: Building FAISS Index")
    print("="*70)
    index = build_faiss_index(embeddings)
    
    # Save index
    index_path = EMBEDDINGS_DIR / "faiss_index.bin"
    save_index(index, index_path)
    
    # Step 4: Test recommendation engine
    print("\n" + "="*70)
    print("STEP 3: Testing Recommendation Engine")
    print("="*70)
    
    try:
        engine = FashionRecommendationEngine()
        
        # Test with product ID
        if len(engine.df) > 0:
            test_id = engine.df.iloc[0]['id']
            print(f"\nTesting with Product ID: {test_id}")
            results = engine.recommend_by_product_id(test_id, k=5)
            print(f"Found {len(results)} recommendations")
            print("\nTop recommendations:")
            for idx, row in results.head(3).iterrows():
                print(f"   - {row['productDisplayName']}")
                print(f"     {row['articleType']} - {row['baseColour']}")
                print(f"     Similarity: {row['similarity_score']:.3f}\n")
        
    except Exception as e:
        print(f"WARNING: Could not test engine: {e}")
        print("   (This is OK if vectorizer not loaded)")
    
    # Step 5: Summary
    print("\n" + "="*70)
    print("RECOMMENDATION SYSTEM READY!")
    print("="*70)
    print("\nFiles created:")
    print(f"   {index_path}")
    print("\nNext steps:")
    print("   1. Test recommendations: python src/recommendation_engine.py")
    print("   2. Build Streamlit app: streamlit run app/streamlit_app_simple.py")
    print("   3. Start using the recommendation system!")
    print("="*70)

if __name__ == "__main__":
    main()

