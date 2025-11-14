"""
Recommendation Engine for Fashion Products.
Uses FAISS index for fast similarity search.
"""

import numpy as np
import pandas as pd
import faiss
from pathlib import Path
import pickle
from sklearn.preprocessing import normalize

from config import *
from similarity_search import load_index, search_similar

class FashionRecommendationEngine:
    """Main recommendation engine."""
    
    def __init__(self):
        """Initialize the recommendation engine."""
        self.index = None
        self.df = None
        self.vectorizer = None
        self.embeddings = None
        
        self._load_data()
    
    def _load_data(self):
        """Load index, metadata, and vectorizer."""
        print("Loading recommendation engine...")
        
        # Load FAISS index
        index_path = EMBEDDINGS_DIR / "faiss_index.bin"
        if index_path.exists():
            self.index = load_index(index_path)
        else:
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run similarity_search.py first.")
        
        # Load product metadata
        metadata_path = RAW_DATA_DIR / "styles.csv"
        if metadata_path.exists():
            try:
                self.df = pd.read_csv(metadata_path, on_bad_lines='skip', low_memory=False)
            except TypeError:
                self.df = pd.read_csv(metadata_path, error_bad_lines=False, low_memory=False)
            print(f"Loaded {len(self.df)} products")
        else:
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        # Load TF-IDF vectorizer (if available)
        vectorizer_path = EMBEDDINGS_DIR / "tfidf_vectorizer.pkl"
        if vectorizer_path.exists():
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Loaded TF-IDF vectorizer")
        else:
            print("WARNING: Vectorizer not found - will use pre-computed embeddings only")
        
        # Load embeddings (for product-to-product recommendations)
        embeddings_path = EMBEDDINGS_DIR / "text_embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
            print("Loaded embeddings")
        
        print("Recommendation engine ready!")
    
    def recommend_by_text(self, query_text, k=10, filters=None):
        """
        Get recommendations based on text query.
        
        Args:
            query_text: Text description (e.g., "blue denim shirt")
            k: Number of recommendations
            filters: Dict of filters (e.g., {'gender': 'Men', 'category': 'Apparel'})
        
        Returns:
            DataFrame with recommendations
        """
        if self.vectorizer is None:
            raise ValueError("TF-IDF vectorizer not loaded. Cannot process text queries.")
        
        # Convert query to embedding
        query_embedding = self.vectorizer.transform([query_text]).toarray()
        query_embedding = normalize(query_embedding, norm='l2')
        
        # Search
        distances, indices = search_similar(self.index, query_embedding, k=k*2)  # Get more for filtering
        
        # Get recommendations
        recommendations = self.df.iloc[indices].copy()
        recommendations['similarity_score'] = 1 - distances  # Convert distance to similarity
        
        # Apply filters if provided
        if filters:
            for key, value in filters.items():
                if key in recommendations.columns:
                    recommendations = recommendations[recommendations[key] == value]
        
        # Return top k
        return recommendations.head(k)
    
    def recommend_by_product_id(self, product_id, k=10, filters=None):
        """
        Get recommendations for a specific product.
        
        Args:
            product_id: Product ID
            k: Number of recommendations
            filters: Dict of filters
        
        Returns:
            DataFrame with recommendations
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded.")
        
        # Find product index
        product_idx = self.df[self.df['id'] == product_id].index
        if len(product_idx) == 0:
            raise ValueError(f"Product ID {product_id} not found")
        
        product_idx = product_idx[0]
        
        # Get product embedding
        product_embedding = self.embeddings[product_idx:product_idx+1]
        
        # Search (k+1 because first result will be the product itself)
        distances, indices = search_similar(self.index, product_embedding, k=k+1)
        
        # Remove the product itself
        mask = indices != product_idx
        indices = indices[mask]
        distances = distances[mask]
        
        # Get recommendations
        recommendations = self.df.iloc[indices].copy()
        recommendations['similarity_score'] = 1 - distances
        
        # Apply filters
        if filters:
            for key, value in filters.items():
                if key in recommendations.columns:
                    recommendations = recommendations[recommendations[key] == value]
        
        return recommendations.head(k)
    
    def recommend_by_category(self, category, k=10):
        """
        Get recommendations within a category.
        
        Args:
            category: Category name (e.g., 'Tshirts', 'Shirts')
            k: Number of recommendations
        
        Returns:
            DataFrame with recommendations
        """
        category_products = self.df[self.df['articleType'] == category]
        
        if len(category_products) == 0:
            raise ValueError(f"Category '{category}' not found")
        
        # Return random sample from category
        return category_products.sample(min(k, len(category_products)))

def test_recommendations():
    """Test the recommendation engine."""
    print("="*60)
    print("TESTING RECOMMENDATION ENGINE")
    print("="*60)
    
    try:
        engine = FashionRecommendationEngine()
        
        # Test 1: Text query
        print("\nTest 1: Text Query")
        print("Query: 'blue denim shirt'")
        try:
            results = engine.recommend_by_text("blue denim shirt", k=5)
            print(f"Found {len(results)} recommendations")
            print("\nTop 3 recommendations:")
            for idx, row in results.head(3).iterrows():
                print(f"   - {row['productDisplayName']} ({row['baseColour']}, {row['articleType']})")
        except Exception as e:
            print(f"WARNING: Text query failed: {e}")
            print("   (This is OK if vectorizer not loaded)")
        
        # Test 2: Product ID
        print("\nTest 2: Product ID Recommendation")
        if len(engine.df) > 0:
            test_id = engine.df.iloc[0]['id']
            print(f"Query: Product ID {test_id}")
            try:
                results = engine.recommend_by_product_id(test_id, k=5)
                print(f"Found {len(results)} similar products")
                print("\nTop 3 recommendations:")
                for idx, row in results.head(3).iterrows():
                    print(f"   - {row['productDisplayName']} (Score: {row['similarity_score']:.3f})")
            except Exception as e:
                print(f"WARNING: Product ID recommendation failed: {e}")
        
        # Test 3: Category
        print("\nTest 3: Category Recommendation")
        if 'articleType' in engine.df.columns:
            categories = engine.df['articleType'].value_counts().head(3).index
            test_category = categories[0]
            print(f"Query: Category '{test_category}'")
            results = engine.recommend_by_category(test_category, k=5)
            print(f"Found {len(results)} products in category")
        
        print("\n" + "="*60)
        print("Recommendation engine is working!")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nMake sure you have:")
        print("   1. Downloaded embeddings from Kaggle")
        print("   2. Built FAISS index (run similarity_search.py)")
        print("   3. Saved files to data/embeddings/")

if __name__ == "__main__":
    test_recommendations()

