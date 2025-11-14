"""
Streamlit Web App - Simplified Version
Works around file system timeout issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import faiss with error handling for Streamlit Cloud
try:
    import faiss
    faiss_available = True
except ImportError:
    faiss_available = False
    st.error("FAISS library not found!")
    st.markdown("""
    **The app requires FAISS to be installed.**
    
    For Streamlit Cloud, ensure `requirements.txt` in the repository root includes:
    ```
    faiss-cpu
    ```
    
    The app will not work without FAISS. Please check your deployment logs.
    """)
    st.stop()

# Page config
st.set_page_config(
    page_title="Fashion Recommendation System",
    layout="wide"
)

# Title
st.title("Fashion Recommendation System")
st.markdown("Find similar fashion products using AI-powered recommendations")

# Load data directly (avoiding import issues)
@st.cache_data
def load_metadata():
    """Load product metadata."""
    # Try multiple paths for Streamlit Cloud compatibility
    possible_paths = [
        Path('fashion-recommendation-system/data/raw/styles.csv'),  # From repo root
        Path('data/raw/styles.csv'),  # From app directory
        Path(__file__).parent.parent / 'data' / 'raw' / 'styles.csv'  # Relative to script
    ]
    
    for csv_path in possible_paths:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, on_bad_lines='skip', low_memory=False)
                return df
            except TypeError:
                df = pd.read_csv(csv_path, error_bad_lines=False, low_memory=False)
                return df
    
    return None

@st.cache_resource
def load_index():
    """Load FAISS index."""
    # Try multiple paths for Streamlit Cloud compatibility
    possible_paths = [
        Path('fashion-recommendation-system/data/embeddings/faiss_index.bin'),  # From repo root
        Path('data/embeddings/faiss_index.bin'),  # From app directory
        Path(__file__).parent.parent / 'data' / 'embeddings' / 'faiss_index.bin'  # Relative to script
    ]
    
    for index_path in possible_paths:
        if index_path.exists():
            return faiss.read_index(str(index_path))
    return None

@st.cache_data
def load_embeddings():
    """Load embeddings."""
    # Try multiple paths for Streamlit Cloud compatibility
    possible_paths = [
        Path('fashion-recommendation-system/data/embeddings/text_embeddings.npy'),  # From repo root
        Path('data/embeddings/text_embeddings.npy'),  # From app directory
        Path(__file__).parent.parent / 'data' / 'embeddings' / 'text_embeddings.npy'  # Relative to script
    ]
    
    for emb_path in possible_paths:
        if emb_path.exists():
            return np.load(emb_path)
    return None

# Load data
with st.spinner("Loading data..."):
    df = load_metadata()
    index = load_index()
    embeddings = load_embeddings()

if index is None or embeddings is None:
    st.error("ERROR: FAISS index or embeddings not found!")
    st.info("Please run: python3 src/build_recommendation_system.py")
    st.stop()

st.success(f"Loaded {len(df):,} products")

# Sidebar
st.sidebar.header("Search Options")
search_type = st.sidebar.radio("Search Type", ["Product ID", "Category"])
num_results = st.sidebar.slider("Number of Results", 5, 20, 10)

# Filters
st.sidebar.header("Filters")
if 'gender' in df.columns:
    gender_filter = st.sidebar.selectbox("Gender", ["All"] + list(df['gender'].unique()))
else:
    gender_filter = "All"

# Main content
if search_type == "Product ID":
    st.header("Product ID Search")
    
    # Show sample IDs
    st.caption("Sample Product IDs:")
    sample_ids = df['id'].head(10).tolist()
    st.code(", ".join(map(str, sample_ids)))
    
    product_id = st.number_input(
        "Enter Product ID:",
        min_value=int(df['id'].min()),
        max_value=int(df['id'].max()),
        value=int(df['id'].iloc[0]) if len(df) > 0 else 0
    )
    
    if st.button("Find Similar", type="primary"):
        with st.spinner("Finding similar products..."):
            # Find product index
            product_idx = df[df['id'] == product_id].index
            if len(product_idx) == 0:
                st.error(f"Product ID {product_id} not found")
            else:
                product_idx = product_idx[0]
                
                # Get product embedding
                product_embedding = embeddings[product_idx:product_idx+1].astype('float32')
                
                # Search
                k = num_results + 1
                distances, indices = index.search(product_embedding, k)
                
                # Remove the product itself
                mask = indices[0] != product_idx
                indices_filtered = indices[0][mask][:num_results]
                distances_filtered = distances[0][mask][:num_results]
                
                # Get recommendations
                recommendations = df.iloc[indices_filtered].copy()
                recommendations['similarity'] = 1 - distances_filtered
                
                # Apply gender filter
                if gender_filter != "All" and 'gender' in recommendations.columns:
                    recommendations = recommendations[recommendations['gender'] == gender_filter]
                
                if len(recommendations) > 0:
                    st.success(f"Found {len(recommendations)} similar products!")
                    
                    # Show original
                    original = df.iloc[product_idx]
                    st.info(f"**Original:** {original['productDisplayName']}")
                    st.divider()
                    
                    # Show recommendations
                    for idx, row in recommendations.iterrows():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("Similarity", f"{row['similarity']:.2f}")
                        with col2:
                            st.write(f"**{row['productDisplayName']}**")
                            if 'articleType' in row:
                                st.caption(f"Type: {row['articleType']}")
                            if 'baseColour' in row:
                                st.caption(f"Color: {row['baseColour']}")
                        st.divider()
                else:
                    st.warning("No recommendations found. Try different filters.")

else:
    st.header("Category Browse")
    
    if 'articleType' in df.columns:
        categories = df['articleType'].value_counts().head(20).index.tolist()
        selected_category = st.selectbox("Select Category:", categories)
        
        if st.button("Browse", type="primary"):
            category_products = df[df['articleType'] == selected_category]
            if gender_filter != "All":
                category_products = category_products[category_products['gender'] == gender_filter]
            
            results = category_products.sample(min(num_results, len(category_products)))
            
            st.success(f"Found {len(results)} products")
            
            cols = st.columns(3)
            for idx, (i, row) in enumerate(results.iterrows()):
                with cols[idx % 3]:
                    st.write(f"**{row['productDisplayName']}**")
                    if 'baseColour' in row:
                        st.caption(f"Color: {row['baseColour']}")
                    st.divider()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"### Dataset: {len(df):,} products")

