# Multimodal Fashion Recommendation System

A production-ready recommendation system that processes 44,424 fashion products to provide real-time product recommendations using text embeddings and similarity search.

## Overview

This system extracts text features from product descriptions using TF-IDF vectorization, builds a FAISS index for fast similarity search, and provides recommendations through a Streamlit web interface.

## Features

- Text feature extraction using TF-IDF (512-dim embeddings)
- FAISS-based similarity search for fast retrieval
- Product-to-product recommendations
- Text-based query recommendations
- Interactive web interface
- Real-time search (<100ms latency)

## Dataset

- Source: Fashion Product Images Dataset (Kaggle)
- Size: 44,424 products
- Features: Product images, metadata, descriptions, attributes
- License: CC0 (Public Domain)

## Project Structure

```
fashion-recommendation-system/
├── README.md
├── requirements.txt
├── src/
│   ├── recommendation_engine.py
│   ├── similarity_search.py
│   ├── feature_extraction.py
│   ├── data_exploration.py
│   ├── build_recommendation_system.py
│   ├── config.py
│   └── utils.py
├── app/
│   └── streamlit_app_simple.py
├── data/
│   ├── raw/
│   └── embeddings/
└── kaggle_notebook.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sirisha0812/Fashion-Data-Science-Portfolio.git
cd Fashion-Data-Science-Portfolio/fashion-recommendation-system
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Extract Features

Run kaggle_notebook.py in Kaggle to extract text embeddings. This generates text_embeddings.npy.

### Step 2: Build Recommendation System

```bash
python3 src/build_recommendation_system.py
```

### Step 3: Run Web Application

```bash
streamlit run app/streamlit_app_simple.py
```

## Methods

- Text Feature Extraction: TF-IDF vectorization (512-dim)
- Similarity Search: FAISS L2 distance
- Search Speed: <100ms per query

## Results

- Indexed 44,424 products
- Real-time similarity search
- Working web interface

## Technology Stack

- Python 3.8+
- Pandas, NumPy
- Scikit-learn
- FAISS
- Streamlit
- PyTorch

## License

This project is provided for educational purposes.

Last Updated: November 2024
