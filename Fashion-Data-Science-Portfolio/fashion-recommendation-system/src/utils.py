"""
Utility functions for the recommendation system.
"""

import torch
from pathlib import Path
import numpy as np
from PIL import Image

def get_device():
    """Get the best available device (CUDA or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path):
    """Load and preprocess an image."""
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def save_embeddings(embeddings, filepath):
    """Save embeddings to numpy file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, embeddings)
    print(f"Saved embeddings to {filepath}")
