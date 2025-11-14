"""
Feature extraction for multimodal fashion recommendations.
Extracts visual and textual embeddings from product images and descriptions.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

from config import *
from utils import get_device, load_image, save_embeddings

class VisualFeatureExtractor:
    """Extract visual features from fashion product images."""
    
    def __init__(self, model_name='resnet50', device=None):
        self.device = device or get_device()
        self.model_name = model_name
        self.model = self._load_model()
        self.transform = self._get_transform()
        
    def _load_model(self):
        """Load pre-trained CNN model."""
        if self.model_name == 'resnet50':
            # Use newer weights API
            try:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            except:
                # Fallback for older torchvision versions
                model = models.resnet50(pretrained=True)
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == 'efficientnet_b0':
            # Note: Requires timm library: pip install timm
            try:
                import timm
                model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
            except ImportError:
                print("timm not installed. Using ResNet50 instead.")
                model = models.resnet50(pretrained=True)
                model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        model.eval()
        model.to(self.device)
        return model
    
    def _get_transform(self):
        """Get image preprocessing transform."""
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_path):
        """Extract features from a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
                # Flatten and normalize
                features = features.squeeze().cpu().numpy()
                features = features.flatten()
                # Normalize to unit length
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm
                return features
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def extract_batch(self, image_paths, batch_size=32):
        """Extract features from multiple images in batches."""
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_indices = []
            
            for idx, path in enumerate(batch_paths):
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.transform(image).to(self.device)
                    batch_images.append(image_tensor)
                    valid_indices.append(i + idx)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if batch_images:
                batch_tensor = torch.stack(batch_images)
                with torch.no_grad():
                    features = self.model(batch_tensor)
                    features = features.squeeze().cpu().numpy()
                    # Reshape and normalize
                    features = features.reshape(len(batch_images), -1)
                    norms = np.linalg.norm(features, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    features = features / norms
                    all_features.extend(features)
        
        return np.array(all_features), valid_indices


class TextFeatureExtractor:
    """Extract textual features from product descriptions."""
    
    def __init__(self, model_name='distilbert-base-uncased', device=None):
        self.device = device or get_device()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
    
    def extract_features(self, text):
        """Extract features from a single text."""
        if not text or text.strip() == "":
            # Return zero vector if no text
            return np.zeros(self.model.config.hidden_size)
        
        try:
            inputs = self.tokenizer(
                text,
                max_length=MAX_TEXT_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                features = features.flatten()
                # Normalize
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm
                return features
        except Exception as e:
            print(f"Error processing text: {e}")
            return np.zeros(self.model.config.hidden_size)
    
    def extract_batch(self, texts, batch_size=32):
        """Extract features from multiple texts in batches."""
        all_features = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting text features"):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                max_length=MAX_TEXT_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                # Normalize
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                norms[norms == 0] = 1
                features = features / norms
                all_features.extend(features)
        
        return np.array(all_features)


if __name__ == "__main__":
    # Example usage
    print("Testing feature extractors...")
    
    # Test visual extractor
    visual_extractor = VisualFeatureExtractor()
    print(f"Visual model loaded: {visual_extractor.model_name}")
    
    # Test text extractor
    text_extractor = TextFeatureExtractor()
    print(f"Text model loaded: {text_extractor.model_name}")
    
    # Test with sample
    test_text = "A stylish blue denim jacket with modern design"
    text_features = text_extractor.extract_features(test_text)
    print(f"Text features shape: {text_features.shape}")
    
    print("Feature extractors ready!")

