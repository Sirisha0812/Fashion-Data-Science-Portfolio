"""
Data exploration script for Fashion Product Images dataset.
Run this to understand your dataset before starting feature extraction.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

def explore_dataset():
    """Explore the fashion product images dataset."""
    
    print("=" * 60)
    print("FASHION PRODUCT IMAGES DATASET EXPLORATION")
    print("=" * 60)
    
    # Try to find metadata file
    possible_paths = [
        RAW_DATA_DIR / "styles.csv",
        RAW_DATA_DIR / "products.csv",
        RAW_DATA_DIR / "metadata.csv",
    ]
    
    metadata_path = None
    for path in possible_paths:
        if path.exists():
            metadata_path = path
            break
    
    if not metadata_path:
        print(f"\nERROR: Metadata file not found!")
        print(f"Please download the Fashion Product Images dataset from Kaggle")
        print(f"\nExpected locations:")
        for path in possible_paths:
            print(f"  - {path}")
        return None
    
    print(f"\nFound metadata: {metadata_path}")
    
    # Load metadata
    try:
        df = pd.read_csv(metadata_path, on_bad_lines='skip', low_memory=False)
        print(f"\nDataset Shape: {df.shape}")
        print(f"   Rows: {df.shape[0]:,}")
        print(f"   Columns: {df.shape[1]}")
    except Exception as e:
        print(f"\nERROR: Error loading metadata: {e}")
        return None
    
    # Display columns
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    # Missing values
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col in df.columns:
            if missing[col] > 0:
                pct = (missing[col] / len(df)) * 100
                print(f"   {col}: {missing[col]:,} ({pct:.1f}%)")
    else:
        print("   No missing values!")
    
    # Check for images
    print(f"\nImage Directory Check:")
    possible_img_dirs = [
        RAW_DATA_DIR / "images",
        DATA_DIR / "images",
        RAW_DATA_DIR,
    ]
    
    images_found = False
    for img_dir in possible_img_dirs:
        if img_dir.exists():
            image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            if image_files:
                print(f"   Found {len(image_files):,} images in {img_dir}")
                images_found = True
                break
    
    if not images_found:
        print(f"   WARNING: No images found. Download images from Kaggle.")
        print(f"   Expected locations:")
        for img_dir in possible_img_dirs:
            print(f"      - {img_dir}")
    
    # Category analysis
    print(f"\nCategory Analysis:")
    if 'articleType' in df.columns:
        print(f"   Article Types:")
        value_counts = df['articleType'].value_counts()
        print(f"      Unique: {df['articleType'].nunique()}")
        print(f"      Top 10:")
        for val, count in value_counts.head(10).items():
            print(f"         {val}: {count:,}")
    
    if 'masterCategory' in df.columns:
        print(f"\n   Master Categories:")
        cat_counts = df['masterCategory'].value_counts()
        for cat, count in cat_counts.items():
            pct = (count / len(df)) * 100
            print(f"      {cat}: {count:,} ({pct:.1f}%)")
    
    if 'subCategory' in df.columns:
        print(f"\n   Sub Categories (Top 10):")
        sub_counts = df['subCategory'].value_counts()
        for sub, count in sub_counts.head(10).items():
            print(f"      {sub}: {count:,}")
    
    # Color analysis
    if 'baseColour' in df.columns:
        print(f"\nColor Analysis:")
        color_counts = df['baseColour'].value_counts()
        print(f"   Unique colors: {df['baseColour'].nunique()}")
        print(f"   Top 15 colors:")
        for color, count in color_counts.head(15).items():
            print(f"      {color}: {count:,}")
    
    # Gender analysis
    if 'gender' in df.columns:
        print(f"\nGender Distribution:")
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            pct = (count / len(df)) * 100
            print(f"   {gender}: {count:,} ({pct:.1f}%)")
    
    # Year/Season analysis
    if 'year' in df.columns:
        print(f"\nYear Distribution:")
        year_counts = df['year'].value_counts().sort_index()
        print(f"   Years: {df['year'].min()} - {df['year'].max()}")
        print(f"   Top years:")
        for year, count in year_counts.head(10).items():
            print(f"      {year}: {count:,}")
    
    if 'season' in df.columns:
        print(f"\nSeason Distribution:")
        season_counts = df['season'].value_counts()
        for season, count in season_counts.items():
            pct = (count / len(df)) * 100
            print(f"   {season}: {count:,} ({pct:.1f}%)")
    
    # Product names
    if 'productDisplayName' in df.columns:
        print(f"\nProduct Names:")
        non_null = df['productDisplayName'].notna().sum()
        avg_len = df['productDisplayName'].astype(str).str.len().mean()
        print(f"   Non-null: {non_null:,} ({non_null/len(df)*100:.1f}%)")
        print(f"   Avg length: {avg_len:.1f} characters")
        if non_null > 0:
            sample = df['productDisplayName'].dropna().iloc[0]
            print(f"   Sample: {str(sample)[:80]}...")
    
    print(f"\n" + "=" * 60)
    print("Exploration complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"   1. Review the dataset structure above")
    print(f"   2. Download images if not already done")
    print(f"   3. Start feature extraction (use Kaggle for GPU)")
    print(f"   4. Run kaggle_notebook.py to extract features")
    
    return df


if __name__ == "__main__":
    df = explore_dataset()
    
    # Save summary
    if df is not None:
        summary_path = PROCESSED_DATA_DIR / "dataset_summary.txt"
        try:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                f.write(f"Dataset Shape: {df.shape}\n")
                f.write(f"Columns: {', '.join(df.columns)}\n")
                f.write(f"\nMissing Values:\n{df.isnull().sum().to_string()}\n")
                if 'articleType' in df.columns:
                    f.write(f"\nArticle Type Distribution:\n")
                    f.write(df['articleType'].value_counts().to_string())
            print(f"\nSummary saved to: {summary_path}")
        except Exception as e:
            print(f"\nWARNING: Could not save summary: {e}")
