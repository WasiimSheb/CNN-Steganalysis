#!/usr/bin/env python3
"""
BOSSbase Dataset Loader for Steganography Detection
Loads the actual BOSSbase dataset from the archive folder
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm

def load_boss_dataset(train_cover_dir, train_stego_dir, test_cover_dir, test_stego_dir, 
                     img_size=256, max_images_per_class=None):
    """
    Load BOSSbase dataset from the archive structure
    
    Args:
        train_cover_dir: Path to training cover images
        train_stego_dir: Path to training stego images  
        test_cover_dir: Path to test cover images
        test_stego_dir: Path to test stego images
        img_size: Target image size (should be 256 for BOSSbase)
        max_images_per_class: Limit number of images (None for all)
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    
    def load_images_from_folder(folder_path, max_images=None):
        """Load images from a folder"""
        images = []
        image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                      glob.glob(os.path.join(folder_path, "*.png")) + \
                      glob.glob(os.path.join(folder_path, "*.bmp")) + \
                      glob.glob(os.path.join(folder_path, "*.pgm"))  # BOSSbase uses .pgm
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Loading {len(image_files)} images from {folder_path}")
        
        for img_path in tqdm(image_files):
            try:
                # Load image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # BOSSbase images should already be 256x256, but resize just in case
                    if img.shape != (img_size, img_size):
                        img = cv2.resize(img, (img_size, img_size))
                    # Normalize to [0, 1]
                    img = img.astype(np.float32) / 255.0
                    # Add channel dimension
                    img = np.expand_dims(img, axis=-1)
                    images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        return np.array(images)
    
    # Load training data
    print("Loading training cover images...")
    train_cover = load_images_from_folder(train_cover_dir, max_images_per_class)
    
    print("Loading training stego images...")
    train_stego = load_images_from_folder(train_stego_dir, max_images_per_class)
    
    # Load test data
    print("Loading test cover images...")
    test_cover = load_images_from_folder(test_cover_dir, max_images_per_class)
    
    print("Loading test stego images...")
    test_stego = load_images_from_folder(test_stego_dir, max_images_per_class)
    
    # Ensure equal numbers
    min_train = min(len(train_cover), len(train_stego))
    min_test = min(len(test_cover), len(test_stego))
    
    train_cover = train_cover[:min_train]
    train_stego = train_stego[:min_train]
    test_cover = test_cover[:min_test]
    test_stego = test_stego[:min_test]
    
    print(f"Using {min_train} training images per class")
    print(f"Using {min_test} test images per class")
    
    # Combine training data
    X_train_full = np.concatenate([train_cover, train_stego], axis=0)
    y_train_full = np.concatenate([np.zeros(len(train_cover)), np.ones(len(train_stego))], axis=0)
    
    # Combine test data
    X_test = np.concatenate([test_cover, test_stego], axis=0)
    y_test = np.concatenate([np.zeros(len(test_cover)), np.ones(len(test_stego))], axis=0)
    
    # Convert labels to one-hot encoding
    def to_onehot(y):
        y_onehot = np.zeros((len(y), 2))
        y_onehot[y == 0, 0] = 1  # Cover images
        y_onehot[y == 1, 1] = 1  # Stego images
        return y_onehot
    
    y_train_full = to_onehot(y_train_full)
    y_test = to_onehot(y_test)
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, 
        stratify=np.argmax(y_train_full, axis=1)
    )
    
    print(f"\nFinal dataset splits:")
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples") 
    print(f"Testing: {X_test.shape[0]} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_boss_dataset(X_train, y_train, X_val, y_val, X_test, y_test, save_dir="./"):
    """Save the BOSSbase dataset as numpy files"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("Saving dataset as .npy files...")
    np.save(os.path.join(save_dir, 'X_training.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_training.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_validating.npy'), X_val)
    np.save(os.path.join(save_dir, 'y_validating.npy'), y_val)
    np.save(os.path.join(save_dir, 'X_testing.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_testing.npy'), y_test)
    
    print(f"Dataset saved to {save_dir}")
    print("Files created:")
    print("- X_training.npy, y_training.npy")
    print("- X_validating.npy, y_validating.npy") 
    print("- X_testing.npy, y_testing.npy")

if __name__ == "__main__":
    print("Loading BOSSbase Dataset...")
    print("=" * 50)
    
    # Define paths to your BOSSbase dataset
    train_cover_dir = "archive/boss_256_0.4/cover"
    train_stego_dir = "archive/boss_256_0.4/stego"
    test_cover_dir = "archive/boss_256_0.4_test/cover"
    test_stego_dir = "archive/boss_256_0.4_test/stego"
    
    # Load the dataset (limit to 1000 per class for faster initial testing)
    # Remove max_images_per_class=1000 to use the full dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_boss_dataset(
        train_cover_dir, train_stego_dir, test_cover_dir, test_stego_dir,
        max_images_per_class=1000  # Remove this line to use full dataset
    )
    
    # Save the dataset
    save_boss_dataset(X_train, y_train, X_val, y_val, X_test, y_test)
    
    print("\n✅ BOSSbase dataset loaded and saved successfully!")
    print("You can now run: python spatial_stego_detect.py") 