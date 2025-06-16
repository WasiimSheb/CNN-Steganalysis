#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loader for Steganography Detection
Helps prepare and load datasets for training
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob
from tqdm import tqdm

def load_images_from_folder(folder_path, img_size=256, max_images=None):
    """
    Load images from a folder and resize them
    
    Args:
        folder_path: Path to folder containing images
        img_size: Target size for images (default 256x256)
        max_images: Maximum number of images to load (None for all)
    
    Returns:
        numpy array of images
    """
    images = []
    image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                  glob.glob(os.path.join(folder_path, "*.png")) + \
                  glob.glob(os.path.join(folder_path, "*.bmp")) + \
                  glob.glob(os.path.join(folder_path, "*.tiff"))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Loading {len(image_files)} images from {folder_path}")
    
    for img_path in tqdm(image_files):
        try:
            # Load image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize image
                img = cv2.resize(img, (img_size, img_size))
                # Normalize to [0, 1]
                img = img.astype(np.float32) / 255.0
                # Add channel dimension
                img = np.expand_dims(img, axis=-1)
                images.append(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    return np.array(images)

def prepare_steganography_dataset(cover_folder, stego_folder, img_size=256, test_size=0.2, val_size=0.1, max_images_per_class=None):
    """
    Prepare a steganography dataset from cover and stego image folders
    
    Args:
        cover_folder: Path to folder containing cover images
        stego_folder: Path to folder containing stego images
        img_size: Target image size
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        max_images_per_class: Maximum images per class (None for all)
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    
    print("Loading cover images...")
    cover_images = load_images_from_folder(cover_folder, img_size, max_images_per_class)
    
    print("Loading stego images...")
    stego_images = load_images_from_folder(stego_folder, img_size, max_images_per_class)
    
    # Ensure equal number of cover and stego images
    min_images = min(len(cover_images), len(stego_images))
    cover_images = cover_images[:min_images]
    stego_images = stego_images[:min_images]
    
    print(f"Using {min_images} images per class")
    
    # Combine images and create labels
    X = np.concatenate([cover_images, stego_images], axis=0)
    y = np.concatenate([np.zeros(len(cover_images)), np.ones(len(stego_images))], axis=0)
    
    # Convert labels to one-hot encoding
    y_onehot = np.zeros((len(y), 2))
    y_onehot[y == 0, 0] = 1  # Cover images
    y_onehot[y == 1, 1] = 1  # Stego images
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=test_size, random_state=42, stratify=y
    )
    
    # Split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"Dataset prepared:")
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Testing: {X_test.shape[0]} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_dataset(X_train, y_train, X_val, y_val, X_test, y_test, save_dir="./"):
    """
    Save the prepared dataset as numpy files
    
    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Dataset splits
        save_dir: Directory to save the files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, 'X_training.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_training.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_validating.npy'), X_val)
    np.save(os.path.join(save_dir, 'y_validating.npy'), y_val)
    np.save(os.path.join(save_dir, 'X_testing.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_testing.npy'), y_test)
    
    print(f"Dataset saved to {save_dir}")

def create_sample_boss_dataset(num_samples=1000, img_size=256, save_dir="./"):
    """
    Create a sample dataset mimicking BOSSbase structure for testing
    
    Args:
        num_samples: Number of samples per class
        img_size: Image size
        save_dir: Directory to save the dataset
    """
    print(f"Creating sample BOSSbase-like dataset with {num_samples} samples per class...")
    
    # Create random cover images (more structured noise)
    cover_images = []
    for i in tqdm(range(num_samples), desc="Creating cover images"):
        # Create more realistic image-like patterns
        img = np.random.randn(img_size, img_size) * 0.3 + 0.5
        # Add some structure
        x, y = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size))
        img += 0.1 * np.sin(10 * x) * np.cos(10 * y)
        img = np.clip(img, 0, 1)
        img = np.expand_dims(img, axis=-1)
        cover_images.append(img)
    
    # Create stego images (cover + slight modifications)
    stego_images = []
    for i in tqdm(range(num_samples), desc="Creating stego images"):
        # Start with a cover-like image
        img = np.random.randn(img_size, img_size) * 0.3 + 0.5
        x, y = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size))
        img += 0.1 * np.sin(10 * x) * np.cos(10 * y)
        # Add steganographic modifications (subtle noise)
        img += np.random.randn(img_size, img_size) * 0.05
        img = np.clip(img, 0, 1)
        img = np.expand_dims(img, axis=-1)
        stego_images.append(img)
    
    cover_images = np.array(cover_images, dtype=np.float32)
    stego_images = np.array(stego_images, dtype=np.float32)
    
    # Combine and create labels
    X = np.concatenate([cover_images, stego_images], axis=0)
    y = np.concatenate([np.zeros(num_samples), np.ones(num_samples)], axis=0)
    
    # Convert to one-hot
    y_onehot = np.zeros((len(y), 2))
    y_onehot[y == 0, 0] = 1
    y_onehot[y == 1, 1] = 1
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=np.argmax(y_train, axis=1)
    )
    
    # Save dataset
    save_dataset(X_train, y_train, X_val, y_val, X_test, y_test, save_dir)
    
    print("Sample dataset created and saved!")
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    # Example usage
    print("Data Loader for Steganography Detection")
    print("=" * 50)
    
    # Check if user has real data
    cover_folder = input("Enter path to cover images folder (or press Enter to create sample data): ").strip()
    
    if cover_folder and os.path.exists(cover_folder):
        stego_folder = input("Enter path to stego images folder: ").strip()
        if stego_folder and os.path.exists(stego_folder):
            # Load real data
            X_train, y_train, X_val, y_val, X_test, y_test = prepare_steganography_dataset(
                cover_folder, stego_folder, max_images_per_class=1000
            )
            save_dataset(X_train, y_train, X_val, y_val, X_test, y_test)
        else:
            print("Stego folder not found. Creating sample data...")
            create_sample_boss_dataset()
    else:
        print("Creating sample BOSSbase-like dataset...")
        create_sample_boss_dataset() 