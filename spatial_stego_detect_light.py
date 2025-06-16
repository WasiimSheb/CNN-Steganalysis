#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight SpatialStegoDetect - Faster CNN for Steganography Detection
Optimized for faster training on local computers
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
import time as tm

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def SpatialStegoDetectLight(img_size=256):
    """
    Lightweight CNN Architecture for Spatial Steganography Detection
    Much faster training with fewer parameters
    """
    tf.keras.backend.clear_session()

    # Input
    inputs = Input(shape=(img_size, img_size, 1), name="input_1")

    # Block 1: Initial feature extraction
    x = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2))(x)  # 64x64

    # Block 2: Feature extraction
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2))(x)  # 32x32

    # Block 3: Deeper features
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2))(x)  # 16x16

    # Block 4: High-level features
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2))(x)  # 8x8

    # Block 5: Final feature extraction
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Global Average Pooling instead of large dense layers
    x = GlobalAveragePooling2D()(x)

    # Small dense layers for classification
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(128)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    # Output layer
    outputs = Dense(2, activation='softmax', name="output_1")(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Optimizer - using Adam with higher learning rate for overfitting
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Lightweight SpatialStegoDetect model generated")
    return model

def train_light(model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=64, epochs=20, model_name=""):
    """Lightweight training function"""
    start_time = tm.time()
    
    # Create logs directory
    log_dir = f"logs/{model_name}_{time()}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir)
    filepath = f"{log_dir}/saved-model-{{epoch:02d}}-{{val_accuracy:.2f}}.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max')
    
    # Remove early stopping to allow overfitting
    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    
    # Reduce learning rate on plateau (less aggressive to allow overfitting)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=8, min_lr=1e-5)
    
    # Training
    history = model.fit(
        X_train, y_train, 
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_valid, y_valid),
        callbacks=[tensorboard, checkpoint, reduce_lr],
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Create results directory
    results_dir = f"data/{model_name}/"
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}training_results_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    training_time = tm.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"{'='*50}")
    
    return {'test_loss': test_loss, 'test_accuracy': test_accuracy}, history

def main():
    """Main function for lightweight training"""
    print("Starting Lightweight SpatialStegoDetect Training...")
    print("This version is optimized for faster training on local computers")
    print("="*60)
    
    # Load data
    try:
        print("Loading BOSSbase dataset...")
        X_train = np.load('X_training.npy')
        y_train = np.load('y_training.npy')
        X_valid = np.load('X_validating.npy')
        y_valid = np.load('y_validating.npy')
        X_test = np.load('X_testing.npy')
        y_test = np.load('y_testing.npy')
        
        print(f"✅ Dataset loaded successfully:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Validation: {X_valid.shape[0]} samples")
        print(f"   Testing: {X_test.shape[0]} samples")
        
    except FileNotFoundError:
        print("❌ Dataset files not found!")
        print("Please run: python load_boss_dataset.py")
        return
    
    # Create lightweight model
    print("\nCreating lightweight model...")
    model = SpatialStegoDetectLight(img_size=256)
    
    print(f"\nModel Summary:")
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Training parameters - increased epochs to encourage overfitting
    EPOCHS = 50
    BATCH_SIZE = 64
    model_name = "LightSpatialStego"
    
    print(f"\nStarting training:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Optimizer: Adam (lr=0.001)")
    print("   Features: Early stopping, Learning rate reduction")
    
    # Train model
    results, history = train_light(
        model, X_train, y_train, X_valid, y_valid, X_test, y_test,
        batch_size=BATCH_SIZE, epochs=EPOCHS, model_name=model_name
    )
    
    # Save final model
    final_model_path = f"models/{model_name}_final.keras"
    os.makedirs("models", exist_ok=True)
    model.save(final_model_path)
    print(f"\n✅ Final model saved to: {final_model_path}")
    
    # Performance summary
    accuracy_percent = results['test_accuracy'] * 100
    if accuracy_percent > 70:
        print(f"🎉 Excellent performance! {accuracy_percent:.1f}% accuracy")
    elif accuracy_percent > 60:
        print(f"✅ Good performance! {accuracy_percent:.1f}% accuracy")
    elif accuracy_percent > 55:
        print(f"⚠️  Moderate performance: {accuracy_percent:.1f}% accuracy")
    else:
        print(f"❌ Poor performance: {accuracy_percent:.1f}% accuracy - may need more data or tuning")

if __name__ == "__main__":
    main() 