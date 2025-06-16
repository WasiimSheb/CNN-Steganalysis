#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Steganography Detection CNN
Specialized for detecting subtle steganographic changes
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
import time as tm

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Add
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_residual_block(x, filters, kernel_size=3):
    """Create a residual block for better gradient flow"""
    shortcut = x
    
    # First conv layer
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # Second conv layer
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Add shortcut connection if dimensions match
    if shortcut.shape[-1] == filters:
        x = Add()([x, shortcut])
    
    x = LeakyReLU()(x)
    return x

def StegoDetectOptimized(img_size=256):
    """
    Optimized CNN for steganography detection with residual connections
    """
    tf.keras.backend.clear_session()

    inputs = Input(shape=(img_size, img_size, 1), name="input_1")

    # Initial feature extraction with smaller stride to preserve details
    x = Conv2D(32, (7, 7), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # Residual blocks for better feature learning
    x = create_residual_block(x, 32)
    x = MaxPooling2D((2, 2))(x)  # 128x128
    
    x = create_residual_block(x, 64)
    x = create_residual_block(x, 64)
    x = MaxPooling2D((2, 2))(x)  # 64x64
    
    x = create_residual_block(x, 128)
    x = create_residual_block(x, 128)
    x = MaxPooling2D((2, 2))(x)  # 32x32
    
    x = create_residual_block(x, 256)
    x = create_residual_block(x, 256)
    x = MaxPooling2D((2, 2))(x)  # 16x16
    
    # Final feature extraction
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # Global pooling
    x = GlobalAveragePooling2D()(x)
    
    # Classification layers with dropout
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(64)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use a lower learning rate for better convergence
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Optimized StegoDetect model created")
    return model

def train_optimized(model, X_train, y_train, X_valid, y_valid, X_test, y_test, epochs=30):
    """Training with data augmentation and advanced callbacks"""
    start_time = tm.time()
    
    # Data augmentation to help the model generalize
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        zoom_range=0.05,
        fill_mode='nearest'
    )
    
    # Create logs directory
    log_dir = f"logs/OptimizedStego_{time()}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Advanced callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir),
        tf.keras.callbacks.ModelCheckpoint(
            f"{log_dir}/best_model.keras",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Training with data augmentation
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train) // 32,
        epochs=epochs,
        validation_data=(X_valid, y_valid),
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Plot results
    results_dir = "data/OptimizedStego/"
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Learning rate plot
    plt.subplot(1, 3, 3)
    plt.plot(history.history['learning_rate'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    training_time = tm.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Final Test Loss: {test_loss:.4f}")
    
    # Performance analysis
    if test_accuracy > 0.75:
        print("🎉 Excellent performance! The model can detect steganography well.")
    elif test_accuracy > 0.65:
        print("✅ Good performance! The model shows promise for stego detection.")
    elif test_accuracy > 0.55:
        print("⚠️  Moderate performance. The steganographic changes might be very subtle.")
    else:
        print("❌ Poor performance. This suggests the steganographic algorithm is very sophisticated.")
    
    print(f"{'='*60}")
    
    return {'test_loss': test_loss, 'test_accuracy': test_accuracy}, history

def analyze_predictions(model, X_test, y_test, num_samples=10):
    """Analyze model predictions to understand what it's learning"""
    predictions = model.predict(X_test[:num_samples])
    
    print("\nPrediction Analysis:")
    print("Sample | True Label | Predicted | Confidence")
    print("-" * 45)
    
    for i in range(num_samples):
        true_label = "Cover" if y_test[i][0] == 1 else "Stego"
        pred_label = "Cover" if predictions[i][0] > predictions[i][1] else "Stego"
        confidence = max(predictions[i])
        
        print(f"{i+1:6d} | {true_label:10s} | {pred_label:9s} | {confidence:.3f}")

def main():
    """Main training function"""
    print("Optimized Steganography Detection Training")
    print("Specialized for detecting subtle steganographic changes")
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
        
        print(f"✅ Dataset loaded:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Validation: {X_valid.shape[0]} samples")
        print(f"   Testing: {X_test.shape[0]} samples")
        
    except FileNotFoundError:
        print("❌ Dataset not found! Please run: python load_boss_dataset.py")
        return
    
    # Create optimized model
    print("\nCreating optimized model with residual connections...")
    model = StegoDetectOptimized()
    
    print("\nModel Summary:")
    model.summary()
    
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Train model
    print(f"\nStarting optimized training with data augmentation...")
    results, history = train_optimized(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    
    # Analyze predictions
    analyze_predictions(model, X_test, y_test)
    
    # Save final model
    final_model_path = "models/OptimizedStego_final.keras"
    os.makedirs("models", exist_ok=True)
    model.save(final_model_path)
    print(f"\n✅ Model saved to: {final_model_path}")

if __name__ == "__main__":
    main() 