#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpatialStegoDetect - CNN Architecture for Steganography Detection
"""

import os
import random
import glob
import numpy as np
import pandas as pd
from scipy import ndimage, signal
from skimage.util.shape import view_as_blocks
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from time import time
import time as tm
import datetime

from sklearn.model_selection import train_test_split
# from keras.utils import np_utils  # Removed - not needed in newer versions
from tensorflow.keras.initializers import Constant, RandomNormal, glorot_normal
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Activation, Conv2D, LeakyReLU, DepthwiseConv2D, SeparableConv2D, AveragePooling2D, Concatenate, Reshape, Dense
from tensorflow.keras.models import Model, Sequential

import tensorflow as tf
from tensorflow.keras.layers import Lambda, Layer, ReLU, SpatialDropout2D, Input, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical, plot_model

# tensorflow_addons removed for compatibility
# import tensorflow_addons as tfa
# from tensorflow_addons.layers.adaptive_pooling import AdaptiveAveragePooling2D
# import tensorflow_addons.utils.keras_utils as conv_utils

import tensorflow.keras.backend as K
from tensorflow.keras.layers import concatenate, Reshape

# Create SRM filters if not available
def create_srm_filters():
    """Create 30 SRM filters for preprocessing"""
    # This is a simplified version - you may need to replace with actual SRM filters
    srm_weights = np.random.randn(5, 5, 1, 30) * 0.1
    np.save('SRM.npy', srm_weights)
    return srm_weights

# Load or create SRM filters
try:
    srm_weights = np.load('SRM.npy')
    print(f"Loaded SRM filters with shape: {srm_weights.shape}")
except FileNotFoundError:
    print("SRM.npy not found. Creating random SRM filters...")
    srm_weights = create_srm_filters()
    print(f"Created SRM filters with shape: {srm_weights.shape}")

biasSRM = np.ones(30)

# Activation function
T3 = 3
def Tanh3(x):
    tanh3 = K.tanh(x) * T3
    return tanh3

def SpatialStegoDectect(img_size=256, num_inputs=1):
    """
    CNN Architecture for Spatial Steganography Detection
    """
    tf.keras.backend.clear_session()

    # Inputs
    inputs = []
    for i in range(num_inputs):
        inputs.append(Input(shape=(img_size, img_size, 1), name=f"input_{i+1}"))

    # Reshape inputs to have the same spatial dimensions
    reshapes = []
    for i in range(num_inputs):
        pool_size = img_size // (2 ** i)
        reshape = AveragePooling2D(pool_size=(pool_size, pool_size))(inputs[i])
        reshapes.append(reshape)

    # Block 1
    layers = tf.keras.layers.Conv2D(30, (5, 5), strides=(1, 1), padding='valid')(inputs[0])
    layers = BatchNormalization()(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)

    # Block 2
    layers = tf.keras.layers.DepthwiseConv2D((1, 1), strides=(1, 1), padding='valid')(layers)
    layers = BatchNormalization()(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.SeparableConv2D(30, (3, 3), strides=(1, 1), padding='valid')(layers)
    layers = BatchNormalization()(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)

    # Block 3
    layers = tf.keras.layers.DepthwiseConv2D((1, 1), strides=(1, 1), padding='valid')(layers)
    layers = BatchNormalization()(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.SeparableConv2D(30, (3, 3), strides=(1, 1), padding='valid')(layers)
    layers = BatchNormalization()(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)

    # Block 4
    layers = tf.keras.layers.Conv2D(60, (3, 3), strides=(1, 1), padding='valid')(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(layers)

    # Block 5
    layers = tf.keras.layers.Conv2D(60, (3, 3), strides=(1, 1), padding='valid')(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(layers)

    # Block 6
    layers = tf.keras.layers.DepthwiseConv2D((1, 1), strides=(1, 1), padding='same')(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.SeparableConv2D(384, (3, 3), strides=(1, 1), padding='same')(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)

    # Block 7
    layers = tf.keras.layers.DepthwiseConv2D((1, 1), strides=(1, 1), padding='same')(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)
    layers = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)

    # Block 8
    layers = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)

    # Block 9
    layers = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(layers)
    layers = BatchNormalization()(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)

    # Block 10
    layers = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(layers)
    layers = BatchNormalization()(layers)

    # Block 11
    layers = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)

    # Block 12
    layers = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(layers)
    layers = BatchNormalization()(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)

    # Block 13 (Multi-scale pooling)
    pool_1 = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(layers)
    pool_2 = tf.keras.layers.AveragePooling2D((4, 4), strides=(4, 4))(layers)
    pool_3 = tf.keras.layers.AveragePooling2D((8, 8), strides=(8, 8))(layers)
    pool_4 = tf.keras.layers.AveragePooling2D((16, 16), strides=(16, 16))(layers)

    # Adjust the shapes of the pooling layers
    pool_1 = tf.keras.layers.Conv2D(256, (1, 1), padding='same')(pool_1)
    pool_2 = tf.keras.layers.Conv2D(256, (1, 1), padding='same')(pool_2)
    pool_3 = tf.keras.layers.Conv2D(256, (1, 1), padding='same')(pool_3)
    pool_4 = tf.keras.layers.Conv2D(256, (1, 1), padding='same')(pool_4)

    # Global Average Pooling to flatten the feature maps
    layers = tf.keras.layers.GlobalAveragePooling2D()(layers)
    
    # Add multi-scale features
    pool_1_flat = tf.keras.layers.GlobalAveragePooling2D()(pool_1)
    pool_2_flat = tf.keras.layers.GlobalAveragePooling2D()(pool_2)
    pool_3_flat = tf.keras.layers.GlobalAveragePooling2D()(pool_3)
    pool_4_flat = tf.keras.layers.GlobalAveragePooling2D()(pool_4)
    
    # Concatenate all features
    layers = concatenate([layers, pool_1_flat, pool_2_flat, pool_3_flat, pool_4_flat])

    # Block 14
    layers = tf.keras.layers.Dense(4096)(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)

    # Block 15
    layers = tf.keras.layers.Dense(4096)(layers)
    layers = tf.keras.layers.LeakyReLU()(layers)

    # Block 16
    layers = tf.keras.layers.Dense(2, activation='softmax', name="output_1")(layers)

    # Model generation
    model = Model(inputs=inputs, outputs=layers)

    # Optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.95)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print("SpatialStegoDetect model generated")

    return model

def train(model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size, epochs, initial_epoch=0, threshold=0, model_name=""):
    """Training function for the model"""
    start_time = tm.time()
    
    # Create logs directory
    log_dir = f"logs/{model_name}_{time()}"
    os.makedirs(log_dir, exist_ok=True)
    
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir)
    filepath = f"{log_dir}/saved-model-{{epoch:02d}}-{{val_accuracy:.2f}}.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=False, mode='max')
    
    # model.reset_states()  # Not needed for Functional models
    history = model.fit(X_train, y_train, epochs=epochs, 
                       callbacks=[tensorboard, checkpoint], 
                       batch_size=batch_size, validation_data=(X_valid, y_valid), initial_epoch=initial_epoch)
    
    metrics = model.evaluate(X_test, y_test, verbose=0)
    
    # Create results directory
    results_dir = f"data/{model_name}/"
    os.makedirs(results_dir, exist_ok=True)
      
    # Plot training results
    with plt.style.context('seaborn-v0_8-white'):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Accuracy Vs Epochs')
        plt.ylabel('Accuracy (in %)')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.grid('on')
        plt.savefig(f'{results_dir}Accuracy_SpatialStegoDectect_{model_name}.png', format='png', dpi=300)
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss Vs Epochs')
        plt.ylabel('Loss (in %)')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.grid('on')
        plt.savefig(f'{results_dir}Loss_SpatialStegoDectect_{model_name}.png', format='png', dpi=300)
        plt.show()

    TIME = tm.time() - start_time
    print(f"Time {model_name} = {TIME:.2f} [seconds]")
    return {k: v for k, v in zip(model.metrics_names, metrics)}, history

def Final_Results_Test(model, Trained_Models, X_test, y_test):
    """Test function to find the best model"""
    B_accuracy = 0  # B --> Best
    for filename in os.listdir(Trained_Models):
        if filename != ('train') and filename != ('validation') and filename.endswith('.keras'):
            print(filename)
            model.load_weights(os.path.join(Trained_Models, filename))
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f'Loss={loss:.4f} and Accuracy={accuracy:0.4f}\n') 
            if accuracy > B_accuracy:
                B_accuracy = accuracy
                B_loss = loss
                B_name = filename
    
    print("\n\nBest Model:")
    print(B_name)
    print(f'Loss={B_loss:.4f} and Accuracy={B_accuracy:0.4f}\n')
    return B_name, B_accuracy, B_loss

def create_sample_data(img_size=256, num_samples=1000):
    """Create sample data for testing if real data is not available"""
    print("Creating sample data for testing...")
    
    # Create random image data
    X_train = np.random.randn(num_samples, img_size, img_size, 1).astype(np.float32)
    X_valid = np.random.randn(num_samples//5, img_size, img_size, 1).astype(np.float32)
    X_test = np.random.randn(num_samples//5, img_size, img_size, 1).astype(np.float32)
    
    # Create binary labels (cover vs stego)
    y_train = np.random.randint(0, 2, (num_samples, 2)).astype(np.float32)
    y_valid = np.random.randint(0, 2, (num_samples//5, 2)).astype(np.float32)
    y_test = np.random.randint(0, 2, (num_samples//5, 2)).astype(np.float32)
    
    # Ensure one-hot encoding
    for i in range(len(y_train)):
        if y_train[i][0] == 1:
            y_train[i][1] = 0
        else:
            y_train[i][0] = 0
            y_train[i][1] = 1
    
    for i in range(len(y_valid)):
        if y_valid[i][0] == 1:
            y_valid[i][1] = 0
        else:
            y_valid[i][0] = 0
            y_valid[i][1] = 1
            
    for i in range(len(y_test)):
        if y_test[i][0] == 1:
            y_test[i][1] = 0
        else:
            y_test[i][0] = 0
            y_test[i][1] = 1
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def main():
    """Main function to run the training"""
    print("Starting SpatialStegoDetect Training...")
    
    # Try to load real data, otherwise create sample data
    try:
        print("Loading training data...")
        X_train = np.load('X_training.npy')
        y_train = np.load('y_training.npy')
        X_valid = np.load('X_validating.npy')
        y_valid = np.load('y_validating.npy')
        X_test = np.load('X_testing.npy')
        y_test = np.load('y_testing.npy')
        
        print(f"Data loaded successfully:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_valid shape: {X_valid.shape}")
        print(f"y_valid shape: {y_valid.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        
    except FileNotFoundError:
        print("Data files not found. Creating sample data for testing...")
        X_train, y_train, X_valid, y_valid, X_test, y_test = create_sample_data()
        
        print(f"Sample data created:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_valid shape: {X_valid.shape}")
        print(f"y_valid shape: {y_valid.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
    
    # Model parameters
    EPOCHS = 10
    base_name = "WOW"
    m_name = "SpatialStegoDectect"
    
    # Create and train model
    print("\nCreating model...")
    model = SpatialStegoDectect()
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Training
    print(f"\nStarting training for {EPOCHS} epochs...")
    name = f"Model_{m_name}_{base_name}"
    results, history = train(model, X_train, y_train, X_valid, y_valid, X_test, y_test, 
                           batch_size=32, epochs=EPOCHS, model_name=name)
    
    print(f"\nTraining completed!")
    print(f"Final test results: {results}")
    
    # Save final model
    final_model_path = f"models/{name}_final.keras"
    os.makedirs("models", exist_ok=True)
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main() 