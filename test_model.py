#!/usr/bin/env python3
"""
Quick test script to verify the CNN model works
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Reduce TensorFlow warnings

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

try:
    from spatial_stego_detect import SpatialStegoDectect, create_sample_data
    print("✓ Successfully imported spatial_stego_detect")
    
    # Test model creation
    print("\nCreating model...")
    model = SpatialStegoDectect(img_size=256)
    print("✓ Model created successfully!")
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Test with small sample data
    print("\nTesting with sample data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = create_sample_data(
        img_size=256, num_samples=100
    )
    
    print(f"✓ Sample data created:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_valid.shape}")
    print(f"  Testing: {X_test.shape}")
    
    # Test model prediction
    print("\nTesting model prediction...")
    sample_pred = model.predict(X_test[:5], verbose=0)
    print(f"✓ Prediction successful! Shape: {sample_pred.shape}")
    print(f"  Sample predictions: {sample_pred[:2]}")
    
    print("\n🎉 All tests passed! Your model is ready to use.")
    print("\nNext steps:")
    print("1. Prepare your dataset using data_loader.py")
    print("2. Run spatial_stego_detect.py to train the model")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 