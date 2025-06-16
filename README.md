# CNN Steganography Detection

A Convolutional Neural Network implementation for spatial steganography detection using the SpatialStegoDetect architecture.

## Architecture

- 16-block CNN with SRM preprocessing filters
- Multi-scale pooling and feature extraction
- Binary classification (cover vs steganographic images)
- Input: 256x256 grayscale images

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python spatial_stego_detect.py
```

### With BOSSbase Dataset
```bash
python load_boss_dataset.py  # Prepare dataset
python spatial_stego_detect_light.py  # Train lightweight model
```

## Files

- `spatial_stego_detect.py` - Original CNN architecture (25.6M parameters)
- `spatial_stego_detect_light.py` - Lightweight version (1.7M parameters)  
- `stego_detect_optimized.py` - Optimized with residual blocks
- `load_boss_dataset.py` - BOSSbase dataset loader
- `data_loader.py` - General dataset utilities

## Model Variants

| Model | Parameters | Description |
|-------|------------|-------------|
| Original | 25.6M | Full SpatialStegoDetect architecture |
| Lightweight | 1.7M | Reduced complexity for faster training |
| Optimized | Variable | Residual blocks with skip connections |

## Dataset Format

Expected NumPy arrays:
- `X_training.npy`, `y_training.npy` - Training data
- `X_validating.npy`, `y_validating.npy` - Validation data  
- `X_testing.npy`, `y_testing.npy` - Test data

Labels: One-hot encoded [cover, stego]

## Training Configuration

- Optimizer: SGD (lr=0.005, momentum=0.95)
- Loss: Binary crossentropy
- Batch size: 32-64
- Callbacks: TensorBoard, ModelCheckpoint, ReduceLROnPlateau

## Output

- Model checkpoints: `logs/`
- Training plots: `data/`
- Final models: `models/`

## Performance

Tested on BOSSbase dataset with WOW steganographic algorithm. Results depend on dataset quality and steganographic method sophistication. 