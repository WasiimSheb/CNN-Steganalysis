# CNN Steganography Detection

This project implements a Convolutional Neural Network (CNN) architecture for spatial steganography detection, based on the SpatialStegoDetect model.

## Features

- Deep CNN architecture with 16 blocks for steganography detection
- Support for BOSSbase dataset and custom datasets
- Multi-scale pooling for enhanced feature extraction
- Batch normalization and LeakyReLU activations
- Comprehensive training and evaluation pipeline
- Sample data generation for testing

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- See `requirements.txt` for complete list

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Use with your own dataset

1. Organize your data into two folders:
   - `cover_images/` - containing cover (original) images
   - `stego_images/` - containing steganographic images

2. Run the data loader to prepare your dataset:
```bash
python data_loader.py
```

3. Train the model:
```bash
python spatial_stego_detect.py
```

### Option 2: Test with sample data

If you don't have a dataset ready, the system will automatically create sample data for testing:

```bash
python spatial_stego_detect.py
```

## File Structure

```
CNN-Steganalysis/
├── spatial_stego_detect.py    # Main CNN architecture and training
├── data_loader.py             # Data preparation utilities
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── SRM.npy                   # SRM filters (auto-generated if missing)
├── data/                     # Training results and plots
├── logs/                     # TensorBoard logs and model checkpoints
└── models/                   # Saved final models
```

## Model Architecture

The SpatialStegoDetect model consists of:

1. **Preprocessing**: 30 SRM (Spatial Rich Model) filters
2. **Feature Extraction**: 13 convolutional blocks with:
   - Standard convolutions
   - Depthwise separable convolutions
   - Batch normalization
   - LeakyReLU activations
3. **Multi-scale Pooling**: Captures features at different scales
4. **Classification**: Two fully connected layers + softmax output

## Training Parameters

- **Optimizer**: SGD with learning rate 0.005 and momentum 0.95
- **Loss Function**: Binary crossentropy
- **Batch Size**: 32 (adjustable)
- **Epochs**: 10 (adjustable)
- **Image Size**: 256x256 pixels

## Dataset Format

The model expects:
- **Input**: Grayscale images of size 256x256
- **Labels**: One-hot encoded [cover, stego]
- **File Format**: NumPy arrays (.npy files)

Expected files:
- `X_training.npy` - Training images
- `y_training.npy` - Training labels
- `X_validating.npy` - Validation images
- `y_validating.npy` - Validation labels
- `X_testing.npy` - Test images
- `y_testing.npy` - Test labels

## Usage Examples

### Basic Training
```python
from spatial_stego_detect import SpatialStegoDectect, train

# Create model
model = SpatialStegoDectect(img_size=256)

# Train model (assuming data is loaded)
results, history = train(model, X_train, y_train, X_valid, y_valid, X_test, y_test, 
                        batch_size=32, epochs=10, model_name="MyModel")
```

### Custom Data Loading
```python
from data_loader import prepare_steganography_dataset

# Prepare dataset from folders
X_train, y_train, X_val, y_val, X_test, y_test = prepare_steganography_dataset(
    cover_folder="path/to/cover/images",
    stego_folder="path/to/stego/images",
    img_size=256
)
```

## Output

The training process generates:
- **Model checkpoints**: Saved in `logs/` directory
- **Training plots**: Accuracy and loss curves in `data/` directory
- **TensorBoard logs**: For monitoring training progress
- **Final model**: Saved in `models/` directory

## Monitoring Training

Use TensorBoard to monitor training progress:
```bash
tensorboard --logdir=logs/
```

## Performance

The model is designed to detect steganographic content in images with high accuracy. Performance depends on:
- Quality and size of training dataset
- Steganographic algorithm used
- Image characteristics

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or image size
2. **Missing SRM.npy**: Will be auto-generated with random filters
3. **No dataset**: Sample data will be created automatically
4. **TensorFlow warnings**: Usually safe to ignore

### GPU Support

To use GPU acceleration:
```python
# Check GPU availability
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

## Customization

### Modify Architecture
Edit the `SpatialStegoDectect()` function in `spatial_stego_detect.py`

### Change Training Parameters
Modify parameters in the `main()` function:
- `EPOCHS`: Number of training epochs
- `batch_size`: Training batch size
- `img_size`: Input image dimensions

### Add New Datasets
Extend `data_loader.py` to support additional dataset formats

## References

This implementation is based on research in steganography detection using deep learning. The architecture incorporates:
- SRM filters for preprocessing
- Multi-scale feature extraction
- Advanced CNN architectures for binary classification

## License

This project is for educational and research purposes. 