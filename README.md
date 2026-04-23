"""
DEEPFAKE DETECTION PROJECT - COMPLETE SETUP GUIDE

## Project Structure

```
deepfake-detection/
├── dataset/                          # Training dataset (organized images)
│   ├── real/                         # Real face images (label: 0)
│   └── fake/                         # Deepfake images (label: 1)
├── src/                              # Source code modules
│   ├── __init__.py
│   ├── dataset.py                    # PyTorch Dataset class
│   ├── model.py                      # MobileNetV3/EfficientNet/ResNet50 models
│   ├── train.py                      # Training loop with validation
│   ├── evaluate.py                   # Evaluation metrics & visualization
│   └── inference.py                  # Prediction on new images
├── utils/                            # Utility modules
│   ├── __init__.py
│   ├── transforms.py                 # Image augmentation & normalization
│   └── metrics.py                    # Plotting utilities
├── configs/                          # Configuration files
│   └── config.py                     # Configurable training parameters
├── outputs/                          # Training outputs
│   ├── models/
│   │   └── best_model.pth           # Best trained model
│   └── logs/
│       ├── training_history.json     # Training metrics
│       ├── confusion_matrix.png      # Test confusion matrix
│       ├── roc_curve.png             # ROC curve
│       └── precision_recall_curve.png # PR curve
├── main.py                           # Main training orchestrator
├── organize_dataset.py               # Data organization script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Workflow Overview

```
📊 DATASET (140k real and fake face images)
        ↓
🔄 dataset.py (DeepfakeDataset class)
   ├─ Load images from real/ and fake/ folders
   ├─ Assign labels (real=0, fake=1)
   └─ Train/Val/Test split (70/15/15)
        ↓
🎨 transforms.py (Augmentation & Normalization)
   ├─ Train: Flip, Rotation, ColorJitter, GaussianBlur
   ├─ Val/Test: Resize only
   └─ ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ↓
🧠 model.py (DeepfakeModel with modular backbones)
   ├─ MobileNetV3-Large (recommended: fast, 3.2M params)
   ├─ EfficientNet-B0 (balanced: 4.3M params)
   └─ ResNet-50 (high accuracy: 24M params)
        ↓
🔧 train.py (DeepfakeTrainer)
   ├─ Forward pass
   ├─ BCE Loss calculation
   ├─ AdamW Optimizer (lr=1e-4, weight_decay=1e-5)
   ├─ Learning rate scheduling (ReduceLROnPlateau)
   ├─ Validation loop
   ├─ Early stopping (patience=10)
   └─ Best model saving
        ↓
📈 evaluate.py (DeepfakeEvaluator)
   ├─ Accuracy, Precision, Recall, F1, ROC-AUC
   ├─ Confusion matrix heatmap
   ├─ Classification report
   ├─ ROC curve
   └─ Precision-Recall curve
        ↓
🎯 inference.py (DeepfakeInference)
   ├─ Single image prediction
   ├─ Batch prediction
   ├─ Confidence scores
   └─ Visualization with matplotlib
```

## Installation

### 1. Create Python virtual environment (optional but recommended)

```bash
python -m venv venv
venv\\Scripts\\activate  # On Windows
# or
source venv/bin/activate  # On Linux/Mac
```

### 2. Install dependencies

```bash
pip install numpy pandas pillow torch torchvision tqdm scikit-learn matplotlib seaborn
```

## Quick Start

### 1. Organize Dataset

The images need to be organized from the raw folder structure to training structure:

```bash
# Copy images from real_vs_fake to dataset/real and dataset/fake
python organize_dataset.py --source ../real_vs_fake --target dataset
```

This creates:
- `dataset/real/` - All real face images
- `dataset/fake/` - All deepfake images

### 2. Train Model

#### Option A: Simple Training (Recommended)

```bash
# Train with MobileNetV3 (default, fastest)
python main.py --config mobilenetv3 --mode train
```

#### Option B: Advanced Training

```bash
# Different architectures:
python main.py --config mobilenetv3 --mode train  # Fast, lightweight
python main.py --config efficientnet --mode train # Balanced
python main.py --config resnet50 --mode train     # High accuracy

# Quick experiments:
python main.py --config lightweight --mode train  # 10 epochs, quick test

# Maximum accuracy:
python main.py --config high_accuracy --mode train  # 100 epochs, slower
```

### 3. Run Inference

```bash
# Single image prediction
from src.inference import predict_single_image
result = predict_single_image('path/to/image.jpg')
# Output:
# Prediction: Real / Fake
# Confidence: 92.34%

# Batch prediction
from src.inference import predict_batch_images
results = predict_batch_images(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# Custom inference
from src.inference import DeepfakeInference
engine = DeepfakeInference('outputs/models/best_model.pth')
result = engine.predict('image.jpg')
engine.visualize_prediction('image.jpg')  # Display result
```

## Configuration Guide

### Model Configurations

| Config | Architecture | Speed | Accuracy | Parameters | Best For |
|--------|-------------|-------|----------|-----------|----------|
| mobilenetv3 | MobileNetV3-Large | ⚡⚡⚡ Fast | Good | 3.2M | Real-time inference |
| efficientnet | EfficientNet-B0 | ⚡⚡ Medium | Very Good | 4.3M | Balanced |
| resnet50 | ResNet-50 | ⚡ Slow | Excellent | 24M | Maximum accuracy |
| lightweight | MobileNetV3 | ⚡⚡⚡ Very Fast | Good | 3.2M | Quick experiments |
| high_accuracy | ResNet-50 | ⚡ Slow | Excellent | 24M | Fine-tuned accuracy |

### Custom Configuration

Edit `configs/config.py` to create custom configurations:

```python
from configs.config import Config, DataConfig, ModelConfig, TrainingConfig

config = Config(
    data=DataConfig(
        dataset_path='dataset',
        batch_size=32,
        image_size=224,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    ),
    model=ModelConfig(
        model_name='mobilenetv3',  # or 'efficientnet', 'resnet50'
        pretrained=True,
        dropout=0.3
    ),
    training=TrainingConfig(
        num_epochs=50,
        learning_rate=1e-4,
        weight_decay=1e-5,
        optimizer='adamw',
        use_scheduler=True,
        use_early_stopping=True,
        early_stopping_patience=10
    )
)
```

## Training Details

### Loss Function
- **Binary Cross Entropy (BCE)**: Standard loss for binary classification
- Optimizers: AdamW (recommended)
- Learning rate: 1e-4 (configurable)
- Learning rate scheduling: ReduceLROnPlateau

### Data Augmentation (Training Only)
- Horizontal flip (50%)
- Rotation (±10°)
- Color jitter (brightness, contrast, saturation, hue)
- Gaussian blur (random kernel and sigma)

### Validation & Early Stopping
- Validates after each epoch
- Early stopping if no improvement for N epochs
- Saves best model based on validation accuracy

## Evaluation Metrics

The evaluation provides:

1. **Classification Metrics**
   - Accuracy: Overall correctness
   - Precision: True positive rate among predicted positives
   - Recall: True positive rate among actual positives
   - F1 Score: Harmonic mean of precision and recall
   - ROC AUC: Area under ROC curve

2. **Visualizations**
   - Confusion Matrix (saved as PNG)
   - ROC Curve
   - Precision-Recall Curve

3. **Output Files**
   - `outputs/logs/confusion_matrix.png`
   - `outputs/logs/roc_curve.png`
   - `outputs/logs/precision_recall_curve.png`
   - `outputs/logs/evaluation_metrics.json`

## Inference Usage

### Single Image

```python
from src.inference import predict_single_image

result = predict_single_image(
    image_path='path/to/image.jpg',
    model_path='outputs/models/best_model.pth',
    visualize=True
)

print(result)
# Output:
# {
#     'image_path': 'path/to/image.jpg',
#     'label': 'Real',
#     'confidence': 0.9234,
#     'probability': 0.0766,
#     'is_fake': False
# }
```

### Batch Processing

```python
from src.inference import predict_batch_images

results = predict_batch_images(
    image_paths=['img1.jpg', 'img2.jpg', 'img3.jpg'],
    visualize=True
)

# Print summary
for result in results:
    print(f"{result['image_path']}: {result['label']} ({result['confidence']:.2%})")
```

### Custom Inference Engine

```python
from src.inference import DeepfakeInference

# Initialize
engine = DeepfakeInference(
    model_path='outputs/models/best_model.pth',
    device='cuda'
)

# Single prediction
result = engine.predict('image.jpg')

# Visualize
engine.visualize_prediction('image.jpg', save_path='output.png')

# Batch
results = engine.predict_batch(['img1.jpg', 'img2.jpg'])
engine.visualize_batch(['img1.jpg', 'img2.jpg'])

# Print
engine.print_prediction('image.jpg')
```

## Performance Benchmarks

### MobileNetV3 (Recommended)
- Training time: ~2-4 hours (50 epochs, 32 batch size, single GPU)
- Model size: ~13 MB
- Inference time: ~50ms per image
- Expected accuracy: 90-95% on test set

### EfficientNet-B0
- Training time: ~3-5 hours
- Model size: ~29 MB
- Inference time: ~70ms per image
- Expected accuracy: 92-96%

### ResNet-50
- Training time: ~6-8 hours
- Model size: ~102 MB
- Inference time: ~100ms per image
- Expected accuracy: 94-98%

## Project Workflow Summary

```
1. DATASET PREPARATION
   └─ organize_dataset.py: Move images to dataset/real and dataset/fake

2. TRAINING
   └─ main.py: Orchestrates entire training pipeline
      ├─ Loads config
      ├─ Creates dataloaders
      ├─ Initializes model
      ├─ Trains with validation
      └─ Saves best model to outputs/models/best_model.pth

3. EVALUATION
   └─ evaluate.py: DeepfakeEvaluator
      ├─ Loads trained model
      ├─ Computes metrics
      ├─ Generates confusion matrix
      ├─ Plots ROC curve
      └─ Saves visualizations

4. INFERENCE
   └─ inference.py: DeepfakeInference
      ├─ Loads trained model
      ├─ Makes predictions
      ├─ Outputs confidence scores
      └─ Visualizes results
```

## Future Enhancements

- [ ] Multi-model ensemble for improved accuracy
- [ ] TensorBoard integration for training visualization
- [ ] Model quantization for mobile deployment
- [ ] Real-time webcam inference
- [ ] API server for inference
- [ ] Data augmentation with CLAHE
- [ ] Focal loss for handling class imbalance
- [ ] Knowledge distillation for model compression

## References

- MobileNetV3: https://arxiv.org/abs/1905.02175
- EfficientNet: https://arxiv.org/abs/1905.11946
- ResNet: https://arxiv.org/abs/1512.03385
- Binary Classification: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

## Support

For issues or questions:
1. Review sample outputs in outputs/logs/
2. Check training history in outputs/logs/training_history.json
3. Check the [DETAILED_WORKFLOW_ALL_MODELS.md](DETAILED_WORKFLOW_ALL_MODELS.md) for complete guidance

---

Last Updated: April 5, 2026
Project: Deepfake Detection with PyTorch
Dataset: 140k real and fake face images
"""

if __name__ == '__main__':
    print(__doc__)
