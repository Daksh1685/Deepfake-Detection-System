# Complete Commands Reference Guide - Deepfake Detection Project

## Overview
This document contains all commands needed to run every script in the deepfake detection project, organized by purpose and execution order.

---

## 📋 Table of Contents
1. [Initial Setup](#initial-setup)
2. [Data Organization](#data-organization)
3. [Training Commands](#training-commands)
4. [Evaluation Commands](#evaluation-commands)
5. [Analysis Commands](#analysis-commands)
6. [Visualization & Comparison Commands](#visualization--comparison-commands)
7. [Inference Commands](#inference-commands)
8. [Utility & Helper Commands](#utility--helper-commands)

---

## Initial Setup

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install All Dependencies

```bash
# Install from requirements.txt (recommended)
pip install -r requirements.txt

# OR install individual packages
pip install numpy==1.24.3 pandas==2.0.3 Pillow==10.0.0
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install tqdm==4.66.1 scikit-learn==1.3.0 matplotlib==3.7.2 seaborn==0.12.2
pip install tensorboard==2.13.0 opencv-python==4.8.0.76
pip install jupyter==1.0.0 ipython==8.14.0
```

### 3. Verify Installation

```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check other dependencies
python -c "import cv2, numpy, pandas, sklearn; print('All imports successful!')"
```

---

## Data Organization

### Organize Dataset from real_vs_fake to dataset/

**Purpose:** Consolidates images from the raw folder structure into a unified training structure.

```bash
# Basic usage (default paths)
python organize_dataset.py

# With custom source and target paths
python organize_dataset.py --source ../real_vs_fake --target dataset

# With explicit path specification
python organize_dataset.py --source e:\Deepfake_project\real_vs_fake --target e:\Deepfake_project\deepfake-detection\dataset

# Verbose output
python organize_dataset.py --source ../real_vs_fake --target dataset --verbose
```

**Expected Output:**
```
======================================================================
Dataset Organization
======================================================================
Source: e:\Deepfake_project\real_vs_fake
Target: e:\Deepfake_project\deepfake-detection\dataset

Processing TRAIN split:
  Real images: 35000 ... ✓
  Fake images: 35000 ... ✓
Processing TEST split:
  Real images: 2500 ... ✓
  Fake images: 2500 ... ✓
Processing VALID split:
  Real images: 2500 ... ✓
  Fake images: 2500 ... ✓

Total images organized: 139,998
Structure: dataset/real/ and dataset/fake/
```

---

## Training Commands

### 1. Main Training Script (All Architectures)

**Purpose:** Unified training interface for all models.

```bash
# Train with MobileNetV3 (default, fastest)
python main.py --config mobilenetv3 --mode train

# Train with EfficientNet-B0 (recommended for best accuracy)
python main.py --config efficientnet --mode train

# Train with ResNet-50 (highest capacity, slower)
python main.py --config resnet50 --mode train

# Quick test with lightweight configuration (10 epochs)
python main.py --config lightweight --mode train

# Maximum accuracy configuration (100 epochs)
python main.py --config high_accuracy --mode train

# Train with custom parameters
python main.py --config efficientnet --epochs 50 --batch_size 32 --learning_rate 0.0001

# Train and evaluate in one go
python main.py --config efficientnet --mode train --evaluate
```

---

### 2. Architecture-Specific Training Scripts

#### MobileNetV3 Training

```bash
# Run dedicated MobileNetV3 training script
python run_efficientnet.py

# Alternative: Run training with EfficientNet
python train_efficientnet.py

# With retrain and save both models
python retrain_and_save_both.py
```

#### ResNet-50 Training

```bash
# Run dedicated ResNet-50 training script
python train_resnet50.py

# ResNet-50 with custom configuration
python train_resnet50.py --epochs 50 --batch_size 32 --learning_rate 1e-4
```

#### EfficientNet Training

```bash
# Run dedicated EfficientNet training script
python train_efficientnet.py

# Run EfficientNet simple evaluation
python evaluate_efficientnet_simple.py
```

---

## Evaluation Commands

### 1. Individual Model Evaluation

#### MobileNetV3 Evaluation

```bash
# Evaluate MobileNetV3
python evaluate_mobilenetv3.py

# Detailed evaluation
python evaluate_mobilenetv3.py --model_path outputs/models/mobilenetv3_best_model.pth --save_visualizations
```

#### EfficientNet-B0 Evaluation

```bash
# Evaluate EfficientNet-B0
python evaluate_efficientnet.py

# Simple evaluation
python evaluate_efficientnet_simple.py

# With custom model path
python evaluate_efficientnet.py --model_path outputs/models/efficientnet_best_model.pth
```

#### ResNet-50 Evaluation

```bash
# Evaluate ResNet-50
python evaluate_resnet50.py

# ResNet-50 with detailed metrics
python evaluate_resnet50.py --model_path outputs/models/resnet50_best_model.pth --save_report
```

---

### 2. Comprehensive Comparison Evaluation

```bash
# Compare all three models
python compare_all_models.py

# Compare architectures
python compare_architectures.py

# Advanced architecture comparison
python compare_all_models.py --generate_summary --save_comparison_report
```

---

## Analysis Commands

### 1. Overfitting Analysis

#### MobileNetV3 Overfitting Analysis

```bash
# Analyze overfitting for MobileNetV3
python analyze_overfitting.py

# Detailed overfitting analysis
python analyze_overfitting_detailed.py

# Specific to MobileNetV3
python analyze_mobilenetv3_overfitting.py
```

#### EfficientNet Overfitting Analysis

```bash
# Analyze EfficientNet overfitting
python analyze_efficientnet.py

# Detailed analysis
python analyze_efficientnet_overfitting.py

# Simple analysis
python analyze_efficientnet_overfitting.py --detailed
```

#### ResNet-50 Overfitting Analysis

```bash
# Analyze ResNet-50 overfitting
python analyze_resnet50_overfitting.py

# Comprehensive analysis
python analyze_resnet50_overfitting.py --save_report
```

---

## Visualization & Comparison Commands

### 1. Graph Generation

#### Generate MobileNetV3 Graphs

```bash
# Generate all MobileNetV3 visualizations
python generate_final_graphs.py

# Model-specific graphs
python generate_final_graphs.py --model mobilenetv3
```

#### Generate EfficientNet Graphs

```bash
# Generate EfficientNet visualizations
python generate_efficientnet_graphs.py

# With custom output path
python generate_efficientnet_graphs.py --output outputs/logs/
```

#### Generate ResNet-50 Graphs

```bash
# Generate ResNet-50 visualizations
python generate_resnet50_graphs.py

# Detailed graph generation
python generate_resnet50_graphs.py --detailed --save_all
```

#### Generate Final Comprehensive Graphs

```bash
# Generate all final comparison graphs
python generate_final_graphs.py

# With specific output format
python generate_final_graphs.py --format png --dpi 300
```

---

### 2. Model Comparison Visualizations

```bash
# Generate architecture comparison
python compare_architectures.py

# Generate comprehensive comparison analysis
python compare_all_models.py

# Generate comparison with detailed metrics
python compare_all_models.py --detailed --generate_report
```

---

## Inference Commands

### 1. Single Image Prediction

```bash
# Python script usage
python -c "
from src.inference import DeepfakeInference

inference = DeepfakeInference(
    model_name='efficientnet',
    checkpoint_path='outputs/models/efficientnet_best_model.pth'
)

label, confidence = inference.predict_single('path/to/image.jpg')
print(f'Prediction: {label} (Confidence: {confidence:.4f})')
"
```

### 2. Batch Image Prediction

```bash
# Batch prediction
python -c "
from src.inference import DeepfakeInference

inference = DeepfakeInference(
    model_name='efficientnet',
    checkpoint_path='outputs/models/efficientnet_best_model.pth'
)

images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = inference.predict_batch(images)

for img, (label, conf) in zip(images, results):
    print(f'{img}: {label} (Confidence: {conf:.4f})')
"
```

### 3. Inference with Visualization

```bash
# Single image with visualization
python -c "
from src.inference import DeepfakeInference
import matplotlib.pyplot as plt

inference = DeepfakeInference(
    model_name='efficientnet',
    checkpoint_path='outputs/models/efficientnet_best_model.pth'
)

label, confidence = inference.predict_single_with_viz('image.jpg')
plt.show()
"
```

---

## Utility & Helper Commands

### 1. Model Information

```bash
# Get model architecture information
python -c "
from src.model import create_model
from configs.config import get_config

config = get_config('efficientnet')
model = create_model(config)
model.get_backbone_info()
"

# Count model parameters
python -c "
from src.model import create_model
from configs.config import get_config

config = get_config('efficientnet')
model = create_model(config)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total Parameters: {total_params:,}')
print(f'Trainable Parameters: {trainable_params:,}')
"
```

### 2. Dataset Information

```bash
# Get dataset statistics
python -c "
from src.dataset import create_dataloaders
from configs.config import get_config

config = get_config('efficientnet')
train_loader, val_loader, test_loader = create_dataloaders(
    dataset_path=config.data.dataset_path,
    batch_size=config.data.batch_size
)

print(f'Train samples: {len(train_loader.dataset):,}')
print(f'Val samples: {len(val_loader.dataset):,}')
print(f'Test samples: {len(test_loader.dataset):,}')
"
```

### 3. Check GPU Availability

```bash
# Check CUDA availability
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"
```

### 4. List Available Configurations

```bash
# Show all available configurations
python -c "
from configs.config import get_config

configs = ['mobilenetv3', 'efficientnet', 'resnet50', 'lightweight', 'high_accuracy']
for cfg_name in configs:
    config = get_config(cfg_name)
    print(f'\n{cfg_name.upper()}:')
    print(f'  Model: {config.model.model_name}')
    print(f'  Epochs: {config.training.num_epochs}')
    print(f'  Learning Rate: {config.training.learning_rate}')
    print(f'  Batch Size: {config.data.batch_size}')
"
```

---

## Quick Start Workflow

### Complete Training and Evaluation Pipeline

```bash
# Step 1: Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Step 2: Organize data
python organize_dataset.py --source ../real_vs_fake --target dataset

# Step 3: Train MobileNetV3
python run_efficientnet.py

# Step 4: Evaluate MobileNetV3
python evaluate_mobilenetv3.py

# Step 5: Train EfficientNet-B0
python train_efficientnet.py

# Step 6: Evaluate EfficientNet-B0
python evaluate_efficientnet.py

# Step 7: Train ResNet-50
python train_resnet50.py

# Step 8: Evaluate ResNet-50
python evaluate_resnet50.py

# Step 9: Compare all models
python compare_all_models.py

# Step 10: Generate final visualizations
python generate_final_graphs.py

# Step 11: Analyze overfitting
python analyze_efficientnet_overfitting.py
python analyze_resnet50_overfitting.py
```

---

## Advanced Usage Examples

### Example 1: Train Custom Configuration

```bash
# Python script for custom training
python -c "
from configs.config import Config, DataConfig, ModelConfig, TrainingConfig
from src.dataset import create_dataloaders
from src.model import create_model
from src.train import DeepfakeTrainer

# Custom configuration
data_config = DataConfig(
    dataset_path='dataset',
    batch_size=64,  # Double batch size
    num_workers=4,
    image_size=224,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    seed=42
)

model_config = ModelConfig(
    model_name='efficientnet',
    pretrained=True,
    dropout=0.4  # Increased dropout
)

training_config = TrainingConfig(
    num_epochs=100,
    learning_rate=5e-5,  # Lower learning rate
    weight_decay=1e-4,
    optimizer='adamw',
    use_scheduler=True,
    use_early_stopping=True,
    early_stopping_patience=15
)

# Create config object
config = Config(
    data=data_config,
    model=model_config,
    training=training_config
)

# Load data
dataloaders = create_dataloaders(
    dataset_path=config.data.dataset_path,
    batch_size=config.data.batch_size
)

# Create and train model
model = create_model(config)
trainer = DeepfakeTrainer(config)
trainer.train(dataloaders[0], dataloaders[1])
"
```

### Example 2: Hyperparameter Tuning

```bash
# Grid search for learning rates
python -c "
from configs.config import get_config
from src.dataset import create_dataloaders
from src.model import create_model
from src.train import DeepfakeTrainer

learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
results = {}

for lr in learning_rates:
    config = get_config('efficientnet')
    config.training.learning_rate = lr
    config.training.num_epochs = 20  # Quick test
    
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path=config.data.dataset_path,
        batch_size=config.data.batch_size
    )
    
    model = create_model(config)
    trainer = DeepfakeTrainer(config)
    history = trainer.train(train_loader, val_loader)
    
    results[lr] = history['val_accuracy'][-1]
    print(f'LR {lr}: {results[lr]:.4f}')
"
```

### Example 3: Model Ensembling

```bash
# Load multiple models for ensemble prediction
python -c "
import torch
from src.inference import DeepfakeInference

# Load models
models = {
    'mobilenetv3': DeepfakeInference('mobilenetv3', 'outputs/models/mobilenetv3_best_model.pth'),
    'efficientnet': DeepfakeInference('efficientnet', 'outputs/models/efficientnet_best_model.pth'),
    'resnet50': DeepfakeInference('resnet50', 'outputs/models/resnet50_best_model.pth')
}

# Ensemble prediction
def ensemble_predict(image_path):
    predictions = []
    for model_name, inference in models.items():
        label, confidence = inference.predict_single(image_path)
        predictions.append(confidence)
    
    avg_confidence = sum(predictions) / len(predictions)
    ensemble_label = 'Fake' if avg_confidence > 0.5 else 'Real'
    
    return ensemble_label, avg_confidence

label, confidence = ensemble_predict('image.jpg')
print(f'Ensemble Prediction: {label} (Confidence: {confidence:.4f})')
"
```

---

## Troubleshooting Commands

### Check Environment

```bash
# Verify all dependencies are installed correctly
python -c "
import sys
packages = ['torch', 'torchvision', 'numpy', 'pandas', 'PIL', 'sklearn', 'matplotlib', 'seaborn', 'cv2', 'tqdm']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg} installed')
    except ImportError:
        print(f'✗ {pkg} NOT installed')
"
```

### Clear Cache

```bash
# Remove PyTorch cache
python -c "import torch; torch.hub.set_dir('.cache/torch_hub')"

# Remove model cache
rmdir /s outputs\models  # Windows
rm -rf outputs/models   # Linux/Mac
```

### Reset Training

```bash
# Delete all training outputs
rmdir /s outputs\logs outputs\models  # Windows
rm -rf outputs/logs outputs/models   # Linux/Mac

# Recreate directories
mkdir outputs\logs outputs\models  # Windows
mkdir -p outputs/logs outputs/models  # Linux/Mac
```

---

## Performance Benchmarking Commands

### Measure Inference Speed

```bash
# Time single inference
python -c "
import time
import torch
from src.inference import DeepfakeInference

inference = DeepfakeInference('efficientnet', 'outputs/models/efficientnet_best_model.pth')

# Warmup
_ = inference.predict_single('image.jpg')

# Time inference
start = time.time()
for _ in range(100):
    _ = inference.predict_single('image.jpg')
end = time.time()

avg_time = (end - start) / 100
print(f'Average inference time: {avg_time*1000:.2f} ms')
"
```

### Memory Usage

```bash
# Check model memory footprint
python -c "
import torch
from src.model import create_model
from configs.config import get_config

models = ['mobilenetv3', 'efficientnet', 'resnet50']

for model_name in models:
    config = get_config(model_name)
    model = create_model(config)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    
    # Estimate memory (assuming float32 = 4 bytes per parameter)
    memory_mb = params * 4 / (1024 * 1024)
    
    print(f'{model_name}: {params:,} params ({memory_mb:.2f} MB)')
"
```

---

## Notes

- Always activate the virtual environment before running commands
- Ensure the dataset is organized before training
- GPU significantly speeds up training (8-10x faster)
- Recommended training sequence: MobileNetV3 → EfficientNet → ResNet-50
- All trained models are saved in `outputs/models/`
- All logs and visualizations are saved in `outputs/logs/`
- Use early stopping to prevent overfitting and save time

---

**Last Updated:** April 2026  
**Project Status:** Complete and Production Ready

