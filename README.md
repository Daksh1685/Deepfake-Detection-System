# Deepfake Detection Using Deep Learning

A lightweight deep learning project for detecting manipulated forensic face images using modern CNN architectures.

## Features

* Detects real vs fake forensic face images
* Built with PyTorch and transfer learning
* Supports MobileNetV3, EfficientNet-B0, and ResNet-50
* High accuracy with fast inference
* Easy training and testing pipeline

## Model Performance

| Model             | Accuracy | Size   |
| ----------------- | -------- | ------ |
| MobileNetV3       | 99.91%   | 39 MB  |
| EfficientNet-B0  | 99.96%   | 52 MB  |
| ResNet-50         | 99.95%   | 283 MB |

## Dataset Structure

```bash
dataset/
├── real/
└── fake/
```

## Installation

```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

## Train Model

```bash
python main.py --config efficientnet --mode train
```

## Inference

```python
from src.inference import DeepfakeInference

model = DeepfakeInference(
    model_path='outputs/models/efficientnet_best_model.pth',
    backbone='efficientnet'
)

result = model.predict_single('image.jpg')
print(result)
```

## Tech Stack

* Python
* PyTorch
* TorchVision
* OpenCV
* NumPy

## Applications

* Digital forensics
* Identity verification
* Fake profile detection
* Cybercrime investigation
* Face tampering analysis

---


