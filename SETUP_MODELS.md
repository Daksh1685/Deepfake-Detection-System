# 📥 Model Download Guide

This document explains how to download and setup pre-trained models for the deepfake detection project.

## Quick Start (Recommended)

### Option 1: Hugging Face (Best for Private Models)

1. **Create/get your Hugging Face token**:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token (select "read" access)
   - Copy your token

2. **Download models**:
```bash
# Download all models
python scripts/download_models.py --source huggingface --token YOUR_HF_TOKEN

# Download single model
python scripts/download_models.py --source huggingface --token YOUR_HF_TOKEN --model efficientnet
```

### Option 2: Google Drive (Easiest)

```bash
# Download all models
python scripts/download_models.py --source gdrive

# Download single model
python scripts/download_models.py --source gdrive --model efficientnet
```

---

## Setup Instructions

### 1. Install Optional Dependencies

For downloading from Hugging Face:
```bash
pip install huggingface-hub
```

For downloading from Google Drive:
```bash
pip install gdown
```

Or install both:
```bash
pip install huggingface-hub gdown
```

### 2. Setup Model Hosting (For Repository Owners)

#### **A. Hugging Face Setup** (Recommended)

1. Create a Hugging Face account: https://huggingface.co
2. Create a **private** repository for models:
   - Name: `deepfake-detection-models`
   - Type: Private
   - Model type: PyTorch
3. Upload your trained models:
   ```bash
   git clone https://huggingface.co/yourusername/deepfake-detection-models
   cd deepfake-detection-models
   
   # Copy your model files
   cp /path/to/efficientnet_best_model.pth .
   cp /path/to/mobilenetv3_best_model.pth .
   cp /path/to/resnet50_best_model.pth .
   
   git add .
   git commit -m "Add pre-trained models"
   git push
   ```

4. Share access:
   - Go to repository settings
   - Add collaborators or use tokens for automation
   - Users can then download with their HF token

#### **B. Google Drive Setup** (Easiest Alternative)

1. Create a Google Drive folder for models
2. Upload your `.pth` files
3. Get file IDs:
   - Right-click file → Share
   - Set to "Anyone with the link can view" OR require authentication
   - Extract file ID from sharing link: `https://drive.google.com/file/d/FILEID/view`
4. Update `scripts/download_models.py` with your file IDs

---

## Model Information

| Model | Size | Parameters | Accuracy | Download Time (1Mbps) |
|-------|------|-----------|----------|----------------------|
| MobileNetV3 | 39 MB | 3.2M | 99.91% | ~5 min |
| EfficientNet-B0 | 52 MB | 4.3M | 99.96% | ~7 min |
| ResNet-50 | 283 MB | 23.5M | 99.95% | ~38 min |

## Usage After Download

Once models are downloaded to `outputs/models/`:

```python
from src.inference import DeepfakeInference

# Automatically uses downloaded models
inference = DeepfakeInference(
    model_path='outputs/models/efficientnet_best_model.pth',
    backbone='efficientnet'
)

result = inference.predict_single('image.jpg')
print(f"Is Fake: {result['is_fake']}, Confidence: {result['confidence']:.4f}")
```

## Troubleshooting

### Models not downloading?

1. **Check internet connection**:
```bash
ping google.com
```

2. **Verify source availability**:
```bash
# For Hugging Face
huggingface-cli repo-info yourusername/deepfake-detection-models

# For Google Drive (test one file)
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=YOUR_FILE_ID', 'test.txt')"
```

3. **Check permissions**:
   - Hugging Face: Verify your token is valid
   - Google Drive: Ensure sharing settings allow access

4. **Disk space**:
   - All 3 models need ~375 MB
   - Ensure you have enough free space

### Authentication fails?

For Hugging Face:
```bash
# Re-authenticate
huggingface-cli login
# Then run without --token flag
python scripts/download_models.py --source huggingface
```

For Google Drive:
- Google Drive doesn't require authentication for public files
- If private, use Google Drive web UI instead

---

## Manual Download

If automated download fails:

1. **Get files from Hugging Face**:
   - Visit: https://huggingface.co/yourusername/deepfake-detection-models
   - Download `.pth` files manually
   - Place in `outputs/models/`

2. **Get files from Google Drive**:
   - Visit shared folder link
   - Download files manually
   - Place in `outputs/models/`

3. **Verify download**:
```bash
ls -lh outputs/models/
# Should show all three .pth files with correct sizes
```

---

## For Developers: Adding New Models

To add a new pre-trained model:

1. Train your model using `main.py`
2. Save model checkpoint to `outputs/models/`
3. Upload to your hosting service (HF or Google Drive)
4. Update `scripts/download_models.py`:

```python
MODELS = {
    'your_model_name': {
        'hf': 'yourusername/deepfake-models/your_model_name.pth',
        'gdrive': 'GOOGLE_DRIVE_ID',
        'size': 'XXmb'
    }
}
```

---

## Privacy & Security Notes

✅ **Hugging Face**:
- Private repositories supported
- Token-based access control
- No data leakage

✅ **Google Drive**:
- Share with specific users
- Revoke access anytime
- Track who accessed models

❌ **What NOT to do**:
- Don't commit `.pth` files to main repo
- Don't make private models publicly accessible without consent
- Don't share tokens in code or issues

---

## Alternative: Train Your Own Models

If you prefer not to download pre-trained models:

```bash
# Organize your dataset
python organize_dataset.py --source raw_data --dest dataset

# Train from scratch (takes 2-4 hours on GPU)
python main.py --config efficientnet --mode train

# Models will be saved to outputs/models/
```

---

**Questions?** See [COMPLETE_COMMANDS_REFERENCE.md](../COMPLETE_COMMANDS_REFERENCE.md) or [PROFESSIONAL_PROJECT_REPORT.md](../PROFESSIONAL_PROJECT_REPORT.md)
