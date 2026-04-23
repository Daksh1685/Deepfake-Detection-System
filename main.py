"""
Main execution script for deepfake detection project.

Workflow:
    dataset (140k images)
        ↓
    dataset.py (load images, labels)
        ↓
    transforms (augmentation, normalization)
        ↓
    model.py (MobileNetV3/EfficientNet/ResNet50)
        ↓
    train.py (training loop with validation)
        ↓
    evaluate.py (metrics, confusion matrix, ROC curve)
        ↓
    inference.py (make predictions on new images)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
from configs.config import get_config
from src.dataset import create_dataloaders
from src.model import create_model
from src.train import DeepfakeTrainer
from src.evaluate import DeepfakeEvaluator
from src.inference import DeepfakeInference


def print_workflow():
    """Print the deepfake detection workflow."""
    print("\n" + "="*80)
    print(" "*20 + "DEEPFAKE DETECTION WORKFLOW")
    print("="*80)
    
    workflow = """
    [DATA] DATASET (140k real and fake face images)
            ↓
    [LOAD] dataset.py (DeepfakeDataset class)
            │
            ├─ Load images from real/ and fake/ folders
            ├─ Assign labels (real=0, fake=1)
            └─ Train/Val/Test split (70/15/15)
            ↓
    🎨 transforms (Augmentation & Normalization)
            │
            ├─ Train: Flip, Rotation, ColorJitter, GaussianBlur
            ├─ Val/Test: Resize only
            └─ ImageNet normalization
            ↓
    🧠 model.py (DeepfakeModel with modular backbones)
            │
            ├─ MobileNetV3-Large (recommended, fast)
            ├─ EfficientNet-B0 (balanced)
            └─ ResNet-50 (high accuracy, slower)
            ↓
    🔧 train.py (DeepfakeTrainer)
            │
            ├─ Forward pass
            ├─ BCE Loss calculation
            ├─ AdamW Optimizer
            ├─ Learning rate scheduling
            ├─ Validation loop
            ├─ Early stopping
            └─ Best model saving
            ↓
    [EVAL] evaluate.py (DeepfakeEvaluator)
            │
            ├─ Accuracy, Precision, Recall, F1, ROC-AUC
            ├─ Confusion matrix
            ├─ Classification report
            ├─ ROC curve
            └─ Precision-Recall curve
            ↓
    [INFER] inference.py (DeepfakeInference)
            │
            ├─ Single image prediction
            ├─ Batch prediction
            ├─ Confidence score
            └─ Visualization with matplotlib
    """
    
    print(workflow)
    print("="*80 + "\n")


def train_deepfake_model(config_name: str = 'mobilenetv3'):
    """
    Complete training pipeline.
    
    Args:
        config_name (str): Configuration to use
            - 'mobilenetv3': Fast, efficient (recommended)
            - 'efficientnet': Balanced performance
            - 'resnet50': High accuracy
            - 'lightweight': Quick experiments
            - 'high_accuracy': Maximum accuracy
    """
    # Print workflow
    print_workflow()
    
    # =========================================================================
    # 1. LOAD CONFIGURATION
    # =========================================================================
    print("Step 1: Loading Configuration")
    print("-" * 80)
    config = get_config(config_name)
    config.print_config()
    
    # Set random seeds for reproducibility
    np.random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    if config.training.device == 'cuda':
        torch.cuda.manual_seed(config.data.seed)
    
    # =========================================================================
    # 2. LOAD DATASET
    # =========================================================================
    print("Step 2: Loading Dataset")
    print("-" * 80)
    print(f"Loading from: {config.data.dataset_path}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir=config.data.dataset_path,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        seed=config.data.seed
    )
    
    print(f"[OK] Train samples: {len(train_loader.dataset):,}")
    print(f"[OK] Val samples: {len(val_loader.dataset):,}")
    print(f"[OK] Test samples: {len(test_loader.dataset):,}")
    print()
    
    # =========================================================================
    # 3. CREATE MODEL
    # =========================================================================
    print("Step 3: Creating Model")
    print("-" * 80)
    print(f"Architecture: {config.model.model_name}")
    
    model = create_model(
        backbone=config.model.model_name,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout
    )
    
    info = model.get_backbone_info()
    print(f"[OK] Backbone: {info['backbone']}")
    print(f"[OK] Total parameters: {info['total_params']:,}")
    print(f"[OK] Trainable parameters: {info['trainable_params']:,}")
    print()
    
    # =========================================================================
    # 4. TRAIN MODEL
    # =========================================================================
    print("Step 4: Training Model")
    print("-" * 80)
    
    trainer = DeepfakeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.training.device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        output_dir=config.output.output_dir
    )
    
    results = trainer.train(
        num_epochs=config.training.num_epochs,
        early_stopping_patience=config.training.early_stopping_patience
    )
    
    # Save training history
    trainer.save_history()
    print()
    
    # =========================================================================
    # 5. EVALUATE MODEL
    # =========================================================================
    print("Step 5: Evaluating Model")
    print("-" * 80)
    
    model_path = Path(config.output.models_dir) / 'best_model.pth'
    
    evaluator = DeepfakeEvaluator(
        model_path=str(model_path),
        test_loader=test_loader,
        device=config.training.device,
        output_dir=config.output.output_dir
    )
    
    evaluator.generate_report()
    
    # =========================================================================
    # 6. SAVE FINAL SUMMARY
    # =========================================================================
    print("Step 6: Saving Summary")
    print("-" * 80)
    
    summary = {
        'config': config_name,
        'model': config.model.model_name,
        'best_epoch': results['best_epoch'],
        'best_accuracy': results['best_accuracy'],
        'final_metrics': evaluator.metrics
    }
    
    print(f"[OK] Configuration: {config_name}")
    print(f"[OK] Model: {config.model.model_name}")
    print(f"[OK] Best Epoch: {results['best_epoch']}")
    print(f"[OK] Best Validation Accuracy: {results['best_accuracy']:.4f}")
    print(f"[OK] Test Accuracy: {evaluator.metrics['accuracy']:.4f}")
    print()
    
    # =========================================================================
    # 7. TRAINING COMPLETE
    # =========================================================================
    print("="*80)
    print(" "*25 + "[DONE] TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📁 Output files saved to:")
    print(f"   Models: {config.output.models_dir}")
    print(f"   Logs:   {config.output.logs_dir}")
    print(f"\n[RESULTS] Key Results:")
    print(f"   Best Validation Accuracy: {results['best_accuracy']:.4f}")
    print(f"   Test Set Accuracy:        {evaluator.metrics['accuracy']:.4f}")
    print(f"   Test Set Precision:       {evaluator.metrics['precision']:.4f}")
    print(f"   Test Set Recall:          {evaluator.metrics['recall']:.4f}")
    print(f"   Test Set F1 Score:        {evaluator.metrics['f1']:.4f}")
    print(f"   Test Set ROC-AUC:         {evaluator.metrics['roc_auc']:.4f}")
    print("\n💡 Next steps:")
    print("   1. Review confusion matrix: outputs/logs/confusion_matrix.png")
    print("   2. Check ROC curve: outputs/logs/roc_curve.png")
    print("   3. Use inference.py to predict on new images")
    print("="*80 + "\n")
    
    return results, evaluator.metrics


def inference_example(model_path: str = 'outputs/models/best_model.pth',
                     image_path: str = None):
    """
    Run inference on example images.
    
    Args:
        model_path (str): Path to trained model
        image_path (str): Path to image file (optional)
    """
    print("="*80)
    print(" "*20 + "INFERENCE - DEEPFAKE DETECTION")
    print("="*80 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Create inference engine
        print("Loading inference engine...")
        inference_engine = DeepfakeInference(
            model_path=model_path,
            device=device
        )
        
        print("\n[OK] Inference engine ready!")
        print("\nUsage examples:")
        print("-" * 80)
        print("\n1. Single image prediction:")
        print("   from src.inference import predict_single_image")
        print("   result = predict_single_image('path/to/image.jpg')")
        print()
        print("2. Batch prediction:")
        print("   from src.inference import predict_batch_images")
        print("   results = predict_batch_images(['img1.jpg', 'img2.jpg'])")
        print()
        print("3. Custom inference:")
        print("   engine = DeepfakeInference('outputs/models/best_model.pth')")
        print("   result = engine.predict('image.jpg')")
        print("   engine.visualize_prediction('image.jpg')")
        print("-" * 80 + "\n")
        
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("Please train the model first using: python main.py train\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Deepfake Detection Project')
    parser.add_argument('--config', type=str, default='mobilenetv3',
                       choices=['mobilenetv3', 'efficientnet', 'resnet50', 'lightweight', 'high_accuracy'],
                       help='Configuration to use (default: mobilenetv3)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                       help='Mode to run (default: train)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"\n[START] Starting training with {args.config} configuration...\n")
        train_deepfake_model(config_name=args.config)
    else:
        inference_example()
