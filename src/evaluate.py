"""Evaluation script for deepfake detection model."""

import json
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    auc as sklearn_auc
)

# Import custom modules
from .model import DeepfakeModel
from .dataset import get_single_dataloader


class DeepfakeEvaluator:
    """Evaluator class for deepfake detection models."""
    
    def __init__(self, 
                 model_path: str,
                 test_loader: DataLoader,
                 device: str = 'cuda',
                 output_dir: str = 'outputs'):
        """
        Initialize the evaluator.
        
        Args:
            model_path (str): Path to the trained model checkpoint
            test_loader (DataLoader): Test data loader
            device (str): Device to evaluate on ('cuda' or 'cpu')
            output_dir (str): Directory to save outputs
        """
        self.device = device
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(device)
        self.model.eval()
        
        # Metrics storage
        self.predictions = None
        self.probabilities = None
        self.ground_truth = None
        self.metrics = {}
    
    def _load_model(self, model_path: str) -> nn.Module:
        """
        Load trained model from checkpoint.
        
        Args:
            model_path (str): Path to the model checkpoint
        
        Returns:
            nn.Module: Loaded model
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Try to infer model architecture from state dict
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint
        
        # Create model and load weights
        # Default to mobilenetv3, but can be inferred from state dict
        model = self._create_model_from_checkpoint(state_dict)
        
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Best accuracy: {checkpoint.get('best_accuracy', 'unknown'):.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ Model weights loaded")
        
        return model
    
    def _create_model_from_checkpoint(self, state_dict: dict) -> nn.Module:
        """
        Create model architecture from state dict.
        
        Args:
            state_dict (dict): Model state dictionary
        
        Returns:
            nn.Module: Model with appropriate architecture
        """
        # Infer architecture from state dict keys
        if 'backbone.features' in str(state_dict.keys()):
            # Likely MobileNetV3 or EfficientNet
            return DeepfakeModel(backbone='mobilenetv3', pretrained=False)
        elif 'backbone.layer4' in str(state_dict.keys()):
            # Likely ResNet
            return DeepfakeModel(backbone='resnet50', pretrained=False)
        else:
            # Default to MobileNetV3
            return DeepfakeModel(backbone='mobilenetv3', pretrained=False)
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run inference on test set and compute metrics.
        
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        print(f"\n{'='*70}")
        print(f"Evaluating on Test Set")
        print(f"{'='*70}\n")
        
        all_predictions = []
        all_probabilities = []
        all_ground_truth = []
        
        progress_bar = tqdm(self.test_loader, desc='Evaluating', total=len(self.test_loader))
        
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Get predictions
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).int().cpu().numpy()
            
            all_predictions.extend(preds.flatten())
            all_probabilities.extend(probs.flatten())
            all_ground_truth.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        self.predictions = np.array(all_predictions)
        self.probabilities = np.array(all_probabilities)
        self.ground_truth = np.array(all_ground_truth)
        
        # Compute metrics
        self._compute_metrics()
        
        return self.metrics
    
    def _compute_metrics(self) -> None:
        """Compute evaluation metrics."""
        # Basic metrics
        accuracy = accuracy_score(self.ground_truth, self.predictions)
        precision = precision_score(self.ground_truth, self.predictions, zero_division=0)
        recall = recall_score(self.ground_truth, self.predictions, zero_division=0)
        f1 = f1_score(self.ground_truth, self.predictions, zero_division=0)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(self.ground_truth, self.probabilities)
        except:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(self.ground_truth, self.predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics from confusion matrix
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = recall  # Same as recall
        
        self.metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'confusion_matrix': cm.tolist()
        }
    
    def print_metrics(self) -> None:
        """Print evaluation metrics."""
        print(f"\n{'='*70}")
        print(f"EVALUATION METRICS")
        print(f"{'='*70}")
        print(f"Accuracy:    {self.metrics['accuracy']:.4f}")
        print(f"Precision:   {self.metrics['precision']:.4f}")
        print(f"Recall:      {self.metrics['recall']:.4f}")
        print(f"Sensitivity: {self.metrics['sensitivity']:.4f}")
        print(f"Specificity: {self.metrics['specificity']:.4f}")
        print(f"F1 Score:    {self.metrics['f1']:.4f}")
        print(f"ROC AUC:     {self.metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {self.metrics['true_negatives']}")
        print(f"  False Positives: {self.metrics['false_positives']}")
        print(f"  False Negatives: {self.metrics['false_negatives']}")
        print(f"  True Positives:  {self.metrics['true_positives']}")
        print(f"{'='*70}\n")
    
    def print_classification_report(self) -> str:
        """
        Print classification report.
        
        Returns:
            str: Classification report
        """
        report = classification_report(
            self.ground_truth,
            self.predictions,
            target_names=['Real', 'Fake'],
            digits=4
        )
        
        print("\nCLASSIFICATION REPORT")
        print("="*70)
        print(report)
        print("="*70)
        
        return report
    
    def plot_confusion_matrix(self, save_path: str = None) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            save_path (str): Path to save the figure
        """
        if save_path is None:
            save_path = self.log_dir / 'confusion_matrix.png'
        else:
            save_path = Path(save_path)
        
        cm = confusion_matrix(self.ground_truth, self.predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'],
                    yticklabels=['Real', 'Fake'],
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
        plt.close()
    
    def plot_roc_curve(self, save_path: str = None) -> None:
        """
        Plot and save ROC curve.
        
        Args:
            save_path (str): Path to save the figure
        """
        if save_path is None:
            save_path = self.log_dir / 'roc_curve.png'
        else:
            save_path = Path(save_path)
        
        try:
            fpr, tpr, _ = roc_curve(self.ground_truth, self.probabilities)
            roc_auc = sklearn_auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve saved to: {save_path}")
            plt.close()
        except Exception as e:
            print(f"✗ Failed to plot ROC curve: {e}")
    
    def plot_precision_recall_curve(self, save_path: str = None) -> None:
        """
        Plot and save Precision-Recall curve.
        
        Args:
            save_path (str): Path to save the figure
        """
        if save_path is None:
            save_path = self.log_dir / 'precision_recall_curve.png'
        else:
            save_path = Path(save_path)
        
        from sklearn.metrics import precision_recall_curve, auc as sklearn_auc
        
        try:
            precision, recall, _ = precision_recall_curve(self.ground_truth, self.probabilities)
            pr_auc = sklearn_auc(recall, precision)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='darkorange', lw=2,
                    label=f'Precision-Recall curve (AUC = {pr_auc:.4f})')
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            plt.legend(loc="best", fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Precision-Recall curve saved to: {save_path}")
            plt.close()
        except Exception as e:
            print(f"✗ Failed to plot Precision-Recall curve: {e}")
    
    def save_metrics_json(self, save_path: str = None) -> None:
        """
        Save metrics to JSON file.
        
        Args:
            save_path (str): Path to save the JSON file
        """
        if save_path is None:
            save_path = self.log_dir / 'evaluation_metrics.json'
        else:
            save_path = Path(save_path)
        
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        print(f"✓ Metrics saved to: {save_path}")
    
    def generate_report(self) -> None:
        """Generate complete evaluation report."""
        # Run evaluation
        self.evaluate()
        
        # Print metrics
        self.print_metrics()
        
        # Print classification report
        self.print_classification_report()
        
        # Generate plots
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        
        # Save metrics
        self.save_metrics_json()
        
        print(f"\n{'='*70}")
        print("Evaluation Complete! All outputs saved to:")
        print(f"  - {self.log_dir}/confusion_matrix.png")
        print(f"  - {self.log_dir}/roc_curve.png")
        print(f"  - {self.log_dir}/precision_recall_curve.png")
        print(f"  - {self.log_dir}/evaluation_metrics.json")
        print(f"{'='*70}\n")


def evaluate_model(model_path: str = 'outputs/models/best_model.pth',
                   dataset_dir: str = 'dataset',
                   batch_size: int = 32,
                   device: str = 'cuda',
                   output_dir: str = 'outputs') -> Dict[str, float]:
    """
    Evaluate a trained deepfake detection model.
    
    Args:
        model_path (str): Path to the trained model checkpoint
        dataset_dir (str): Path to the dataset directory
        batch_size (int): Batch size for evaluation
        device (str): Device to evaluate on
        output_dir (str): Directory to save outputs
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Load test dataset
    print(f"Loading test dataset from: {dataset_dir}")
    test_loader = get_single_dataloader(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        is_train=False
    )
    
    print(f"Test samples: {len(test_loader.dataset)}\n")
    
    # Create evaluator
    evaluator = DeepfakeEvaluator(
        model_path=model_path,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir
    )
    
    # Generate report
    evaluator.generate_report()
    
    return evaluator.metrics


if __name__ == '__main__':
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Evaluate model
    metrics = evaluate_model(
        model_path='outputs/models/best_model.pth',
        dataset_dir='dataset',
        batch_size=32,
        device=device,
        output_dir='outputs'
    )
