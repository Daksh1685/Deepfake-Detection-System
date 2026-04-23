"""Training script for deepfake detection model."""

import os
import json
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import custom modules
from .model import DeepfakeModel, create_model
from .dataset import create_dataloaders, get_single_dataloader
from utils.metrics import plot_confusion_matrix, plot_roc_curve


class DeepfakeTrainer:
    """Trainer class for deepfake detection models."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader, 
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 output_dir: str = 'outputs'):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): The model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            device (str): Device to train on ('cuda' or 'cpu')
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): Weight decay for AdamW
            output_dir (str): Directory to save outputs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.model_dir = self.output_dir / 'models'
        self.log_dir = self.output_dir / 'logs'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler for learning rate decay
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.best_model_path = self.model_dir / 'best_model.pth'
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for images, labels in progress_bar:
            # Move data to device
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            dict: Validation metrics (loss, accuracy, precision, recall, f1)
        """
        self.model.eval()
        total_loss = 0.0
        predictions = []
        ground_truth = []
        probabilities = []
        
        progress_bar = tqdm(self.val_loader, desc='Validating', leave=False)
        
        with torch.no_grad():
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Store predictions
                preds = (outputs > 0.5).int()
                predictions.extend(preds.cpu().numpy())
                ground_truth.extend(labels.cpu().numpy())
                probabilities.extend(outputs.cpu().numpy())
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        predictions = np.array(predictions).flatten()
        ground_truth = np.array(ground_truth).flatten()
        probabilities = np.array(probabilities).flatten()
        
        accuracy = np.mean(predictions == ground_truth)
        
        # Calculate precision, recall, f1
        tp = np.sum((predictions == 1) & (ground_truth == 1))
        fp = np.sum((predictions == 1) & (ground_truth == 0))
        fn = np.sum((predictions == 0) & (ground_truth == 1))
        tn = np.sum((predictions == 0) & (ground_truth == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_loss = total_loss / len(self.val_loader)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'probabilities': probabilities
        }
        
        return metrics
    
    def train(self, num_epochs: int = 50, early_stopping_patience: int = 10) -> Dict:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs (int): Number of epochs to train
            early_stopping_patience (int): Patience for early stopping
        
        Returns:
            dict: Training history and best metrics
        """
        print(f"\n{'='*70}")
        print(f"Training Deepfake Detection Model - {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"{'='*70}\n")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Print metrics
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy:  {val_metrics['accuracy']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall:    {val_metrics['recall']:.4f}")
            print(f"  Val F1:        {val_metrics['f1']:.4f}")
            print(f"  LR: {current_lr:.2e}\n")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                self.save_model(is_best=True)
                print(f"  ✓ Best model saved! (Accuracy: {self.best_val_accuracy:.4f})\n")
            else:
                patience_counter += 1
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['accuracy'])
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best Validation Accuracy: {self.best_val_accuracy:.4f}")
        print(f"Model saved to: {self.best_model_path}")
        print(f"{'='*70}\n")
        
        return {
            'best_epoch': self.best_epoch,
            'best_accuracy': self.best_val_accuracy,
            'history': self.history,
            'val_metrics': val_metrics
        }
    
    def save_model(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_accuracy': self.best_val_accuracy
        }
        
        if is_best:
            torch.save(checkpoint, self.best_model_path)
    
    def save_history(self) -> None:
        """Save training history to JSON."""
        history_file = self.log_dir / 'training_history.json'
        
        # Convert numpy types to Python native types for JSON serialization
        json_history = {}
        for key, values in self.history.items():
            json_history[key] = [float(v) for v in values]
        
        with open(history_file, 'w') as f:
            json.dump(json_history, f, indent=4)
        
        print(f"Training history saved to: {history_file}")
    
    def load_best_model(self) -> None:
        """Load the best model checkpoint."""
        if self.best_model_path.exists():
            checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from: {self.best_model_path}")
        else:
            print(f"Best model not found at: {self.best_model_path}")


def main(dataset_dir: str = 'dataset',
         backbone: str = 'mobilenetv3',
         batch_size: int = 32,
         num_epochs: int = 50,
         learning_rate: float = 1e-4,
         weight_decay: float = 1e-5,
         device: str = 'cuda',
         train_split: float = 0.7,
         val_split: float = 0.15,
         test_split: float = 0.15,
         seed: int = 42) -> None:
    """
    Main training function.
    
    Args:
        dataset_dir (str): Path to dataset directory
        backbone (str): Model backbone ('mobilenetv3', 'efficientnet', 'resnet50')
        batch_size (int): Batch size for dataloaders
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for optimizer
        device (str): Device to train on
        train_split (float): Training data proportion
        val_split (float): Validation data proportion
        test_split (float): Test data proportion
        seed (int): Random seed for reproducibility
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    
    # Create dataloaders
    print(f"Loading datasets from: {dataset_dir}")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=4,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=seed
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")
    
    # Create model
    print(f"Creating {backbone} model...")
    model = create_model(backbone=backbone, pretrained=True, dropout=0.3)
    info = model.get_backbone_info()
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}\n")
    
    # Create trainer
    trainer = DeepfakeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        output_dir='outputs'
    )
    
    # Train
    results = trainer.train(num_epochs=num_epochs, early_stopping_patience=10)
    
    # Save history
    trainer.save_history()
    
    # Load best model
    trainer.load_best_model()


if __name__ == '__main__':
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Train with default parameters
    main(
        dataset_dir='dataset',
        backbone='mobilenetv3',
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-4,
        weight_decay=1e-5,
        device=device,
        seed=42
    )
