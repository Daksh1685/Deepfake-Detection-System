"""Inference script for deepfake detection."""

from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

import torch
import torch.nn as nn

from model import DeepfakeModel
from utils.transforms import get_val_transforms


class DeepfakeInference:
    """Inference engine for deepfake detection."""
    
    def __init__(self, 
                 model_path: str = 'outputs/models/best_model.pth',
                 device: str = 'cuda'):
        """
        Initialize the inference engine.
        
        Args:
            model_path (str): Path to the trained model checkpoint
            device (str): Device to run inference on ('cuda' or 'cpu')
        
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        self.device = device
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        # Load model
        self.model = self._load_model()
        self.model.to(device)
        self.model.eval()
        
        # Load transforms
        self.transform = get_val_transforms(image_size=224)
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Device: {device}\n")
    
    def _load_model(self) -> nn.Module:
        """
        Load trained model from checkpoint.
        
        Returns:
            nn.Module: Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model (infer architecture)
        state_dict = (checkpoint['model_state_dict'] 
                     if isinstance(checkpoint, dict) else checkpoint)
        
        model = self._create_model_from_checkpoint(state_dict)
        
        # Load weights
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
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
            return DeepfakeModel(backbone='mobilenetv3', pretrained=False)
        elif 'backbone.layer4' in str(state_dict.keys()):
            return DeepfakeModel(backbone='resnet50', pretrained=False)
        else:
            return DeepfakeModel(backbone='mobilenetv3', pretrained=False)
    
    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load and validate image.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            Image.Image: Loaded image in RGB format
        
        Raises:
            FileNotFoundError: If image doesn't exist
            ValueError: If image cannot be loaded
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
    
    @torch.no_grad()
    def predict(self, image_path: str) -> Dict[str, any]:
        """
        Predict if an image is a deepfake.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            dict: Prediction results with label, confidence, and probability
        """
        # Load image
        image = self._load_image(image_path)
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        output = self.model(image_tensor)
        probability = output.item()
        
        # Determine label
        label = 'Fake' if probability > 0.5 else 'Real'
        confidence = probability if probability > 0.5 else (1 - probability)
        
        return {
            'image_path': str(image_path),
            'label': label,
            'confidence': float(confidence),
            'probability': float(probability),
            'is_fake': probability > 0.5,
            'original_image': image
        }
    
    @torch.no_grad()
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict multiple images.
        
        Args:
            image_paths (List[str]): List of image paths
        
        Returns:
            List[Dict]: List of prediction results
        """
        results = []
        
        for image_path in tqdm(image_paths, desc='Predicting', total=len(image_paths)):
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        return results
    
    def visualize_prediction(self, 
                            image_path: str, 
                            save_path: Optional[str] = None) -> None:
        """
        Visualize image with prediction.
        
        Args:
            image_path (str): Path to the image file
            save_path (str, optional): Path to save the figure
        """
        # Get prediction
        result = self.predict(image_path)
        
        # Prepare visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Display image
        ax.imshow(result['original_image'])
        ax.axis('off')
        
        # Determine color based on prediction
        label = result['label']
        confidence = result['confidence']
        color = '#FF6B6B' if label == 'Fake' else '#4ECDC4'  # Red for fake, teal for real
        
        # Add title with prediction
        title = f"{label.upper()}\nConfidence: {confidence:.2%}"
        ax.set_title(title, fontsize=16, fontweight='bold', color=color, pad=20)
        
        # Add prediction box
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        textstr = f"Probability: {result['probability']:.4f}\n(Real: {1-result['probability']:.4f}, Fake: {result['probability']:.4f})"
        plt.text(0.5, -0.1, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='center', bbox=bbox_props)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Prediction image saved to: {save_path}")
        
        plt.show()
    
    def visualize_batch(self, 
                       image_paths: List[str], 
                       max_display: int = 9,
                       save_path: Optional[str] = None) -> None:
        """
        Visualize multiple predictions in a grid.
        
        Args:
            image_paths (List[str]): List of image paths
            max_display (int): Maximum number of images to display (default: 9)
            save_path (str, optional): Path to save the figure
        """
        # Get predictions
        results = self.predict_batch(image_paths[:max_display])
        
        # Create grid
        n_images = len(results)
        n_cols = 3
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        
        if n_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each result
        for idx, result in enumerate(results):
            ax = axes[idx]
            
            # Display image
            ax.imshow(result['original_image'])
            ax.axis('off')
            
            # Color based on prediction
            label = result['label']
            confidence = result['confidence']
            color = '#FF6B6B' if label == 'Fake' else '#4ECDC4'
            
            # Title
            title = f"{label.upper()}\n{confidence:.2%}"
            ax.set_title(title, fontsize=12, fontweight='bold', color=color)
        
        # Hide extra axes
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Grid of predictions saved to: {save_path}")
        
        plt.show()
    
    def print_prediction(self, image_path: str) -> None:
        """
        Print prediction in a formatted way.
        
        Args:
            image_path (str): Path to the image file
        """
        result = self.predict(image_path)
        
        print(f"\n{'='*60}")
        print(f"DEEPFAKE DETECTION PREDICTION")
        print(f"{'='*60}")
        print(f"Image: {result['image_path']}")
        print(f"Prediction: {result['label'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"  - Real:    {1 - result['probability']:.4f}")
        print(f"  - Fake:    {result['probability']:.4f}")
        print(f"{'='*60}\n")
        
        # Interpretation
        if result['label'] == 'Real':
            interpretation = "Image appears to be AUTHENTIC (Real)"
        else:
            interpretation = "Image appears to be MANIPULATED (Fake)"
        
        print(f"Interpretation: {interpretation}")
        print()


def predict_single_image(image_path: str,
                        model_path: str = 'outputs/models/best_model.pth',
                        device: str = 'cuda',
                        visualize: bool = True) -> Dict:
    """
    Predict a single image using the trained model.
    
    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the trained model
        device (str): Device to run inference on
        visualize (bool): Whether to visualize the prediction
    
    Returns:
        dict: Prediction results
    
    Example:
        >>> result = predict_single_image('path/to/image.jpg')
        >>> print(result['label'], result['confidence'])
    """
    # Create inference engine
    inference = DeepfakeInference(model_path=model_path, device=device)
    
    # Get prediction
    result = inference.predict(image_path)
    
    # Print result
    inference.print_prediction(image_path)
    
    # Visualize
    if visualize:
        inference.visualize_prediction(image_path)
    
    return result


def predict_batch_images(image_paths: List[str],
                        model_path: str = 'outputs/models/best_model.pth',
                        device: str = 'cuda',
                        visualize: bool = True) -> List[Dict]:
    """
    Predict multiple images using the trained model.
    
    Args:
        image_paths (List[str]): List of image paths
        model_path (str): Path to the trained model
        device (str): Device to run inference on
        visualize (bool): Whether to visualize predictions
    
    Returns:
        List[Dict]: List of prediction results
    
    Example:
        >>> results = predict_batch_images(['img1.jpg', 'img2.jpg'])
        >>> for r in results:
        ...     print(r['label'], r['confidence'])
    """
    # Create inference engine
    inference = DeepfakeInference(model_path=model_path, device=device)
    
    # Get predictions
    results = inference.predict_batch(image_paths)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {len(results)}")
    
    real_count = sum(1 for r in results if r['label'] == 'Real')
    fake_count = sum(1 for r in results if r['label'] == 'Fake')
    
    print(f"Real images: {real_count}")
    print(f"Fake images: {fake_count}")
    
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"Average confidence: {avg_confidence:.2%}")
    print(f"{'='*60}\n")
    
    # Visualize
    if visualize:
        inference.visualize_batch(image_paths)
    
    return results


if __name__ == '__main__':
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Example usage - single image prediction
    example_image = 'dataset/real/sample.jpg'  # Update with actual image path
    
    print("Deepfake Detection Inference Engine")
    print("="*60)
    print("\nTo use this script:")
    print("1. Single image: predict_single_image('path/to/image.jpg')")
    print("2. Batch: predict_batch_images(['img1.jpg', 'img2.jpg'])")
    print("3. Custom: engine = DeepfakeInference()")
    print("           result = engine.predict('path/to/image.jpg')")
    print("           engine.visualize_prediction('path/to/image.jpg')")
    print("="*60)
