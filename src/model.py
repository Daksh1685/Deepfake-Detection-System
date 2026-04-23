"""Model architecture for deepfake detection."""

from typing import Literal
import torch
import torch.nn as nn
from torchvision import models


class DeepfakeModel(nn.Module):
    """
    Modular deepfake detection model supporting multiple backbone architectures.
    
    Supports:
    - MobileNetV3-Large
    - EfficientNet-B0
    - ResNet-50
    
    The model uses the pretrained backbone for feature extraction and
    a custom classifier head for binary classification.
    """
    
    def __init__(self, backbone: Literal['mobilenetv3', 'efficientnet', 'resnet50'] = 'mobilenetv3', 
                 pretrained: bool = True, dropout: float = 0.3):
        """
        Initialize the DeepfakeModel.
        
        Args:
            backbone (str): Backbone architecture to use.
                - 'mobilenetv3': MobileNetV3-Large (recommended for efficiency)
                - 'efficientnet': EfficientNet-B0
                - 'resnet50': ResNet-50
            pretrained (bool): Whether to load pretrained weights (default: True)
            dropout (float): Dropout rate for the classifier head (default: 0.3)
        
        Raises:
            ValueError: If an unsupported backbone is specified
        """
        super(DeepfakeModel, self).__init__()
        
        if backbone.lower() not in ['mobilenetv3', 'efficientnet', 'resnet50']:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                "Choose from: 'mobilenetv3', 'efficientnet', 'resnet50'"
            )
        
        self.backbone_name = backbone.lower()
        self.pretrained = pretrained
        
        # Load backbone and get feature dimension
        self.backbone, feature_dim = self._load_backbone()
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def _load_backbone(self) -> tuple:
        """
        Load the backbone architecture.
        
        Returns:
            tuple: (backbone model, output feature dimension)
        """
        if self.backbone_name == 'mobilenetv3':
            return self._load_mobilenetv3()
        elif self.backbone_name == 'efficientnet':
            return self._load_efficientnet()
        elif self.backbone_name == 'resnet50':
            return self._load_resnet50()
    
    def _load_mobilenetv3(self) -> tuple:
        """
        Load MobileNetV3-Large backbone.
        
        MobileNetV3-Large: Efficient model with ~5.4M parameters
        Output feature dimension: 960
        
        Returns:
            tuple: (backbone, feature_dim)
        """
        model = models.mobilenet_v3_large(pretrained=self.pretrained)
        
        # Remove the classification layer
        model.classifier = nn.Identity()
        
        # MobileNetV3-Large outputs 960 features
        feature_dim = 960
        
        return model, feature_dim
    
    def _load_efficientnet(self) -> tuple:
        """
        Load EfficientNet-B0 backbone.
        
        EfficientNet-B0: Balanced efficiency and accuracy
        Output feature dimension: 1280
        
        Returns:
            tuple: (backbone, feature_dim)
        """
        model = models.efficientnet_b0(pretrained=self.pretrained)
        
        # Remove the classification layer
        model.classifier = nn.Identity()
        
        # EfficientNet-B0 outputs 1280 features
        feature_dim = 1280
        
        return model, feature_dim
    
    def _load_resnet50(self) -> tuple:
        """
        Load ResNet-50 backbone.
        
        ResNet-50: More computationally intensive but potentially higher accuracy
        Output feature dimension: 2048
        
        Returns:
            tuple: (backbone, feature_dim)
        """
        model = models.resnet50(pretrained=self.pretrained)
        
        # Remove the fully connected layer
        model.fc = nn.Identity()
        
        # ResNet-50 outputs 2048 features
        feature_dim = 2048
        
        return model, feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Binary classification probability (0-1) of shape (batch_size, 1)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Flatten if needed (for some BackBones)
        if features.dim() > 2:
            features = features.mean(dim=[2, 3])  # Global average pooling
        
        # Pass through classifier
        output = self.classifier(features)
        
        return output
    
    def get_backbone_info(self) -> dict:
        """
        Get information about the current backbone.
        
        Returns:
            dict: Information about the backbone architecture
        """
        info = {
            'backbone': self.backbone_name,
            'pretrained': self.pretrained,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        return info
    
    def freeze_backbone(self) -> None:
        """Freeze backbone weights for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone weights for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_backbone_except_last_layer(self) -> None:
        """
        Freeze all backbone layers except the last one.
        Useful for fine-tuning strategy.
        """
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the last block (if applicable)
        if self.backbone_name == 'mobilenetv3':
            # Unfreeze last block in MobileNetV3
            for param in self.backbone.features[-1].parameters():
                param.requires_grad = True
        elif self.backbone_name == 'efficientnet':
            # Unfreeze last block in EfficientNet
            for param in self.backbone.features[-1].parameters():
                param.requires_grad = True
        elif self.backbone_name == 'resnet50':
            # Unfreeze layer4 (last residual block) in ResNet
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True


def create_model(backbone: str = 'mobilenetv3', pretrained: bool = True, 
                 dropout: float = 0.3) -> DeepfakeModel:
    """
    Factory function to create a deepfake detection model.
    
    Args:
        backbone (str): Backbone architecture ('mobilenetv3', 'efficientnet', 'resnet50')
        pretrained (bool): Whether to use pretrained weights
        dropout (float): Dropout rate for classifier
    
    Returns:
        DeepfakeModel: Initialized model
    
    Example:
        >>> model = create_model(backbone='mobilenetv3', pretrained=True)
        >>> print(model.get_backbone_info())
    """
    return DeepfakeModel(backbone=backbone, pretrained=pretrained, dropout=dropout)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Deepfake Detection Model - Testing Different Backbones")
    print("=" * 60)
    
    for backbone_name in ['mobilenetv3', 'efficientnet', 'resnet50']:
        print(f"\n{backbone_name.upper()} Architecture:")
        print("-" * 60)
        
        model = create_model(backbone=backbone_name, pretrained=True)
        info = model.get_backbone_info()
        
        print(f"Backbone: {info['backbone']}")
        print(f"Pretrained: {info['pretrained']}")
        print(f"Total Parameters: {info['total_params']:,}")
        print(f"Trainable Parameters: {info['trainable_params']:,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print(f"Output Shape: {output.shape}")
        print(f"Output (probability): {output.item():.4f}")
    
    print("\n" + "=" * 60)
