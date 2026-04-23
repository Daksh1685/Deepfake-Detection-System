"""Configuration file for deepfake detection project."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import torch


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_path: str = 'dataset'
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    
    # Data split
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Random seed for reproducibility
    seed: int = 42


@dataclass
class ModelConfig:
    """Model configuration."""
    # Architecture: 'mobilenetv3', 'efficientnet', 'resnet50'
    model_name: Literal['mobilenetv3', 'efficientnet', 'resnet50'] = 'mobilenetv3'
    pretrained: bool = True
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimizer: 'adam', 'adamw', 'sgd'
    optimizer: Literal['adam', 'adamw', 'sgd'] = 'adamw'
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-6
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Loss function: 'bce', 'focal', 'weighted_bce'
    loss_function: Literal['bce', 'focal', 'weighted_bce'] = 'bce'
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    # Flip
    use_horizontal_flip: bool = True
    flip_probability: float = 0.5
    
    # Rotation
    use_rotation: bool = True
    rotation_degrees: int = 10
    
    # Color jitter
    use_color_jitter: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    
    # Gaussian blur
    use_gaussian_blur: bool = True
    blur_kernel_size: int = 3
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 2.0
    
    # Affine transformations
    use_affine: bool = False
    affine_translate_x: float = 0.1
    affine_translate_y: float = 0.1
    affine_scale_min: float = 0.9
    affine_scale_max: float = 1.1


@dataclass
class OutputConfig:
    """Output and logging configuration."""
    output_dir: str = 'outputs'
    models_dir: str = 'outputs/models'
    logs_dir: str = 'outputs/logs'
    
    # Checkpoint saving
    save_best_model: bool = True
    save_checkpoint_interval: int = 5  # Save every N epochs
    
    # Logging
    use_tensorboard: bool = False
    log_interval: int = 10  # Log every N batches


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    eval_batch_size: int = 32
    use_model_ensemble: bool = False
    save_predictions: bool = True
    save_confidence_scores: bool = True


class Config:
    """Master configuration class."""
    
    def __init__(self, 
                 data: DataConfig = None,
                 model: ModelConfig = None,
                 training: TrainingConfig = None,
                 augmentation: AugmentationConfig = None,
                 output: OutputConfig = None,
                 evaluation: EvaluationConfig = None):
        """Initialize configuration."""
        self.data = data or DataConfig()
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.augmentation = augmentation or AugmentationConfig()
        self.output = output or OutputConfig()
        self.evaluation = evaluation or EvaluationConfig()
        
        # Create output directories
        self._create_output_dirs()
    
    def _create_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        Path(self.output.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output.logs_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'augmentation': self.augmentation.__dict__,
            'output': self.output.__dict__,
            'evaluation': self.evaluation.__dict__
        }
    
    def print_config(self) -> None:
        """Print configuration."""
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        
        print("\n📊 DATA CONFIG")
        print(f"  Dataset path: {self.data.dataset_path}")
        print(f"  Batch size: {self.data.batch_size}")
        print(f"  Image size: {self.data.image_size}x{self.data.image_size}")
        print(f"  Train/Val/Test split: {self.data.train_split}/{self.data.val_split}/{self.data.test_split}")
        
        print("\n🧠 MODEL CONFIG")
        print(f"  Architecture: {self.model.model_name}")
        print(f"  Pretrained: {self.model.pretrained}")
        print(f"  Dropout: {self.model.dropout}")
        
        print("\n🔧 TRAINING CONFIG")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Optimizer: {self.training.optimizer}")
        print(f"  Loss function: {self.training.loss_function}")
        print(f"  Device: {self.training.device}")
        print(f"  Early stopping: {self.training.use_early_stopping}")
        
        print("\n🎨 AUGMENTATION CONFIG")
        print(f"  Horizontal flip: {self.augmentation.use_horizontal_flip}")
        print(f"  Rotation: {self.augmentation.use_rotation} ({self.augmentation.rotation_degrees}°)")
        print(f"  Color jitter: {self.augmentation.use_color_jitter}")
        print(f"  Gaussian blur: {self.augmentation.use_gaussian_blur}")
        
        print("\n📁 OUTPUT CONFIG")
        print(f"  Output dir: {self.output.output_dir}")
        print(f"  Save best model: {self.output.save_best_model}")
        
        print("="*70 + "\n")


# Preset configurations for common scenarios
class MobileNetV3Config(Config):
    """Configuration optimized for MobileNetV3."""
    
    def __init__(self):
        super().__init__(
            model=ModelConfig(model_name='mobilenetv3', pretrained=True, dropout=0.3),
            training=TrainingConfig(
                num_epochs=50,
                learning_rate=1e-4,
                weight_decay=1e-5,
                optimizer='adamw'
            ),
            data=DataConfig(
                batch_size=32,
                num_workers=4,
                image_size=224
            )
        )


class EfficientNetConfig(Config):
    """Configuration optimized for EfficientNet."""
    
    def __init__(self):
        super().__init__(
            model=ModelConfig(model_name='efficientnet', pretrained=True, dropout=0.3),
            training=TrainingConfig(
                num_epochs=50,
                learning_rate=5e-5,
                weight_decay=1e-5,
                optimizer='adamw'
            ),
            data=DataConfig(
                batch_size=32,
                num_workers=4,
                image_size=224
            )
        )


class ResNet50Config(Config):
    """Configuration optimized for ResNet50."""
    
    def __init__(self):
        super().__init__(
            model=ModelConfig(model_name='resnet50', pretrained=True, dropout=0.3),
            training=TrainingConfig(
                num_epochs=50,
                learning_rate=1e-4,
                weight_decay=1e-4,
                optimizer='adamw'
            ),
            data=DataConfig(
                batch_size=16,  # Smaller batch size for ResNet50 (larger model)
                num_workers=4,
                image_size=224
            )
        )


class LightweightConfig(Config):
    """Configuration for lightweight training (quick experiments)."""
    
    def __init__(self):
        super().__init__(
            model=ModelConfig(model_name='mobilenetv3', pretrained=True, dropout=0.3),
            training=TrainingConfig(
                num_epochs=10,
                learning_rate=1e-4,
                weight_decay=1e-5,
                early_stopping_patience=5
            ),
            data=DataConfig(
                batch_size=32,
                num_workers=2,
                image_size=224
            )
        )


class HighAccuracyConfig(Config):
    """Configuration for maximum accuracy (slower, higher compute)."""
    
    def __init__(self):
        super().__init__(
            model=ModelConfig(model_name='resnet50', pretrained=True, dropout=0.4),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=5e-5,
                weight_decay=1e-4,
                optimizer='adamw',
                early_stopping_patience=15
            ),
            data=DataConfig(
                batch_size=16,
                num_workers=4,
                image_size=256
            ),
            augmentation=AugmentationConfig(
                use_affine=True,
                use_gaussian_blur=True
            )
        )


# Quick configuration selector
def get_config(config_name: str = 'mobilenetv3') -> Config:
    """
    Get a configuration by name.
    
    Args:
        config_name (str): Name of the configuration
            - 'mobilenetv3': Optimized for MobileNetV3
            - 'efficientnet': Optimized for EfficientNet
            - 'resnet50': Optimized for ResNet50
            - 'lightweight': Quick experiments
            - 'high_accuracy': Maximum accuracy
            - 'default': Default configuration
    
    Returns:
        Config: Configuration object
    """
    configs = {
        'mobilenetv3': MobileNetV3Config,
        'efficientnet': EfficientNetConfig,
        'resnet50': ResNet50Config,
        'lightweight': LightweightConfig,
        'high_accuracy': HighAccuracyConfig,
        'default': Config
    }
    
    if config_name.lower() not in configs:
        print(f"Unknown configuration: {config_name}")
        print(f"Available options: {list(configs.keys())}")
        return Config()
    
    return configs[config_name.lower()]()


if __name__ == '__main__':
    print("Available configurations:")
    print("="*70)
    
    for config_name in ['mobilenetv3', 'efficientnet', 'resnet50', 'lightweight', 'high_accuracy']:
        cfg = get_config(config_name)
        cfg.print_config()
