"""Image transformation and augmentation utilities for deepfake detection."""

from torchvision import transforms


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get training image transformations with data augmentation.
    
    Applies the following transformations:
    - Resize to 224x224
    - Random horizontal flip (50% probability)
    - Random rotation (±10 degrees)
    - Color jitter (brightness, contrast, saturation)
    - Gaussian blur (random)
    - Convert to tensor
    - Normalize using ImageNet statistics
    
    Args:
        image_size (int): Target image size (default: 224)
    
    Returns:
        transforms.Compose: Composition of augmentation transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 2.0)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test image transformations (no augmentation).
    
    Applies the following transformations:
    - Resize to 224x224
    - Convert to tensor
    - Normalize using ImageNet statistics
    
    Args:
        image_size (int): Target image size (default: 224)
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


# Alias functions for backward compatibility
def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Alias for train_transforms (backward compatible)."""
    return train_transforms(image_size)


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Alias for val_transforms (backward compatible)."""
    return val_transforms(image_size)
