"""Dataset loading and preprocessing utilities for deepfake detection."""

import os
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Import transforms from utils
from utils.transforms import get_train_transforms, get_val_transforms


class DeepfakeDataset(Dataset):
    """
    Custom PyTorch Dataset for deepfake detection.
    
    Loads images from real/ and fake/ directories and assigns binary labels.
    Labels: 0 = Real, 1 = Fake
    """
    
    def __init__(self, image_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Initialize the DeepfakeDataset.
        
        Args:
            image_dir (str): Path to the dataset directory containing 'real' and 'fake' folders
            transform (transforms.Compose, optional): Image transformations to apply
        
        Raises:
            ValueError: If directory structure is invalid
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Validate directory structure
        real_dir = self.image_dir / "real"
        fake_dir = self.image_dir / "fake"
        
        if not real_dir.exists() or not fake_dir.exists():
            raise ValueError(
                f"Invalid directory structure. Expected 'real' and 'fake' folders in {image_dir}"
            )
        
        # Load image paths and assign labels
        self._load_images(real_dir, label=0)  # Real = 0
        self._load_images(fake_dir, label=1)  # Fake = 1
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {image_dir}")
    
    def _load_images(self, folder_path: Path, label: int) -> None:
        """
        Load image paths from a folder recursively.
        
        Args:
            folder_path (Path): Path to the folder containing images
            label (int): Label to assign (0 for real, 1 for fake)
        """
        # Supported image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        
        for file_path in folder_path.rglob('*'):
            if file_path.suffix.lower() in valid_extensions:
                self.images.append(str(file_path))
                self.labels.append(label)
    
    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an image and its label.
        
        Args:
            idx (int): Index of the image
        
        Returns:
            Tuple[torch.Tensor, int]: Image tensor and label (0 for real, 1 for fake)
        
        Raises:
            IOError: If image cannot be loaded
        """
        image_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Load image in RGB format
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            dict: Count of real and fake images
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = {
            'real': int(counts[0]) if 0 in unique else 0,
            'fake': int(counts[1]) if 1 in unique else 0
        }
        return distribution



def create_dataloaders(
    dataset_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes for data loading
        train_split (float): Proportion of data for training (default: 0.7)
        val_split (float): Proportion of data for validation (default: 0.15)
        test_split (float): Proportion of data for testing (default: 0.15)
        seed (int): Random seed for reproducibility
        shuffle (bool): Whether to shuffle data in dataloaders
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders
    
    Raises:
        ValueError: If splits don't sum to 1.0
    """
    # Validate splits
    total_split = train_split + val_split + test_split
    if not np.isclose(total_split, 1.0):
        raise ValueError(f"Splits must sum to 1.0, got {total_split}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create full dataset
    full_dataset = DeepfakeDataset(
        image_dir=dataset_dir,
        transform=get_val_transforms()
    )
    
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Random split
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create new datasets with appropriate transforms
    train_dataset.dataset = DeepfakeDataset(
        image_dir=dataset_dir,
        transform=get_train_transforms()
    )
    val_dataset.dataset = DeepfakeDataset(
        image_dir=dataset_dir,
        transform=get_val_transforms()
    )
    test_dataset.dataset = DeepfakeDataset(
        image_dir=dataset_dir,
        transform=get_val_transforms()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_single_dataloader(
    dataset_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
    is_train: bool = False
) -> DataLoader:
    """
    Create a single DataLoader for the entire dataset.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        shuffle (bool): Whether to shuffle data
        is_train (bool): Whether this is training (applies augmentation)
    
    Returns:
        DataLoader: DataLoader for the dataset
    """
    transform = get_train_transforms() if is_train else get_val_transforms()
    dataset = DeepfakeDataset(image_dir=dataset_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
    
    return dataloader
