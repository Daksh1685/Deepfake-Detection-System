"""
Data organization script for deepfake detection project.

This script helps organize images from the raw folder structure
(real_vs_fake/train/fake, real_vs_fake/train/real, etc.)
to the training structure (dataset/fake, dataset/real).
"""

import shutil
from pathlib import Path
from tqdm import tqdm


def organize_dataset(source_root: str = '../real_vs_fake', 
                    target_root: str = 'dataset'):
    """
    Organize images from raw structure to training structure.
    
    Source structure:
        real_vs_fake/
        ├── train/
        │   ├── real/
        │   └── fake/
        ├── test/
        │   ├── real/
        │   └── fake/
        └── valid/
            ├── real/
            └── fake/
    
    Target structure:
        dataset/
        ├── real/
        └── fake/
    
    Args:
        source_root (str): Root directory of raw images
        target_root (str): Target directory for organized images
    """
    source_path = Path(source_root)
    target_path = Path(target_root)
    
    # Create target directories
    real_dir = target_path / 'real'
    fake_dir = target_path / 'fake'
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Dataset Organization")
    print(f"{'='*70}")
    print(f"Source: {source_path.absolute()}")
    print(f"Target: {target_path.absolute()}\n")
    
    # Process each split (train, test, valid)
    splits = ['train', 'test', 'valid']
    total_copied = 0
    
    for split in splits:
        split_path = source_path / split
        
        if not split_path.exists():
            print(f"⚠️  {split_path} not found, skipping...")
            continue
        
        print(f"Processing {split.upper()} split:")
        
        # Process real images
        real_path = split_path / 'real'
        if real_path.exists():
            real_files = list(real_path.glob('*'))
            print(f"  Real images: {len(real_files)}", end=' ... ')
            
            for file in tqdm(real_files, leave=False):
                if file.is_file() and file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}:
                    # Create unique filename to avoid overwrites
                    target_file = real_dir / f"{split}_{file.name}"
                    shutil.copy2(file, target_file)
                    total_copied += 1
            
            print(f"✓ {len(real_files)} copied")
        
        # Process fake images
        fake_path = split_path / 'fake'
        if fake_path.exists():
            fake_files = list(fake_path.glob('*'))
            print(f"  Fake images: {len(fake_files)}", end=' ... ')
            
            for file in tqdm(fake_files, leave=False):
                if file.is_file() and file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}:
                    # Create unique filename to avoid overwrites
                    target_file = fake_dir / f"{split}_{file.name}"
                    shutil.copy2(file, target_file)
                    total_copied += 1
            
            print(f"✓ {len(fake_files)} copied")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total images organized: {total_copied:,}")
    
    # Count final images
    real_count = len(list(real_dir.glob('*')))
    fake_count = len(list(fake_dir.glob('*')))
    
    print(f"Real images in dataset/real: {real_count:,}")
    print(f"Fake images in dataset/fake: {fake_count:,}")
    print(f"Total: {real_count + fake_count:,}")
    print(f"{'='*70}\n")
    
    return total_copied


def verify_dataset(dataset_root: str = 'dataset'):
    """
    Verify the organized dataset.
    
    Args:
        dataset_root (str): Root directory of the dataset
    """
    dataset_path = Path(dataset_root)
    
    print(f"\n{'='*70}")
    print(f"Dataset Verification")
    print(f"{'='*70}\n")
    
    real_dir = dataset_path / 'real'
    fake_dir = dataset_path / 'fake'
    
    if not real_dir.exists():
        print("❌ Real directory not found!")
        return False
    
    if not fake_dir.exists():
        print("❌ Fake directory not found!")
        return False
    
    real_images = [f for f in real_dir.glob('*') if f.is_file()]
    fake_images = [f for f in fake_dir.glob('*') if f.is_file()]
    
    print(f"✓ Real images: {len(real_images):,}")
    print(f"✓ Fake images: {len(fake_images):,}")
    print(f"✓ Total images: {len(real_images) + len(fake_images):,}")
    
    if len(real_images) == 0 or len(fake_images) == 0:
        print("\n⚠️  Warning: Some categories are empty!")
        return False
    
    print(f"\n✓ Dataset is ready for training!")
    print(f"{'='*70}\n")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize dataset for training')
    parser.add_argument('--source', type=str, default='../real_vs_fake',
                       help='Source directory with raw images (default: ../real_vs_fake)')
    parser.add_argument('--target', type=str, default='dataset',
                       help='Target directory (default: dataset)')
    
    args = parser.parse_args()
    
    # Organize dataset
    organize_dataset(source_root=args.source, target_root=args.target)
    
    # Verify
    verify_dataset(dataset_root=args.target)
