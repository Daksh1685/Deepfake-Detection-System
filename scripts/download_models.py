"""
Download pre-trained models for deepfake detection.

Supports multiple sources:
1. Hugging Face (recommended for ML projects)
2. Google Drive (easy to setup)
3. GitHub Releases (simple, requires auth)
"""

import os
import sys
from pathlib import Path
from typing import Optional, Literal


class ModelDownloader:
    """Manage model downloads from various sources."""
    
    MODELS = {
        'mobilenetv3': {
            'hf': 'yourusername/deepfake-models/mobilenetv3_best_model.pth',
            'gdrive': 'GOOGLE_DRIVE_ID_1',
            'size': '39MB'
        },
        'efficientnet': {
            'hf': 'yourusername/deepfake-models/efficientnet_best_model.pth',
            'gdrive': 'GOOGLE_DRIVE_ID_2',
            'size': '52MB'
        },
        'resnet50': {
            'hf': 'yourusername/deepfake-models/resnet50_best_model.pth',
            'gdrive': 'GOOGLE_DRIVE_ID_3',
            'size': '283MB'
        }
    }
    
    def __init__(self):
        self.model_dir = Path('outputs/models')
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def download_from_huggingface(self, model_name: str, token: Optional[str] = None) -> Path:
        """
        Download from Hugging Face.
        
        Args:
            model_name: Model name (mobilenetv3, efficientnet, resnet50)
            token: Hugging Face API token (optional for public repos)
        
        Returns:
            Path to downloaded model
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("❌ huggingface-hub not installed. Install with:")
            print("   pip install huggingface-hub")
            return None
        
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = self.model_dir / f'{model_name}_best_model.pth'
        if model_path.exists():
            print(f"✓ {model_name} already exists: {model_path}")
            return model_path
        
        print(f"⏳ Downloading {model_name} from Hugging Face...")
        
        try:
            hf_repo = self.MODELS[model_name]['hf']
            hf_hub_download(
                repo_id=hf_repo.rsplit('/', 1)[0],
                filename=hf_repo.rsplit('/', 1)[1],
                local_dir=str(self.model_dir),
                token=token
            )
            print(f"✓ {model_name} downloaded successfully!")
            return model_path
        except Exception as e:
            print(f"❌ Failed to download from Hugging Face: {e}")
            return None
    
    def download_from_gdrive(self, model_name: str) -> Optional[Path]:
        """
        Download from Google Drive.
        
        Args:
            model_name: Model name (mobilenetv3, efficientnet, resnet50)
        
        Returns:
            Path to downloaded model
        """
        try:
            import gdown
        except ImportError:
            print("❌ gdown not installed. Install with:")
            print("   pip install gdown")
            return None
        
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = self.model_dir / f'{model_name}_best_model.pth'
        if model_path.exists():
            print(f"✓ {model_name} already exists: {model_path}")
            return model_path
        
        print(f"⏳ Downloading {model_name} from Google Drive...")
        
        try:
            file_id = self.MODELS[model_name]['gdrive']
            gdown.download(
                f'https://drive.google.com/uc?id={file_id}',
                str(model_path),
                quiet=False
            )
            print(f"✓ {model_name} downloaded successfully!")
            return model_path
        except Exception as e:
            print(f"❌ Failed to download from Google Drive: {e}")
            return None
    
    def download_all(self, source: Literal['huggingface', 'gdrive'] = 'huggingface', 
                     token: Optional[str] = None) -> dict:
        """
        Download all models.
        
        Args:
            source: Download source (huggingface or gdrive)
            token: API token if needed
        
        Returns:
            Dictionary with download results
        """
        results = {}
        
        for model_name in self.MODELS.keys():
            print(f"\n{'='*60}")
            
            if source == 'huggingface':
                path = self.download_from_huggingface(model_name, token)
            elif source == 'gdrive':
                path = self.download_from_gdrive(model_name)
            else:
                raise ValueError(f"Unknown source: {source}")
            
            results[model_name] = {
                'success': path is not None,
                'path': str(path) if path else None
            }
        
        print(f"\n{'='*60}")
        print("Download Summary:")
        print(f"{'='*60}")
        
        for model_name, result in results.items():
            status = "✓" if result['success'] else "✗"
            print(f"{status} {model_name}: {result.get('path', 'Failed')}")
        
        return results


def main():
    """Command-line interface for model downloading."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download pre-trained deepfake detection models'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='huggingface',
        choices=['huggingface', 'gdrive'],
        help='Download source (default: huggingface)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['all', 'mobilenetv3', 'efficientnet', 'resnet50'],
        help='Model to download (default: all)'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='API token for authentication (HF token for Hugging Face)'
    )
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    
    print("="*60)
    print("DEEPFAKE DETECTION - MODEL DOWNLOADER")
    print("="*60)
    
    if args.model == 'all':
        results = downloader.download_all(source=args.source, token=args.token)
    else:
        print(f"\nDownloading {args.model} from {args.source}...")
        
        if args.source == 'huggingface':
            path = downloader.download_from_huggingface(args.model, args.token)
        else:
            path = downloader.download_from_gdrive(args.model)
        
        results = {
            args.model: {
                'success': path is not None,
                'path': str(path) if path else None
            }
        }
    
    # Exit with appropriate code
    all_success = all(r['success'] for r in results.values())
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
