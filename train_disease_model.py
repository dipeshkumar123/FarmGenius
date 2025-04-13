import os
import sys
import logging
import argparse
import glob
from dotenv import load_dotenv
from PIL import Image
import io
from tqdm import tqdm

# Add the project root directory to the path
sys.path.append(".")

from src.models.disease_model import DiseaseModel
from src.utils.file_utils import get_project_root

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_image_files(dataset_path):
    """
    Check for and remove corrupted image files from the dataset path.
    
    Args:
        dataset_path (str): Path to the image dataset
        
    Returns:
        tuple: (valid_count, corrupted_count, corrupted_files)
    """
    logger.info(f"Validating image files in {dataset_path}")
    
    valid_count = 0
    corrupted_count = 0
    corrupted_files = []
    
    # Supported image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Find all image files recursively
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(dataset_path, "**", f"*{ext}"), recursive=True))
    
    if not all_files:
        logger.warning(f"No image files found in {dataset_path}")
        return 0, 0, []
    
    logger.info(f"Found {len(all_files)} image files to validate")
    
    # Check each file
    for file_path in tqdm(all_files, desc="Validating images"):
        try:
            # Try to open and verify the image
            with open(file_path, 'rb') as f:
                img_data = f.read()
                img = Image.open(io.BytesIO(img_data))
                img.verify()  # Verify the image is valid
                valid_count += 1
        except Exception as e:
            logger.warning(f"Corrupted image found: {file_path} - {str(e)}")
            corrupted_files.append(file_path)
            corrupted_count += 1
    
    logger.info(f"Validation complete: {valid_count} valid images, {corrupted_count} corrupted images")
    return valid_count, corrupted_count, corrupted_files

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train disease image classification model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--dataset", type=str, help="Custom dataset path (if not using the default)")
    parser.add_argument("--img-size", type=int, default=224, help="Image size for training")
    parser.add_argument("--api-key", type=str, help="API key for external disease detection service")
    parser.add_argument("--api-url", type=str, help="URL for external disease detection service")
    parser.add_argument("--skip-validation", action="store_true", help="Skip image validation step")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Set dataset path
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = os.path.join(get_project_root(), 'dataset', 'disease_images')
        # If disease_images doesn't exist, try the archive folder
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(get_project_root(), 'dataset', 'archive')
            
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist")
        return 1
    
    # Check for train and validation directories
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        logger.info(f"Found train and validation directories in {dataset_path}")
        use_split_dirs = True
    else:
        logger.info(f"No train/val split found in {dataset_path}, will use raw dataset directory")
        use_split_dirs = False
    
    # Validate images to avoid training errors
    if not args.skip_validation:
        if use_split_dirs:
            # Validate both train and val directories
            logger.info("Validating train directory...")
            train_valid, train_corrupted, train_files = validate_image_files(train_dir)
            logger.info("Validating validation directory...")
            val_valid, val_corrupted, val_files = validate_image_files(val_dir)
            valid_count = train_valid + val_valid
            corrupted_count = train_corrupted + val_corrupted
            corrupted_files = train_files + val_files
        else:
            valid_count, corrupted_count, corrupted_files = validate_image_files(dataset_path)
        
        # If we found corrupted files, we should handle them
        if corrupted_count > 0:
            logger.warning(f"Found {corrupted_count} corrupted image files")
            
            # Remove corrupted files to prevent training errors
            if input(f"Remove {corrupted_count} corrupted files? (y/n): ").lower() == 'y':
                for file_path in corrupted_files:
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed corrupted file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing {file_path}: {str(e)}")
    
    # Initialize the disease model
    try:
        logger.info("Initializing disease model")
        model = DiseaseModel()
        
        # Start model training
        logger.info(f"Starting model training with {args.epochs} epochs and batch size {args.batch_size}")
        
        # If we have split directories, use specific train and val paths
        if use_split_dirs:
            result = model.train_image_model_with_split(
                train_dir=train_dir,
                val_dir=val_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                img_size=(args.img_size, args.img_size)
            )
        else:
            result = model.train_image_model(
                dataset_path=dataset_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                img_size=(args.img_size, args.img_size)
            )
        
        # Check training result
        if result.get('success', False):
            logger.info("Model training completed successfully")
            logger.info(f"Model saved to: {result['model_path']}")
            logger.info(f"Classes: {len(result['classes'])} disease types")
            logger.info(f"Validation accuracy: {result['accuracy'] * 100:.1f}%")
            return 0
        else:
            logger.error(f"Training failed: {result.get('message', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 