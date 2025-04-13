import os
import sys
import logging
import argparse
import glob
from dotenv import load_dotenv

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test disease image classification")
    parser.add_argument("--image", type=str, help="Path to a single image file to classify")
    parser.add_argument("--dir", type=str, help="Directory of test images (will test all images)")
    parser.add_argument("--api-url", type=str, help="URL for external disease detection service")
    parser.add_argument("--use-live", action="store_true", help="Use live API instead of local model")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check inputs
    if not args.image and not args.dir:
        parser.error("You must provide either --image or --dir")
        return 1
    
    try:
        # Initialize the disease model
        model = DiseaseModel()
        
        # Test single image
        if args.image:
            if not os.path.exists(args.image):
                logger.error(f"Image not found: {args.image}")
                return 1
                
            logger.info(f"Testing classification for: {args.image}")
            result = model.identify_disease_from_image(args.image)
            
            print_classification_result(result, args.image)
            
        # Test all images in a directory
        elif args.dir:
            if not os.path.exists(args.dir):
                logger.error(f"Directory not found: {args.dir}")
                return 1
                
            # Get all image files
            image_files = []
            for ext in ['jpg', 'jpeg', 'png']:
                image_files.extend(glob.glob(os.path.join(args.dir, f'*.{ext}')))
                image_files.extend(glob.glob(os.path.join(args.dir, f'*.{ext.upper()}')))
            
            if not image_files:
                logger.error(f"No image files found in directory: {args.dir}")
                return 1
                
            logger.info(f"Found {len(image_files)} images to classify")
            
            # Process each image
            correct_count = 0
            for img_path in image_files:
                result = model.identify_disease_from_image(img_path)
                
                # Get ground truth from directory structure (assuming format: "Crop Disease/image.jpg")
                parts = os.path.basename(os.path.dirname(img_path)).split()
                if len(parts) > 1:
                    ground_truth = " ".join(parts)
                    
                    # Check if top prediction matches ground truth
                    if result.get('found', False):
                        top_prediction = result['results'][0]['name']
                        if ground_truth.lower() in top_prediction.lower():
                            correct_count += 1
                
                print_classification_result(result, img_path)
                print("-" * 60)
            
            # Print accuracy if we have ground truth labels
            if len(image_files) > 0:
                accuracy = correct_count / len(image_files)
                logger.info(f"Classification accuracy: {accuracy:.2f} ({correct_count}/{len(image_files)})")
        
        return 0
            
    except Exception as e:
        logger.error(f"Error during image classification: {str(e)}")
        return 1

def print_classification_result(result, image_path):
    """Print classification result in a readable format."""
    print(f"\nImage: {os.path.basename(image_path)}")
    
    if result.get('found', False):
        print("Classification results:")
        for i, prediction in enumerate(result['results'], 1):
            confidence = prediction.get('confidence', 0) * 100
            print(f"  {i}. {prediction['name']} ({confidence:.1f}%)")
            
        print("\nDetailed information:")
        top_match = result['results'][0]
        if 'crop' in top_match:
            print(f"  Crop: {top_match['crop']}")
        if 'type' in top_match and top_match['type'] != 'unknown':
            print(f"  Disease type: {top_match['type']}")
        if 'severity' in top_match and top_match['severity'] != 'unknown':
            print(f"  Severity: {top_match['severity']}")
        if 'symptoms' in top_match:
            print(f"  Symptoms: {top_match['symptoms']}")
    else:
        print(f"No disease identified: {result.get('message', 'Unknown error')}")

if __name__ == "__main__":
    sys.exit(main()) 