#!/usr/bin/env python
"""
Utility script to extract and organize the plant disease dataset from archive folder.
This script helps prepare the dataset for use with the disease detection model.
"""

import os
import sys
import shutil
import argparse
import logging
import zipfile
import glob
import traceback
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Allow truncated images to be loaded - helps with some corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def find_archive_files():
    """Find archive files in the dataset/archive directory."""
    # Search in common locations
    search_paths = [
        "dataset/archive",
        "archive",
        "data/archive",
        "datasets/archive"
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            zip_files = glob.glob(os.path.join(path, "*.zip"))
            if zip_files:
                return zip_files
    
    return []

def extract_archives(zip_files, output_dir, force=False):
    """Extract archive files to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    for zip_file in zip_files:
        zip_name = os.path.basename(zip_file)
        extract_dir = os.path.join(output_dir, os.path.splitext(zip_name)[0])
        
        # Skip if already extracted
        if os.path.exists(extract_dir) and not force:
            logger.info(f"Directory {extract_dir} already exists. Use --force to overwrite.")
            continue
            
        if os.path.exists(extract_dir) and force:
            logger.info(f"Removing existing directory {extract_dir}")
            shutil.rmtree(extract_dir)
        
        logger.info(f"Extracting {zip_name} to {extract_dir}")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Get total files for progress bar
            total_files = len(zip_ref.namelist())
            
            # Extract with progress bar
            with tqdm(total=total_files, desc=f"Extracting {zip_name}") as pbar:
                for member in zip_ref.namelist():
                    zip_ref.extract(member, path=extract_dir)
                    pbar.update(1)

def fix_image(img_path, output_path=None):
    """
    Try to fix a corrupted image by opening, processing, and saving it again.
    
    Args:
        img_path (str): Path to the image file
        output_path (str, optional): Path to save the fixed image. If None, overwrite original
        
    Returns:
        bool: True if fixed successfully, False otherwise
    """
    if output_path is None:
        output_path = img_path
        
    try:
        # Try to open the image
        with open(img_path, 'rb') as f:
            img_data = f.read()
            
        # Try to fix by re-encoding
        try:
            # First try the normal way
            img = Image.open(io.BytesIO(img_data))
            img.verify()  # This will raise an exception if the image is invalid
            return True  # Image is already good
        except Exception:
            # If that failed, try another approach
            try:
                # Try opening without verification
                img = Image.open(io.BytesIO(img_data))
                
                # Convert to RGB if needed (handles some PNG issues)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as JPG to standardize format
                img.save(output_path, 'JPEG', quality=90)
                return True
            except Exception:
                return False
    except Exception:
        return False

def organize_dataset(input_dir, output_dir, split_ratio=0.8, min_images_per_class=20):
    """Organize dataset into train/val splits by disease category."""
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist.")
        return False
            
    # Create output directory structure
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Track corrupt images
    corrupt_dir = os.path.join(output_dir, "corrupted")
    fixed_dir = os.path.join(output_dir, "fixed")
    os.makedirs(corrupt_dir, exist_ok=True)
    os.makedirs(fixed_dir, exist_ok=True)
    
    corrupt_count = 0
    fixed_count = 0
    
    # Find all image directories
    all_dirs = []
    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            if any(ext in f.lower() for f in os.listdir(os.path.join(root, dir_name)) 
                  for ext in ['.jpg', '.jpeg', '.png']):
                all_dirs.append(os.path.join(root, dir_name))
    
    # Process each image directory
    total_images = 0
    valid_images = 0
    skipped_classes = []
    
    for src_dir in all_dirs:
        # Get category name from directory
        category = os.path.basename(src_dir)
        logger.info(f"Processing category: {category}")
        
        # Create category directories
        train_category_dir = os.path.join(train_dir, category)
        val_category_dir = os.path.join(val_dir, category)
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(val_category_dir, exist_ok=True)
        
        # Get all images
        all_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            all_images.extend(glob.glob(os.path.join(src_dir, f"*{ext}")))
        
        if not all_images:
            logger.warning(f"No images found in {src_dir}")
            continue
                
        # Validate and fix images before splitting
        valid_images_list = []
        for img_path in tqdm(all_images, desc=f"Validating images for {category}"):
            try:
                # Try to open and verify the image
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                    img = Image.open(io.BytesIO(img_data))
                    img.verify()  # This will raise an exception if the image is invalid
                    valid_images_list.append(img_path)
            except Exception as e:
                # Try to fix the image
                fixed_path = os.path.join(fixed_dir, f"{category}_{os.path.basename(img_path)}")
                if fix_image(img_path, fixed_path):
                    logger.info(f"Fixed corrupted image: {img_path}")
                    valid_images_list.append(fixed_path)
                    fixed_count += 1
                else:
                    logger.warning(f"Corrupted image: {img_path} - {str(e)}")
                    # Copy corrupted file to corrupt directory for inspection
                    corrupt_filename = f"{category}_{os.path.basename(img_path)}"
                    corrupt_path = os.path.join(corrupt_dir, corrupt_filename)
                    try:
                        shutil.copy2(img_path, corrupt_path)
                        corrupt_count += 1
                    except Exception as copy_error:
                        logger.error(f"Error copying corrupt file: {str(copy_error)}")
        
        # Skip if not enough valid images
        if len(valid_images_list) < min_images_per_class:
            logger.warning(f"Skipping {category} - only {len(valid_images_list)} valid images (need at least {min_images_per_class})")
            skipped_classes.append(category)
            continue
            
        # Determine split
        split_idx = int(len(valid_images_list) * split_ratio)
        train_images = valid_images_list[:split_idx]
        val_images = valid_images_list[split_idx:]
        
        # Copy images to train set
        for img in tqdm(train_images, desc=f"Copying train images for {category}"):
            try:
                img_name = os.path.basename(img)
                dst_path = os.path.join(train_category_dir, img_name)
                standardize_image(img, dst_path)
            except Exception as e:
                logger.error(f"Error copying train image {img}: {str(e)}")
                
        # Copy images to validation set
        for img in tqdm(val_images, desc=f"Copying validation images for {category}"):
            try:
                img_name = os.path.basename(img)
                dst_path = os.path.join(val_category_dir, img_name)
                standardize_image(img, dst_path)
            except Exception as e:
                logger.error(f"Error copying validation image {img}: {str(e)}")
                
        total_images += len(all_images)
        valid_images += len(valid_images_list)
        logger.info(f"Added {len(train_images)} training and {len(val_images)} validation images for {category}")
    
    # Report on corrupted images
    if corrupt_count > 0:
        logger.warning(f"{corrupt_count} corrupted images were found and skipped")
        logger.info(f"Corrupted images were saved to {corrupt_dir} for inspection")
    
    if fixed_count > 0:
        logger.info(f"{fixed_count} images were fixed and included in the dataset")
    
    if skipped_classes:
        logger.warning(f"Skipped {len(skipped_classes)} classes with too few valid images: {', '.join(skipped_classes)}")
    
    logger.info(f"Dataset organization complete. Total images: {total_images}, Valid: {valid_images}")
    return True

def standardize_image(src_path, dst_path, target_size=(256, 256)):
    """
    Standardize an image to a consistent format and size
    
    Args:
        src_path (str): Source image path
        dst_path (str): Destination image path
        target_size (tuple): Target size for images (width, height)
    """
    try:
        # Open image
        with open(src_path, 'rb') as f:
            img = Image.open(io.BytesIO(f.read()))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Resize to target size preserving aspect ratio
            img.thumbnail(target_size, Image.LANCZOS)
            
            # Create new image with exact dimensions (padding if needed)
            new_img = Image.new("RGB", target_size, (255, 255, 255))
            
            # Paste original image centered on the new one
            pos_x = (target_size[0] - img.width) // 2
            pos_y = (target_size[1] - img.height) // 2
            new_img.paste(img, (pos_x, pos_y))
            
            # Save as JPEG
            new_img.save(dst_path, 'JPEG', quality=90)
    except Exception as e:
        logger.error(f"Error standardizing image {src_path}: {str(e)}")
        # If standardization fails, just copy the original
        shutil.copy2(src_path, dst_path)

def validate_dataset(dataset_path):
    """
    Validate dataset structure for training.
    
    Args:
        dataset_path (str): Path to the dataset
        
    Returns:
        bool: True if valid, False otherwise
    """
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        logger.error(f"Dataset must contain 'train' and 'val' directories")
        return False
    
    train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    val_classes = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    
    if not train_classes:
        logger.error(f"No class directories found in {train_dir}")
        return False
    
    if not val_classes:
        logger.error(f"No class directories found in {val_dir}")
        return False
    
    # Check that classes match
    train_set = set(train_classes)
    val_set = set(val_classes)
    
    if train_set != val_set:
        logger.warning(f"Train and validation class mismatch:")
        logger.warning(f"Only in train: {train_set - val_set}")
        logger.warning(f"Only in val: {val_set - train_set}")
    
    # Check each class has images
    empty_train = []
    empty_val = []
    
    for cls in train_classes:
        cls_dir = os.path.join(train_dir, cls)
        images = glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.jpeg")) + glob.glob(os.path.join(cls_dir, "*.png"))
        if not images:
            empty_train.append(cls)
    
    for cls in val_classes:
        cls_dir = os.path.join(val_dir, cls)
        images = glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.jpeg")) + glob.glob(os.path.join(cls_dir, "*.png"))
        if not images:
            empty_val.append(cls)
    
    if empty_train:
        logger.warning(f"Empty train classes: {empty_train}")
    
    if empty_val:
        logger.warning(f"Empty validation classes: {empty_val}")
    
    # Return True only if we have classes with images in both train and val
    return len(train_classes) > 0 and len(val_classes) > 0 and len(empty_train) < len(train_classes) and len(empty_val) < len(val_classes)

def main():
    parser = argparse.ArgumentParser(description="Extract and organize plant disease dataset")
    parser.add_argument("--extract-dir", type=str, default="dataset/extracted",
                        help="Directory to extract archives to")
    parser.add_argument("--output-dir", type=str, default="dataset/disease_images",
                        help="Directory to organize the final dataset")
    parser.add_argument("--split-ratio", type=float, default=0.8,
                        help="Train/validation split ratio (default: 0.8)")
    parser.add_argument("--min-images", type=int, default=20,
                        help="Minimum number of images per class (default: 20)")
    parser.add_argument("--force", action="store_true",
                        help="Force extraction even if directories exist")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction step (use if already extracted)")
    parser.add_argument("--input-dir", type=str,
                        help="Input directory with extracted images (if skipping extraction)")
    parser.add_argument("--standardize-size", type=int, default=256,
                        help="Standardize images to this size (default: 256)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate the dataset, don't process it")
    args = parser.parse_args()
    
    try:
        # Validation only mode
        if args.validate_only:
            if args.output_dir and os.path.exists(args.output_dir):
                logger.info(f"Validating dataset at {args.output_dir}")
                is_valid = validate_dataset(args.output_dir)
                if is_valid:
                    logger.info("Dataset validation successful")
                    return 0
                else:
                    logger.error("Dataset validation failed")
                    return 1
            else:
                logger.error(f"Output directory {args.output_dir} not found")
                return 1
        
        if not args.skip_extract:
            # Find and extract archive files
            zip_files = find_archive_files()
            if not zip_files:
                logger.error("No archive files found. Use --input-dir to specify extracted directory.")
                return 1
                
            logger.info(f"Found {len(zip_files)} archive files")
            extract_archives(zip_files, args.extract_dir, args.force)
            input_dir = args.extract_dir
        else:
            if not args.input_dir:
                logger.error("Must provide --input-dir when using --skip-extract")
                return 1
            input_dir = args.input_dir
        
        # Organize dataset
        success = organize_dataset(
            input_dir, 
            args.output_dir, 
            args.split_ratio,
            args.min_images
        )
        
        if not success:
            return 1
            
        # Validate the final dataset
        logger.info(f"Validating dataset at {args.output_dir}")
        is_valid = validate_dataset(args.output_dir)
        if is_valid:
            logger.info("Dataset validation successful")
        else:
            logger.warning("Dataset validation has warnings - check log for details")
            
        logger.info(f"Dataset ready at {args.output_dir}")
        logger.info(f"Train directory: {os.path.join(args.output_dir, 'train')}")
        logger.info(f"Validation directory: {os.path.join(args.output_dir, 'val')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 