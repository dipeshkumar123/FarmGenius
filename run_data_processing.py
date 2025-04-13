#!/usr/bin/env python
"""
Script to run the data processing pipeline for the Farm Chatbot project.
This prepares all datasets for use in training models and running the chatbot.
"""

import os
import sys
import logging
from datetime import datetime

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the data processing function
from src.data_processing.process_all_data import process_all_datasets

if __name__ == "__main__":
    print("=" * 80)
    print(f"Farm Chatbot - Data Processing Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        summary = process_all_datasets()
        
        print("\nProcessing completed successfully!")
        print(f"Time taken: {summary['processing_time']}")
        print("Datasets processed:")
        for dataset in summary['datasets_processed']:
            print(f"  - {dataset}")
        
        print("\nNext steps:")
        print("1. Check the processed data in the 'dataset/processed' directory")
        print("2. Proceed with training models or running the chatbot")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        logging.error(f"Error during processing: {str(e)}", exc_info=True)
        
    print("\n" + "=" * 80) 