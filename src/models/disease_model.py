import logging
import os
import json
import re
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import io
import tensorflow as tf
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image

from src.utils.file_utils import get_project_root, ensure_directory_exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if TensorFlow is available
HAS_TENSORFLOW = True

class DiseaseModel:
    """
    Model for plant disease detection using a locally trained deep learning model.
    The model is trained on a dataset of plant disease images and can identify
    various crop diseases without requiring external API calls.
    """

    def __init__(self):
        """Initialize the disease detection model."""
        self.model = None
        self.class_names = None
        self.disease_db = None
        self._initialize_model()
        self._load_disease_database()
        
    def _initialize_model(self):
        """Initialize the trained disease detection model."""
        try:
            model_path = os.path.join(get_project_root(), 'models', 'disease_model.h5')
            class_map_path = os.path.join(get_project_root(), 'models', 'disease_class_map.json')
            
            if not os.path.exists(model_path) or not os.path.exists(class_map_path):
                logger.warning(f"Model file not found at {model_path} or class names not found at {class_map_path}")
                return
            
            # Load the trained model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load class names
            with open(class_map_path, 'r') as f:
                class_data = json.load(f)
                self.class_names = class_data.get('classes', [])
                
            logger.info(f"Loaded disease detection model with {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"Error initializing disease model: {str(e)}")
            self.model = None
            self.class_names = None

    def identify_disease_from_image(self, image_path: str, crop: Optional[str] = None) -> Dict[str, Any]:
        """
        Identify plant disease from an image using the local model.
        
        Args:
            image_path (str): Path to the image file
            crop (str, optional): Specific crop to focus on
            
        Returns:
            Dict[str, Any]: Disease identification results
        """
        try:
            if not self.model or not self.class_names:
                return {
                    "found": False,
                    "message": "Disease detection model not properly initialized",
                    "results": []
                }
            
            # Load and preprocess the image
            img = self._load_and_preprocess_image(image_path)
            if img is None:
                return {
                    "found": False,
                    "message": "Failed to load or process image",
                    "results": []
                }
            
            # Make prediction
            predictions = self.model.predict(img)
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[::-1][:5]  # Get top 5 predictions
            results = []
            
            for idx in top_indices:
                confidence = float(predictions[0][idx])
                disease_name = self.class_names[idx]
                disease_info = self._get_disease_info(disease_name)
                
                # If crop is specified, only include matching predictions
                if crop:
                    crop = crop.lower()
                    disease_name_lower = disease_name.lower()
                    # More flexible crop matching
                    if not any(crop in part for part in disease_name_lower.split()):
                        continue
                
                result = {
                    "name": disease_name,
                    "confidence": confidence,
                    "crop": disease_info.get("crop", "unknown"),
                    "type": disease_info.get("type", "unknown"),
                    "severity": "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low",
                    "symptoms": disease_info.get("symptoms", []),
                    "treatment": disease_info.get("treatment", [])
                }
                results.append(result)
            
            if not results:
                return {
                    "found": False,
                    "message": "No diseases identified. Please try with a clearer image or different angle.",
                    "results": []
                }
            
            # Sort results by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "found": True,
                "message": "Disease identification successful",
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error identifying disease from image: {str(e)}")
            return {
                "found": False,
                "message": f"Error processing image: {str(e)}",
                "results": []
            }

    def _load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess an image for model input.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Optional[np.ndarray]: Preprocessed image array or None if loading fails
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to target size
            img = img.resize((224, 224), Image.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Ensure correct shape (224, 224, 3)
            if len(img_array.shape) == 3:
                # Add batch dimension
                img_array = np.expand_dims(img_array, axis=0)
            
            # If we somehow got an extra dimension, remove it
            if len(img_array.shape) == 5:
                img_array = np.squeeze(img_array, axis=1)
            
            # Double check the shape
            if img_array.shape != (1, 224, 224, 3):
                logger.warning(f"Unexpected image shape: {img_array.shape}")
                # Reshape if needed
                img_array = img_array.reshape(1, 224, 224, 3)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    def _get_disease_info(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a disease."""
        return self.disease_db.get(disease_name)

    def _load_disease_database(self):
        """Load the disease database from file."""
        try:
            db_path = os.path.join(get_project_root(), 'data', 'disease_database.json')
            if os.path.exists(db_path):
                with open(db_path, 'r', encoding='utf-8') as f:
                    self.disease_db = json.load(f)
            else:
                # Create a basic disease database with supported diseases
                self.disease_db = {
                    "Cashew anthracnose": {
                        "crop": "Cashew",
                        "type": "Fungal",
                        "symptoms": ["Dark brown to black spots on leaves", "Circular or irregular lesions", "Premature leaf drop"],
                        "treatment": ["Remove infected plant parts", "Apply fungicide", "Improve air circulation"]
                    },
                    "Cashew gumosis": {
                        "crop": "Cashew",
                        "type": "Bacterial",
                        "symptoms": ["Gum exudation from trunk", "Bark discoloration", "Tree decline"],
                        "treatment": ["Prune infected branches", "Apply copper-based bactericide", "Maintain tree health"]
                    },
                    "Cashew healthy": {
                        "crop": "Cashew",
                        "type": "Healthy",
                        "symptoms": ["Normal growth", "Green leaves", "No lesions"],
                        "treatment": ["Regular maintenance", "Proper nutrition", "Monitor for pests"]
                    },
                    "Cashew leaf miner": {
                        "crop": "Cashew",
                        "type": "Pest",
                        "symptoms": ["Mining trails in leaves", "Leaf distortion", "Reduced photosynthesis"],
                        "treatment": ["Remove affected leaves", "Apply insecticide", "Use pheromone traps"]
                    },
                    "Cashew red rust": {
                        "crop": "Cashew",
                        "type": "Fungal",
                        "symptoms": ["Reddish-brown spots", "Leaf yellowing", "Premature defoliation"],
                        "treatment": ["Fungicide application", "Remove infected leaves", "Improve drainage"]
                    },
                    "Cassava bacterial blight": {
                        "crop": "Cassava",
                        "type": "Bacterial",
                        "symptoms": ["Water-soaked lesions", "Leaf wilting", "Stem cankers"],
                        "treatment": ["Use disease-free cuttings", "Apply copper-based bactericide", "Crop rotation"]
                    },
                    "Cassava brown spot": {
                        "crop": "Cassava",
                        "type": "Fungal",
                        "symptoms": ["Brown spots on leaves", "Leaf yellowing", "Reduced yield"],
                        "treatment": ["Remove infected leaves", "Apply fungicide", "Improve soil fertility"]
                    },
                    "Cassava green mite": {
                        "crop": "Cassava",
                        "type": "Pest",
                        "symptoms": ["Leaf discoloration", "Stunted growth", "Reduced yield"],
                        "treatment": ["Apply acaricide", "Introduce natural predators", "Remove infested plants"]
                    },
                    "Cassava healthy": {
                        "crop": "Cassava",
                        "type": "Healthy",
                        "symptoms": ["Normal growth", "Green leaves", "No lesions"],
                        "treatment": ["Regular maintenance", "Proper nutrition", "Monitor for pests"]
                    },
                    "Cassava mosaic": {
                        "crop": "Cassava",
                        "type": "Viral",
                        "symptoms": ["Mosaic pattern on leaves", "Stunted growth", "Reduced yield"],
                        "treatment": ["Use virus-free planting material", "Control whiteflies", "Remove infected plants"]
                    },
                    "Maize fall armyworm": {
                        "crop": "Maize",
                        "type": "Pest",
                        "symptoms": ["Leaf damage", "Stem boring", "Ear damage"],
                        "treatment": ["Apply insecticide", "Use Bt maize", "Early planting"]
                    },
                    "Maize grasshoper": {
                        "crop": "Maize",
                        "type": "Pest",
                        "symptoms": ["Leaf defoliation", "Stem damage", "Reduced yield"],
                        "treatment": ["Apply insecticide", "Use bait traps", "Natural predators"]
                    },
                    "Maize healthy": {
                        "crop": "Maize",
                        "type": "Healthy",
                        "symptoms": ["Normal growth", "Green leaves", "No lesions"],
                        "treatment": ["Regular maintenance", "Proper nutrition", "Monitor for pests"]
                    },
                    "Maize leaf beetle": {
                        "crop": "Maize",
                        "type": "Pest",
                        "symptoms": ["Leaf skeletonization", "Reduced photosynthesis", "Yield loss"],
                        "treatment": ["Apply insecticide", "Crop rotation", "Early planting"]
                    },
                    "Maize leaf blight": {
                        "crop": "Maize",
                        "type": "Fungal",
                        "symptoms": ["Large brown lesions", "Leaf death", "Reduced yield"],
                        "treatment": ["Use resistant varieties", "Apply fungicide", "Crop rotation"]
                    },
                    "Maize leaf spot": {
                        "crop": "Maize",
                        "type": "Fungal",
                        "symptoms": ["Small brown spots", "Leaf yellowing", "Premature death"],
                        "treatment": ["Remove infected leaves", "Apply fungicide", "Improve air circulation"]
                    },
                    "Maize streak virus": {
                        "crop": "Maize",
                        "type": "Viral",
                        "symptoms": ["Yellow streaks on leaves", "Stunted growth", "Reduced yield"],
                        "treatment": ["Use resistant varieties", "Control leafhoppers", "Remove infected plants"]
                    },
                    "Tomato healthy": {
                        "crop": "Tomato",
                        "type": "Healthy",
                        "symptoms": ["Normal growth", "Green leaves", "No lesions"],
                        "treatment": ["Regular maintenance", "Proper nutrition", "Monitor for pests"]
                    },
                    "Tomato leaf blight": {
                        "crop": "Tomato",
                        "type": "Fungal",
                        "symptoms": ["Brown spots with yellow halos", "Leaf death", "Fruit rot"],
                        "treatment": ["Remove infected leaves", "Apply fungicide", "Improve air circulation"]
                    },
                    "Tomato leaf curl": {
                        "crop": "Tomato",
                        "type": "Viral",
                        "symptoms": ["Leaf curling", "Stunted growth", "Reduced yield"],
                        "treatment": ["Use virus-free plants", "Control whiteflies", "Remove infected plants"]
                    },
                    "Tomato septoria leaf spot": {
                        "crop": "Tomato",
                        "type": "Fungal",
                        "symptoms": ["Small brown spots with gray centers", "Leaf yellowing", "Defoliation"],
                        "treatment": ["Remove infected leaves", "Apply fungicide", "Improve air circulation"]
                    },
                    "Tomato verticulium wilt": {
                        "crop": "Tomato",
                        "type": "Fungal",
                        "symptoms": ["Yellowing and wilting", "Brown vascular tissue", "Plant death"],
                        "treatment": ["Use resistant varieties", "Soil fumigation", "Crop rotation"]
                    }
                }
                
                # Save the database to file
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                with open(db_path, 'w', encoding='utf-8') as f:
                    json.dump(self.disease_db, f, indent=4)
                    
                logger.info("Created new disease database with information for all supported diseases")
            
        except Exception as e:
            logger.error(f"Error loading disease database: {str(e)}")
            self.disease_db = {}
    
    def get_management_advice(self, disease_id):
        """
        Get management advice for a specific disease.
        
        Args:
            disease_id (str): Disease identifier
        
        Returns:
            dict: Management advice including:
                - found: Whether the disease was found
                - id: Disease ID
                - name: Disease name
                - scientific_name: Scientific name
                - type: Disease type
                - crop: Affected crop
                - symptoms: Disease symptoms
                - conditions: Favorable conditions
                - management: Management strategies
                - severity: Disease severity
        """
        try:
            # Search in disease database
            for disease in self.disease_db:
                if disease['id'] == disease_id:
                    return {
                        'found': True,
                        'id': disease['id'],
                        'name': disease['name'],
                        'scientific_name': disease['scientific_name'],
                        'type': disease['type'],
                        'crop': disease['crop'],
                        'symptoms': disease['symptoms'],
                        'conditions': disease['conditions'],
                        'management': disease['management'],
                        'severity': disease['severity']
                    }
            
            # If disease not found
            return {
                'found': False,
                'message': f"Disease with ID '{disease_id}' not found in the database"
            }
            
        except Exception as e:
            logger.error(f"Error getting disease management advice: {str(e)}")
            return {
                'found': False,
                'message': f"Error retrieving disease management advice: {str(e)}"
            }

    def get_diseases_by_crop(self, crop_name):
        """
        Get common diseases for a specific crop.
        
        Args:
            crop_name (str): Name of the crop
        
        Returns:
            dict: Diseases affecting the crop
        """
        try:
            crop_name = crop_name.lower()
            
            if crop_name in self.crop_diseases and self.crop_diseases[crop_name]:
                diseases = [
                    {
                        'id': d['id'],
                        'name': d['name'],
                        'type': d['type'],
                        'severity': d['severity']
                    }
                    for d in self.crop_diseases[crop_name]
                ]
                
                return {
                    'found': True,
                    'crop': crop_name,
                    'diseases': diseases,
                    'message': f"Found {len(diseases)} diseases for {crop_name}"
                }
            else:
                return {
                    'found': False,
                    'crop': crop_name,
                    'message': f"No disease information available for {crop_name}",
                    'diseases': []
                }
        except Exception as e:
            logger.error(f"Error getting diseases for crop {crop_name}: {str(e)}")
            return {
                'found': False,
                'crop': crop_name,
                'message': f"Error getting diseases: {str(e)}",
                'diseases': []
            }
    
    def format_identification_response(self, identification_result):
        """
        Format disease identification result into a natural language response.
        
        Args:
            identification_result (dict): Disease identification result
        
        Returns:
            str: Formatted response
        """
        if not identification_result.get('found', False):
            return "I couldn't identify any diseases based on that description. Please provide more details about the symptoms you're observing."
            
        # Get the top match
        top_match = identification_result['results'][0]
        
        response = f"Based on your description, this could be **{top_match['name']}** ({top_match['disease_info'].get('scientific_name', 'Unknown')}), "
        response += f"a {top_match['disease_info'].get('type', 'Unknown')} disease that affects {top_match['disease_info'].get('crop', 'Unknown')}.\n\n"
        
        response += f"**Symptoms**: {top_match['disease_info'].get('symptoms', 'No symptoms information available')}\n\n"
        response += f"**Management**: {top_match['disease_info'].get('management', 'No management information available')}\n\n"
        
        if len(identification_result['results']) > 1:
            response += "Other possibilities include:\n"
            for i, disease in enumerate(identification_result['results'][1:3], 1):  # Show up to 2 alternatives
                response += f"{i}. {disease['name']} ({disease['disease_info'].get('scientific_name', 'Unknown')})\n"
                response += f"   Symptoms: {disease['disease_info'].get('symptoms', 'No symptoms information available')}\n"
                response += f"   Management: {disease['disease_info'].get('management', 'No management information available')}\n\n"
            
            response += "\nUse the disease management endpoint with the disease ID for treatment information."
        
        return response
    
    def format_image_identification_response(self, identification_result):
        """
        Format image-based disease identification result into a natural language response.
        
        Args:
            identification_result (dict): Disease identification result from image
        
        Returns:
            str: Formatted response
        """
        if not identification_result.get('found', False):
            return "I couldn't identify any diseases from this image. Please provide a clearer image of the affected plant."
            
        # Get the top match
        top_match = identification_result['results'][0]
        confidence = top_match['confidence'] * 100
        
        response = f"Based on the image, this appears to be **{top_match['name']}** "
        response += f"with {confidence:.1f}% confidence.\n\n"
        
        if 'symptoms' in top_match['disease_info']:
            response += f"**Symptoms**: {top_match['disease_info']['symptoms']}\n\n"
        
        if len(identification_result['results']) > 1:
            response += "Other possibilities include:\n"
            for i, disease in enumerate(identification_result['results'][1:3], 1):  # Show up to 2 alternatives
                conf = disease['confidence'] * 100
                response += f"{i}. {disease['name']} ({conf:.1f}% confidence)\n"
            response += "\nGet management advice with the disease ID for treatment information."
        
        return response
    
    def format_management_advice(self, management_result):
        """
        Format disease management advice into a natural language response.
        
        Args:
            management_result (dict): Disease management result
        
        Returns:
            str: Formatted response
        """
        if not management_result.get('found', False):
            return management_result.get('message', "Disease information not found.")
            
        response = f"# Managing {management_result['name']} ({management_result['scientific_name']})\n\n"
        
        response += f"**Crop affected**: {management_result['crop']}\n"
        response += f"**Type**: {management_result['type']} disease\n"
        response += f"**Severity**: {management_result['severity'].capitalize()}\n\n"
        
        response += f"**Symptoms**: {management_result['symptoms']}\n\n"
        response += f"**Favorable conditions**: {management_result['conditions']}\n\n"
        response += f"**Management strategies**: {management_result['management']}\n"
        
        return response

    def _preprocess_image(self, image_path):
        """
        Preprocess an image for model input.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        try:
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path)
            
            if image is None:
                return None
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocess image: {str(e)}")
            return None

    def train_image_model(self, dataset_path: str, epochs: int = 10, batch_size: int = 32, img_size: Tuple[int, int] = (224, 224)) -> Dict[str, Any]:
        """
        Train the disease detection model on a dataset.
        
        Args:
            dataset_path (str): Path to the dataset directory
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            img_size (tuple): Input image size (height, width)
            
        Returns:
            dict: Training results
        """
        try:
            # Create data generators
            train_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
            val_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input
            )
            
            # Load and prepare the dataset
            train_generator = train_datagen.flow_from_directory(
                os.path.join(dataset_path, 'train'),
                target_size=img_size,
                batch_size=batch_size,
                class_mode='categorical'
            )
            
            val_generator = val_datagen.flow_from_directory(
                os.path.join(dataset_path, 'val'),
                target_size=img_size,
                batch_size=batch_size,
                class_mode='categorical'
            )
            
            # Create the model
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(img_size[0], img_size[1], 3)
            )
            
            # Add custom layers
            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                Dense(1024, activation='relu'),
                Dropout(0.5),
                Dense(len(train_generator.class_indices), activation='softmax')
            ])
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Set up callbacks
            checkpoint_path = os.path.join(get_project_root(), 'models', 'disease_model.h5')
            callbacks = [
                ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
            
            # Train the model
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks
            )
            
            # Save class mapping
            class_mapping = {
                'classes': list(train_generator.class_indices.keys()),
                'class_indices': train_generator.class_indices
            }
            
            class_map_path = os.path.join(get_project_root(), 'models', 'disease_class_map.json')
            with open(class_map_path, 'w') as f:
                json.dump(class_mapping, f, indent=4)
            
            # Get final accuracy
            _, accuracy = model.evaluate(val_generator)
            
            return {
                'success': True,
                'model_path': checkpoint_path,
                'classes': class_mapping['classes'],
                'accuracy': accuracy,
                'history': history.history
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {
                'success': False,
                'message': str(e)
            }
            
    def train_image_model_with_split(self, train_dir: str, val_dir: str, **kwargs) -> Dict[str, Any]:
        """Train model with pre-split train/val directories."""
        try:
            # Create data generators
            train_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
            val_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input
            )
            
            # Load and prepare the dataset
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=kwargs.get('img_size', (224, 224)),
                batch_size=kwargs.get('batch_size', 32),
                class_mode='categorical'
            )
            
            val_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=kwargs.get('img_size', (224, 224)),
                batch_size=kwargs.get('batch_size', 32),
                class_mode='categorical'
            )
            
            # Create the model
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(kwargs.get('img_size', (224, 224))[0], kwargs.get('img_size', (224, 224))[1], 3)
            )
            
            # Add custom layers
            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                Dense(1024, activation='relu'),
                Dropout(0.5),
                Dense(len(train_generator.class_indices), activation='softmax')
            ])
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Set up callbacks
            checkpoint_path = os.path.join(get_project_root(), 'models', 'disease_model.h5')
            callbacks = [
                ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
            
            # Train the model
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=kwargs.get('epochs', 10),
                callbacks=callbacks
            )
            
            # Save class mapping
            class_mapping = {
                'classes': list(train_generator.class_indices.keys()),
                'class_indices': train_generator.class_indices
            }
            
            class_map_path = os.path.join(get_project_root(), 'models', 'disease_class_map.json')
            with open(class_map_path, 'w') as f:
                json.dump(class_mapping, f, indent=4)
            
            # Get final accuracy
            _, accuracy = model.evaluate(val_generator)
            
            return {
                'success': True,
                'model_path': checkpoint_path,
                'classes': class_mapping['classes'],
                'accuracy': accuracy,
                'history': history.history
            }
            
        except Exception as e:
            logger.error(f"Error training model with split: {str(e)}")
            return {
                'success': False,
                'message': str(e)
            }

    def identify_disease(self, description: str, crop: Optional[str] = None) -> Dict[str, Any]:
        """
        Identify plant disease from text description.
        
        Args:
            description (str): Text description of the disease symptoms
            crop (str, optional): Specific crop to focus on
            
        Returns:
            Dict[str, Any]: Disease identification results
        """
        try:
            if not self.disease_db:
                return {
                    "found": False,
                    "message": "Disease database not properly initialized",
                    "results": []
                }
            
            # Convert description to lowercase for matching
            description = description.lower()
            
            # Initialize results list
            results = []
            
            # Split description into words for more flexible matching
            description_words = set(re.findall(r'\b\w+\b', description))
            
            # Look for crop names in the description
            detected_crops = []
            for disease_name, info in self.disease_db.items():
                crop_name = info.get("crop", "").lower()
                if crop_name and crop_name in description:
                    detected_crops.append(crop_name)
            
            # If crop not explicitly provided but detected in description, use it
            if not crop and detected_crops:
                crop = detected_crops[0]
                logger.info(f"Detected crop from description: {crop}")
            
            # Search through disease database
            for disease_name, disease_info in self.disease_db.items():
                # Initialize match score
                symptom_matches = 0
                total_symptoms = 0
                
                # Skip if crop is specified and doesn't match
                if crop:
                    crop = crop.lower()
                    disease_crop = disease_info.get("crop", "").lower()
                    if crop not in disease_crop:
                        continue
                
                # Get symptoms and count them
                symptoms = disease_info.get("symptoms", [])
                total_symptoms = len(symptoms)
                
                if total_symptoms == 0:
                    continue  # Skip entries without symptoms
                
                # Check each symptom for matches
                for symptom in symptoms:
                    symptom_lower = symptom.lower()
                    
                    # Direct match - highest score
                    if symptom_lower in description:
                        symptom_matches += 3
                        continue
                    
                    # Word level matching - partial score
                    symptom_words = set(re.findall(r'\b\w+\b', symptom_lower))
                    matching_words = symptom_words.intersection(description_words)
                    
                    if len(matching_words) > 0:
                        match_ratio = len(matching_words) / len(symptom_words)
                        if match_ratio >= 0.5:  # At least half the words match
                            symptom_matches += 2
                        elif match_ratio > 0.25:  # At least a quarter of words match
                            symptom_matches += 1
                
                # Calculate confidence based on symptom matching
                if symptom_matches > 0:
                    # Normalize confidence between 0-1
                    # The division by (3 * total_symptoms) accounts for the max score possible
                    confidence = min(symptom_matches / (3 * total_symptoms), 1.0)
                    
                    # Create result entry
                    result = {
                        "name": disease_name,
                        "confidence": confidence,
                        "crop": disease_info.get("crop", "unknown"),
                        "type": disease_info.get("type", "unknown"),
                        "severity": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low",
                        "symptoms": symptoms,
                        "treatment": disease_info.get("treatment", [])
                    }
                    results.append(result)
            
            # For better results, always return top matches
            # Sort results by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
            # If at least one result has reasonable confidence (>10%), consider it found
            if results and results[0]["confidence"] >= 0.1:
                return {
                    "found": True,
                    "message": "Disease identification successful",
                    "results": results[:5]  # Return top 5 results
                }
            
            # Return partial results if we have any, but mark as not found
            if results:
                return {
                    "found": False,
                    "message": "Possible diseases found, but confidence is low. Please provide more specific symptoms.",
                    "results": results[:3]  # Return top 3 low-confidence results
                }
            
            return {
                "found": False,
                "message": "No matching diseases found based on the description. Please provide more details about the symptoms.",
                "results": []
            }
            
        except Exception as e:
            logger.error(f"Error identifying disease from description: {str(e)}")
            return {
                "found": False,
                "message": f"Error processing description: {str(e)}",
                "results": []
            }


def main():
    """Test the disease model."""
    model = DiseaseModel()
    
    # Test text-based disease identification
    test_queries = [
        "My rice plants have diamond-shaped spots with gray centers",
        "White powdery stuff on my wheat leaves",
        "Potato leaves with water-soaked spots and white growth underneath",
        "Small insects on my crops causing distorted leaves",
        "Corn leaves with long gray lesions"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Get identification results
        result = model.identify_disease_from_image(query)
        
        # Print formatted response
        if result["found"]:
            print(f"Identified as: {result['results'][0]['name']}")
            print(f"Match score: {result['results'][0]['confidence']}")
            print("\nFormatted response:")
            print(model.format_identification_response(result))
        else:
            print(f"No match found: {result['message']}")
    
    # If TensorFlow is available, train an image model
    if HAS_TENSORFLOW:
        print("\nTesting image model training...")
        dataset_path = os.path.join(get_project_root(), 'dataset', 'archive')
        if os.path.exists(dataset_path):
            # Don't actually train in the test (takes too long), just show it would work
            print(f"Would train on dataset: {dataset_path}")
            print("(Training skipped for this test)")
        else:
            print(f"Dataset not found at {dataset_path}")


if __name__ == "__main__":
    main() 