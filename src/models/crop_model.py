import os
import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from src.utils.file_utils import load_csv_data, get_project_root, ensure_directory_exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CropRecommendationModel:
    """
    Machine learning model to recommend suitable crops based on soil and environmental parameters.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the crop recommendation model.
        
        Args:
            model_path (str, optional): Path to a pre-trained model file
        """
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'N', 'P', 'K', 'ph', 'EC', 'S', 'Cu', 'Fe', 'Mn', 'Zn', 'B'
        ]
        
        # Load pre-trained model if provided
        if model_path:
            self.load(model_path)
        else:
            self._load_default_model()
    
    def _load_default_model(self):
        """Load the model from the default location if available."""
        model_dir = os.path.join(get_project_root(), 'models')
        model_path = os.path.join(model_dir, 'crop_recommendation_model.pkl')
        
        if os.path.exists(model_path):
            try:
                self.load(model_path)
                logger.info(f"Loaded crop recommendation model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                logger.info("Initializing new model")
                self.model = RandomForestClassifier(random_state=42)
                self.scaler = StandardScaler()
        else:
            logger.info("No pre-trained model found. Initializing new model.")
            self.model = RandomForestClassifier(random_state=42)
            self.scaler = StandardScaler()
    
    def train(self, data=None, test_size=0.2, random_state=42, save_path=None):
        """
        Train the crop recommendation model.
        
        Args:
            data (DataFrame, optional): Training data
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            save_path (str, optional): Path to save the trained model
        
        Returns:
            dict: Training results
        """
        # Load data if not provided
        if data is None:
            try:
                # Try to load processed data first
                data = load_csv_data('crop_train_data.csv', directory='processed')
                logger.info(f"Loaded processed crop training data with {len(data)} samples")
            except Exception as e:
                logger.warning(f"Could not load processed data: {str(e)}")
                # Fall back to original data
                data = load_csv_data('crop_data.csv')
                logger.info(f"Loaded original crop data with {len(data)} samples")
        
        # Get features and target
        X = data[self.feature_columns]
        y = data['label']
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Define parameter grid for grid search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Use a smaller grid if the dataset is large
        if len(data) > 1000:
            param_grid = {
                'n_estimators': [100],
                'max_depth': [None, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        
        # Initialize and train the model
        logger.info("Training crop recommendation model with GridSearchCV...")
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=random_state),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Model training completed with accuracy: {accuracy:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Save the model if a path is provided
        if save_path:
            self.save(save_path)
        else:
            # Use default path
            model_dir = os.path.join(get_project_root(), 'models')
            ensure_directory_exists(model_dir)
            default_path = os.path.join(model_dir, 'crop_recommendation_model.pkl')
            self.save(default_path)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'best_params': grid_search.best_params_
        }
    
    def predict(self, soil_params):
        """
        Predict the most suitable crops based on soil parameters.
        
        Args:
            soil_params (dict): Dictionary with soil parameters
                Required keys: N, P, K, ph
                Optional keys: EC, S, Cu, Fe, Mn, Zn, B
        
        Returns:
            dict: Dictionary with predictions
        """
        try:
            if not self.model:
                # If model is not trained, return default recommendations
                default_recommendations = [
                    {"crop": "wheat", "confidence": 0.8},
                    {"crop": "rice", "confidence": 0.7},
                    {"crop": "maize", "confidence": 0.6},
                    {"crop": "cotton", "confidence": 0.5},
                    {"crop": "sugarcane", "confidence": 0.4}
                ]
                return {
                    "input_params": soil_params,
                    "top_recommendations": default_recommendations,
                    "all_recommendations": default_recommendations,
                    "is_default": True
                }
            
            # Create a dataframe from the input parameters
            input_data = {}
            
            # Fill in provided values
            for col in self.feature_columns:
                input_data[col] = soil_params.get(col, 0)  # Default to 0 if not provided
            
            input_df = pd.DataFrame([input_data])
            
            # Scale the input data
            input_scaled = self.scaler.transform(input_df)
            
            # Get class probabilities
            probabilities = self.model.predict_proba(input_scaled)[0]
            classes = self.model.classes_
            
            # Create ordered list of recommendations
            recommendations = []
            for i, cls in enumerate(classes):
                recommendations.append((cls, probabilities[i]))
            
            # Sort by probability (descending)
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            # Return top recommendations with probabilities
            top_recommendations = [
                {"crop": crop, "confidence": float(conf)}
                for crop, conf in recommendations[:5]
            ]
            
            return {
                "input_params": soil_params,
                "top_recommendations": top_recommendations,
                "all_recommendations": [
                    {"crop": crop, "confidence": float(conf)}
                    for crop, conf in recommendations
                ],
                "is_default": False
            }
        except Exception as e:
            logger.error(f"Error making crop prediction: {str(e)}")
            # Return default recommendations on error
            default_recommendations = [
                {"crop": "wheat", "confidence": 0.8},
                {"crop": "rice", "confidence": 0.7},
                {"crop": "maize", "confidence": 0.6},
                {"crop": "cotton", "confidence": 0.5},
                {"crop": "sugarcane", "confidence": 0.4}
            ]
            return {
                "input_params": soil_params,
                "top_recommendations": default_recommendations,
                "all_recommendations": default_recommendations,
                "is_default": True,
                "error": str(e)
            }
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to the model file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        logger.info(f"Model loaded from {filepath}")

def get_crop_description(crop_name):
    """
    Get a description of a crop.
    
    Args:
        crop_name (str): Name of the crop
    
    Returns:
        dict: Dictionary with crop information
    """
    # Dictionary of crop descriptions
    crop_info = {
        'pomegranate': {
            'name': 'Pomegranate',
            'scientific_name': 'Punica granatum',
            'description': 'A fruit-bearing deciduous shrub with vibrant red fruits containing edible seeds.',
            'growing_season': '3-5 years to full production',
            'water_needs': 'Moderate; drought-tolerant once established',
            'ideal_soil': 'Well-drained, slightly acidic to alkaline soil (pH 5.5-7.5)',
            'uses': 'Fresh consumption, juice, medicinal purposes',
            'typical_yield': '15-20 tons per hectare in mature orchards'
        },
        'mango': {
            'name': 'Mango',
            'scientific_name': 'Mangifera indica',
            'description': 'A tropical fruit tree with sweet, pulpy fruits.',
            'growing_season': '4-5 years to initial fruiting, 10+ years to full production',
            'water_needs': 'High during establishment and flowering/fruiting',
            'ideal_soil': 'Deep, well-drained loamy soil (pH 5.5-7.5)',
            'uses': 'Fresh consumption, juice, preserves, pickles',
            'typical_yield': '10-20 tons per hectare in mature orchards'
        },
        'grapes': {
            'name': 'Grapes',
            'scientific_name': 'Vitis vinifera',
            'description': 'A woody vine producing berries in clusters.',
            'growing_season': '2-3 years to initial production, 4-5 for full yield',
            'water_needs': 'Moderate; too much can reduce quality',
            'ideal_soil': 'Well-drained sandy or sandy loam (pH 5.5-6.5)',
            'uses': 'Fresh consumption, wine, juice, raisins',
            'typical_yield': '10-15 tons per hectare for quality wine grapes'
        }
    }
    
    # Return information for the requested crop
    return crop_info.get(crop_name.lower(), {
        'name': crop_name.capitalize(),
        'description': 'No detailed information available for this crop.',
        'ideal_soil': 'Consult a local agricultural extension for specific growing conditions.'
    })

def generate_recommendation_text(crop_name, confidence, soil_params=None):
    """
    Generate a natural language recommendation text.
    
    Args:
        crop_name (str): Name of the recommended crop
        confidence (float): Model confidence
        soil_params (dict, optional): Soil parameters for context
    
    Returns:
        str: Natural language recommendation
    """
    crop_info = get_crop_description(crop_name)
    
    # Format the confidence as a percentage
    confidence_pct = f"{confidence * 100:.1f}%"
    
    # Start with the main recommendation
    recommendation = f"{crop_info['name']} is recommended for your soil conditions with {confidence_pct} confidence. "
    
    # Add soil parameter context if available
    if soil_params:
        recommendation += f"Based on your soil's NPK values of {soil_params.get('N', 'N/A')}-{soil_params.get('P', 'N/A')}-{soil_params.get('K', 'N/A')} "
        if 'ph' in soil_params:
            recommendation += f"and pH of {soil_params.get('ph', 'N/A')}, "
    
    # Add growing information
    recommendation += f"{crop_info['name']} grows best in {crop_info.get('ideal_soil', 'well-prepared soil')}. "
    
    # Add a description of the crop
    recommendation += f"{crop_info.get('description', '')}"
    
    return recommendation

def main():
    """Test the crop recommendation model."""
    # Initialize and train the model
    model = CropRecommendationModel()
    
    try:
        # Try loading training data
        train_results = model.train()
        print(f"Model trained with accuracy: {train_results['accuracy']:.4f}")
        
        # Test with some sample soil parameters
        test_params = [
            {'N': 150, 'P': 50, 'K': 200, 'ph': 6.5, 'EC': 0.5, 'S': 0.2, 'Cu': 15, 'Fe': 100, 'Mn': 50, 'Zn': 40, 'B': 30},
            {'N': 100, 'P': 80, 'K': 150, 'ph': 5.5, 'EC': 1.0, 'S': 0.3, 'Cu': 12, 'Fe': 150, 'Mn': 60, 'Zn': 25, 'B': 60},
            {'N': 120, 'P': 40, 'K': 80, 'ph': 6.0, 'EC': 0.2, 'S': 0.1, 'Cu': 18, 'Fe': 80, 'Mn': 45, 'Zn': 35, 'B': 40}
        ]
        
        for i, params in enumerate(test_params):
            print(f"\nTest {i+1}:")
            predictions = model.predict(params)
            top_crop = predictions['top_recommendations'][0]
            
            print(f"Top recommendation: {top_crop['crop']} (confidence: {top_crop['confidence']:.4f})")
            print(f"Recommendation text: {generate_recommendation_text(top_crop['crop'], top_crop['confidence'], params)}")
            
            print("All recommendations:")
            for rec in predictions['top_recommendations']:
                print(f"  - {rec['crop']}: {rec['confidence']:.4f}")
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        logger.error(f"Error testing model: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 