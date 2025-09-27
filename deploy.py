# deploy.py
"""
EcoDrive AI Deployment Script
Handles model loading, health checks, and production deployment
"""

import streamlit as st
import pickle
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelDeployment:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_info = {}
        self.deployment_status = "Initializing"
        
    def load_models(self):
        """Load trained models for production use"""
        try:
            logger.info("Loading trained models...")
            
            # Load congestion prediction model
            with open('models/congestion_model.pkl', 'rb') as f:
                self.models['congestion'] = pickle.load(f)
            
            # Load emission prediction model
            with open('models/emission_model.pkl', 'rb') as f:
                self.models['emission'] = pickle.load(f)
            
            # Load scalers
            if os.path.exists('models/congestion_scaler.pkl'):
                with open('models/congestion_scaler.pkl', 'rb') as f:
                    self.scalers['congestion'] = pickle.load(f)
            
            # Load model performance metrics
            if os.path.exists('models/model_performance.json'):
                with open('models/model_performance.json', 'r') as f:
                    self.model_info = json.load(f)
            
            self.deployment_status = "Models Loaded Successfully"
            logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            self.deployment_status = f"Model Loading Failed: {str(e)}"
            logger.error(f"Failed to load models: {str(e)}")
            return False
    
    def health_check(self):
        """Perform system health check"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'models_loaded': len(self.models) > 0,
            'deployment_status': self.deployment_status,
            'available_models': list(self.models.keys()),
            'model_performance': self.model_info
        }
        
        return health_status
    
    def predict_congestion(self, features):
        """Make congestion prediction using loaded model"""
        try:
            if 'congestion' not in self.models:
                raise Exception("Congestion model not loaded")
            
            if 'congestion' in self.scalers:
                features_scaled = self.scalers['congestion'].transform([features])
                prediction = self.models['congestion'].predict(features_scaled)[0]
            else:
                prediction = self.models['congestion'].predict([features])[0]
            
            return max(0, min(100, prediction))
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return 50  # Default fallback value
    
    def get_model_info(self):
        """Get model information for monitoring"""
        return {
            'deployment_time': datetime.now().isoformat(),
            'model_count': len(self.models),
            'status': self.deployment_status,
            'performance_metrics': self.model_info
        }

# Global deployment instance
deployment = ModelDeployment()

def initialize_deployment():
    """Initialize deployment on app startup"""
    if not deployment.models:
        success = deployment.load_models()
        if success:
            logger.info("Deployment initialization completed")
        else:
            st.error("❌ Model deployment failed!")
            logger.error("Deployment initialization failed")
    
    return deployment