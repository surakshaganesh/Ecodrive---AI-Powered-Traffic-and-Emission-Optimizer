"""
EcoDrive AI - Model Training Script
This script trains machine learning models for traffic prediction and route optimization
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TrafficModelTrainer:
    def __init__(self):
        self.cities = ['bangalore', 'delhi', 'mumbai', 'hyderabad']
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        
    def generate_training_dataset(self, days=365, samples_per_day=24):
        """Generate comprehensive training dataset"""
        print("Generating training dataset...")
        
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        city_profiles = {
            'bangalore': {
                'base_vehicles': 15000,
                'peak_multiplier': 2.2,
                'peak_hours': [(7, 10), (17, 20)],
                'weather_impact': 0.15
            },
            'delhi': {
                'base_vehicles': 25000,
                'peak_multiplier': 2.5,
                'peak_hours': [(8, 11), (17, 21)],
                'weather_impact': 0.20
            },
            'mumbai': {
                'base_vehicles': 20000,
                'peak_multiplier': 2.8,
                'peak_hours': [(8, 11), (18, 22)],
                'weather_impact': 0.25
            },
            'hyderabad': {
                'base_vehicles': 12000,
                'peak_multiplier': 2.0,
                'peak_hours': [(8, 10), (18, 20)],
                'weather_impact': 0.10
            }
        }
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            for city in self.cities:
                profile = city_profiles[city]
                
                for hour in range(24):
                    # Time-based features
                    is_weekend = current_date.weekday() >= 5
                    is_peak = any(start <= hour < end for start, end in profile['peak_hours'])
                    
                    # Generate realistic traffic patterns
                    base_traffic = profile['base_vehicles']
                    
                    if is_peak and not is_weekend:
                        vehicles = int(base_traffic * profile['peak_multiplier'] * np.random.uniform(0.8, 1.2))
                        congestion = np.random.randint(65, 95)
                    elif is_weekend:
                        vehicles = int(base_traffic * np.random.uniform(0.4, 0.8))
                        congestion = np.random.randint(15, 45)
                    else:
                        vehicles = int(base_traffic * np.random.uniform(0.6, 1.0))
                        congestion = np.random.randint(25, 65)
                    
                    # Weather impact
                    weather_score = np.random.uniform(0.5, 1.0)
                    weather_factor = 1 + (1 - weather_score) * profile['weather_impact']
                    
                    vehicles = int(vehicles * weather_factor)
                    congestion = min(100, int(congestion * weather_factor))
                    
                    # Calculate derived metrics
                    avg_speed = max(8, 45 - (congestion * 0.35))
                    co2_emissions = vehicles * np.random.uniform(0.15, 0.25)
                    
                    data.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'city': city,
                        'hour': hour,
                        'day_of_week': current_date.weekday(),
                        'month': current_date.month,
                        'is_weekend': int(is_weekend),
                        'is_peak': int(is_peak),
                        'weather_score': round(weather_score, 3),
                        'temperature': np.random.uniform(15, 40),
                        'humidity': np.random.uniform(30, 90),
                        'vehicles_count': vehicles,
                        'congestion_level': congestion,
                        'avg_speed': round(avg_speed, 1),
                        'co2_emissions': round(co2_emissions, 2)
                    })
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} training samples")
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        print("Preparing features...")
        
        # Create additional features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Traffic intensity feature
        df['traffic_intensity'] = df['vehicles_count'] / df['avg_speed']
        
        # City encoding (one-hot)
        city_dummies = pd.get_dummies(df['city'], prefix='city')
        df = pd.concat([df, city_dummies], axis=1)
        
        return df
    
    def train_congestion_model(self, df):
        """Train traffic congestion prediction model"""
        print("Training congestion prediction model...")
        
        feature_columns = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak',
                          'weather_score', 'temperature', 'humidity', 'vehicles_count',
                          'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                          'traffic_intensity'] + [col for col in df.columns if col.startswith('city_')]
        
        X = df[feature_columns]
        y = df['congestion_level']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_model = RandomForestRegressor(random_state=42)
        rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='r2', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        
        # Train XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_pred = rf_grid.predict(X_test)
        xgb_pred = xgb_model.predict(X_test_scaled)
        
        rf_r2 = r2_score(y_test, rf_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)
        
        print(f"Random Forest R² Score: {rf_r2:.4f}")
        print(f"XGBoost R² Score: {xgb_r2:.4f}")
        
        # Choose best model
        if rf_r2 > xgb_r2:
            best_model = rf_grid.best_estimator_
            model_name = "RandomForest"
            predictions = rf_pred
        else:
            best_model = xgb_model
            model_name = "XGBoost"
            predictions = xgb_pred
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        self.models['congestion'] = best_model
        self.scalers['congestion'] = scaler
        self.performance_metrics['congestion'] = {
            'model_type': model_name,
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'accuracy_percentage': min(r2 * 100, 95)
        }
        
        return best_model, scaler, feature_columns
    
    def train_emission_model(self, df):
        """Train CO2 emission prediction model"""
        print("Training emission prediction model...")
        
        feature_columns = ['vehicles_count', 'congestion_level', 'avg_speed', 'temperature', 'humidity']
        
        X = df[feature_columns]
        y = df['co2_emissions']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        self.models['emission'] = model
        self.performance_metrics['emission'] = {
            'model_type': 'RandomForest',
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'accuracy_percentage': min(r2 * 100, 95)
        }
        
        print(f"Emission Model R² Score: {r2:.4f}")
        return model, feature_columns
    
    def save_models(self):
        """Save trained models and scalers"""
        print("Saving models...")
        
        # Save models
        for model_name, model in self.models.items():
            with open(f'models/{model_name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            with open(f'models/{scaler_name}_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save performance metrics
        with open('models/model_performance.json', 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        print("Models saved successfully!")
    
    def generate_model_report(self):
        """Generate comprehensive model performance report"""
        print("\n" + "="*50)
        print("ECODRIVE AI - MODEL TRAINING REPORT")
        print("="*50)
        
        for model_name, metrics in self.performance_metrics.items():
            print(f"\n{model_name.upper()} MODEL:")
            print(f"  Model Type: {metrics['model_type']}")
            print(f"  R² Score: {metrics['r2_score']:.4f}")
            print(f"  Mean Squared Error: {metrics['mse']:.4f}")
            print(f"  Mean Absolute Error: {metrics['mae']:.4f}")
            print(f"  Accuracy: {metrics['accuracy_percentage']:.1f}%")
        
        print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)

def main():
    """Main training pipeline"""
    print("Starting EcoDrive AI Model Training...")
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Initialize trainer
    trainer = TrafficModelTrainer()
    
    # Generate and save training data
    df = trainer.generate_training_dataset(days=365)
    df.to_csv('data/traffic_training_data.csv', index=False)
    print("Training data saved to data/traffic_training_data.csv")
    
    # Prepare features
    df = trainer.prepare_features(df)
    
    # Train models
    congestion_model, scaler, feature_cols = trainer.train_congestion_model(df)
    emission_model, emission_features = trainer.train_emission_model(df)
    
    # Save models
    trainer.save_models()
    
    # Generate report
    trainer.generate_model_report()
    
    print("\nModel training completed successfully!")
    print("Files created:")
    print("  - models/congestion_model.pkl")
    print("  - models/emission_model.pkl")
    print("  - models/congestion_scaler.pkl")
    print("  - models/model_performance.json")
    print("  - data/traffic_training_data.csv")

if __name__ == "__main__":
    main()