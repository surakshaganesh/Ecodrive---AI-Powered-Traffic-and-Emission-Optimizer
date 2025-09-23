import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        self.cities_data = self.load_cities_data()
        self.emission_factors = self.load_emission_factors()
    
    def load_cities_data(self):
        """Load or generate cities data"""
        return {
            'bangalore': {
                'population': 12400000,
                'area_sq_km': 741,
                'major_routes': ['ORR', 'Hosur Road', 'Whitefield', 'Electronic City'],
                'peak_hours': [(8, 10), (18, 20)],
                'avg_vehicles_per_hour': 15000
            },
            'delhi': {
                'population': 30290000,
                'area_sq_km': 1484,
                'major_routes': ['Ring Road', 'NH-1', 'Noida Expressway', 'Gurgaon Expressway'],
                'peak_hours': [(8, 11), (17, 20)],
                'avg_vehicles_per_hour': 25000
            },
            'mumbai': {
                'population': 20400000,
                'area_sq_km': 603,
                'major_routes': ['Western Express', 'Eastern Express', 'SV Road'],
                'peak_hours': [(8, 11), (18, 21)],
                'avg_vehicles_per_hour': 20000
            },
            'hyderabad': {
                'population': 10500000,
                'area_sq_km': 650,
                'major_routes': ['ORR', 'Cyberabad', 'Airport Road', 'Gachibowli'],
                'peak_hours': [(8, 10), (18, 20)],
                'avg_vehicles_per_hour': 12000
            }
        }
    
    def load_emission_factors(self):
        """Load emission factors for different vehicles"""
        return {
            'car': {'co2_per_km': 0.21, 'fuel_efficiency': 15},
            'bike': {'co2_per_km': 0.089, 'fuel_efficiency': 35},
            'bus': {'co2_per_km': 0.105, 'fuel_efficiency': 4},
            'ev': {'co2_per_km': 0.05, 'fuel_efficiency': 'N/A'}
        }
    
    def generate_sample_traffic_data(self, city, days=30):
        """Generate sample traffic data for ML training"""
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        city_info = self.cities_data.get(city, self.cities_data['bangalore'])
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            for hour in range(24):
                # Determine if it's peak hour
                is_peak = any(start <= hour < end for start, end in city_info['peak_hours'])
                is_weekend = current_date.weekday() >= 5
                
                # Generate realistic traffic patterns
                base_traffic = city_info['avg_vehicles_per_hour']
                
                if is_peak and not is_weekend:
                    vehicles = int(base_traffic * np.random.uniform(1.5, 2.0))
                    congestion = np.random.randint(60, 95)
                elif is_weekend:
                    vehicles = int(base_traffic * np.random.uniform(0.4, 0.7))
                    congestion = np.random.randint(20, 50)
                else:
                    vehicles = int(base_traffic * np.random.uniform(0.6, 1.0))
                    congestion = np.random.randint(30, 60)
                
                # Add weather effect
                weather_factor = np.random.uniform(0.8, 1.2)
                vehicles = int(vehicles * weather_factor)
                congestion = min(100, int(congestion * weather_factor))
                
                data.append({
                    'datetime': current_date.replace(hour=hour),
                    'city': city,
                    'hour': hour,
                    'day_of_week': current_date.weekday(),
                    'is_weekend': is_weekend,
                    'is_peak': is_peak,
                    'vehicles_count': vehicles,
                    'congestion_level': congestion,
                    'avg_speed': max(10, 50 - (congestion * 0.4)),
                    'co2_emissions': vehicles * 0.5 * np.random.uniform(0.8, 1.2)
                })
        
        return pd.DataFrame(data)
    
    def process_real_time_data(self, raw_data):
        """Process real-time traffic data"""
        processed = {}
        
        for city, data in raw_data.items():
            processed[city] = {
                'congestion_level': self.normalize_congestion(data.get('congestion', 0)),
                'vehicle_count': data.get('vehicles', 0),
                'avg_speed': self.calculate_avg_speed(data.get('congestion', 0)),
                'co2_level': self.estimate_co2_level(data.get('vehicles', 0)),
                'timestamp': datetime.now().isoformat()
            }
        
        return processed
    
    def normalize_congestion(self, congestion):
        """Normalize congestion to 0-100 scale"""
        return max(0, min(100, congestion))
    
    def calculate_avg_speed(self, congestion):
        """Calculate average speed based on congestion"""
        max_speed = 50  # km/h
        return max(10, max_speed - (congestion * 0.4))
    
    def estimate_co2_level(self, vehicle_count):
        """Estimate CO2 emissions based on vehicle count"""
        base_emission = vehicle_count * 0.0005  # kg CO2 per vehicle
        return int(base_emission * 1000)  # Convert to grams for display