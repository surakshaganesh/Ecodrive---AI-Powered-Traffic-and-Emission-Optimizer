# monitoring.py
"""
EcoDrive AI Monitoring and Logging System
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os

class SystemMonitor:
    def __init__(self):
        self.log_file = 'logs/system_metrics.json'
        self.ensure_log_file()
    
    def ensure_log_file(self):
        """Ensure log directory and file exist"""
        os.makedirs('logs', exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)
    
    def log_prediction(self, city, vehicle, distance, emission, response_time):
        """Log prediction request for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'city': city,
            'vehicle': vehicle,
            'distance': distance,
            'emission': emission,
            'response_time_ms': response_time,
            'status': 'success'
        }
        
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            logs.append(log_entry)
            
            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f)
                
        except Exception as e:
            st.error(f"Logging error: {str(e)}")
    
    def get_system_metrics(self):
        """Get system performance metrics"""
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            if not logs:
                return {'total_requests': 0, 'avg_response_time': 0}
            
            df = pd.DataFrame(logs)
            
            metrics = {
                'total_requests': len(df),
                'avg_response_time': df['response_time_ms'].mean(),
                'requests_last_hour': len(df[df['timestamp'] > (datetime.now() - pd.Timedelta(hours=1)).isoformat()]),
                'popular_cities': df['city'].value_counts().to_dict(),
                'popular_vehicles': df['vehicle'].value_counts().to_dict()
            }
            
            return metrics
            
        except Exception as e:
            return {'error': str(e)}
