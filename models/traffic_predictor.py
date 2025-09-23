import numpy as np
import random
from datetime import datetime

class TrafficPredictor:
    def __init__(self):
        pass

    def predict_traffic(self, city, hour=None, day_of_week=None):
        """
        Simulate traffic prediction for a given city and time.
        """
        hour = hour or datetime.now().hour
        day_of_week = day_of_week or datetime.now().weekday()

        is_peak = (8 <= hour <= 10) or (18 <= hour <= 20)
        is_weekend = day_of_week >= 5

        base_congestion = random.randint(20, 60)
        if is_peak and not is_weekend:
            congestion = random.randint(60, 95)
        elif is_weekend:
            congestion = random.randint(20, 50)
        else:
            congestion = base_congestion

        avg_speed = max(10, 50 - (congestion * 0.4))
        vehicles_count = random.randint(5000, 25000)
        co2_level = random.randint(80, 200)

        return {
            'city': city,
            'congestion': congestion,
            'avg_speed': avg_speed,
            'vehicles_count': vehicles_count,
            'co2_level': co2_level,
            'timestamp': datetime.now().isoformat()
        }
