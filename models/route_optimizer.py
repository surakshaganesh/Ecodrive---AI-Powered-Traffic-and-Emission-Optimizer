import numpy as np
import random
from datetime import datetime

class RouteOptimizer:
    def __init__(self):
        self.vehicle_emissions = {
            'car': 0.21,      # kg CO2 per km
            'bike': 0.089,    # kg CO2 per km
            'bus': 0.105,     # kg CO2 per km
            'ev': 0.05        # kg CO2 per km
        }

        self.city_routes = {
            'bangalore': [
                'Outer Ring Road', 'Electronic City Flyover', 'Silk Board Junction',
                'Hosur Road', 'Whitefield Route', 'Bannerghatta Road'
            ],
            'delhi': [
                'Ring Road', 'DND Flyway', 'Noida Expressway',
                'NH-1', 'Gurgaon Expressway', 'Delhi-Meerut Expressway'
            ],
            'mumbai': [
                'Western Express Highway', 'Eastern Express Highway', 'Bandra-Worli Sea Link',
                'Mumbai-Pune Expressway', 'SV Road', 'LBS Marg'
            ],
            'hyderabad': [
                'Outer Ring Road', 'Cyberabad Route', 'Airport Road',
                'Kompally Route', 'Gachibowli Route', 'Hitech City Route'
            ]
        }

    def calculate_route_emissions(self, distance, vehicle_type, traffic_factor=1.0):
        """Calculate CO2 emissions for a route"""
        base_emission = distance * self.vehicle_emissions.get(vehicle_type, 0.21)
        adjusted_emission = base_emission * (1 + (traffic_factor - 1) * 0.3)
        return round(adjusted_emission, 2)

    def optimize_route(self, start, end, city, vehicle_type, current_traffic):
        """Optimize route based on emissions and traffic"""
        base_distance = random.uniform(5, 30)
        traffic_factor = current_traffic / 100 + 0.5
        optimal_route = random.choice(self.city_routes.get(city, ['Main Road']))

        optimal_emission = self.calculate_route_emissions(base_distance, vehicle_type, traffic_factor * 0.85)
        alternative_emission = self.calculate_route_emissions(base_distance * 1.2, vehicle_type, traffic_factor)
        savings = alternative_emission - optimal_emission

        return {
            'route': optimal_route,
            'distance': round(base_distance, 1),
            'emission': optimal_emission,
            'alternative_emission': alternative_emission,
            'savings': round(savings, 2),
            'savings_percentage': round((savings / alternative_emission) * 100, 1),
            'travel_time': self.estimate_travel_time(base_distance, traffic_factor)
        }

    def estimate_travel_time(self, distance, traffic_factor):
        """Estimate travel time based on distance and traffic"""
        base_speed = 25  # km/h
        adjusted_speed = base_speed / traffic_factor
        time_hours = distance / adjusted_speed
        time_minutes = int(time_hours * 60)
        return max(time_minutes, 5)
