import numpy as np
from datetime import datetime
from geopy.distance import geodesic
import requests

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

    # ==============================
    # Helpers
    # ==============================

    def get_route_coordinates(self, start_lat, start_lon, end_lat, end_lon):
        """Fetch actual driving route using OSRM API"""
        try:
            url = f"https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
            response = requests.get(url)
            data = response.json()
            coords = data['routes'][0]['geometry']['coordinates']
            return [(lat, lon) for lon, lat in coords]  # flip (lon, lat) → (lat, lon)
        except Exception:
            # fallback straight line
            return [(start_lat, start_lon), (end_lat, end_lon)]

    def calculate_distance(self, start_lat, start_lon, end_lat, end_lon):
        """Consistent distance using geopy"""
        return round(geodesic((start_lat, start_lon), (end_lat, end_lon)).km, 1)

    def calculate_route_emissions(self, distance, vehicle_type, traffic_factor=1.0):
        """Calculate CO2 emissions"""
        base_emission = distance * self.vehicle_emissions.get(vehicle_type, 0.21)
        adjusted_emission = base_emission * (1 + (traffic_factor - 1) * 0.3)
        return round(adjusted_emission, 2)

    def estimate_travel_time(self, distance, traffic_factor, highway=False):
        """Estimate travel time based on speed and traffic"""
        if highway:
            base_speed = 60  # highway speed in km/h
        else:
            base_speed = 25  # city speed in km/h

        adjusted_speed = base_speed / traffic_factor
        time_hours = distance / adjusted_speed
        return max(int(time_hours * 60), 5)

    # ==============================
    # Main Optimizer
    # ==============================

    def optimize_route(self, start_coords, end_coords, city, vehicle_type, current_traffic, inter_city=False):
        """
        Optimize route between start and end.
        start_coords = (lat, lon)
        end_coords   = (lat, lon)
        """

        # Compute distance using geodesic
        distance = self.calculate_distance(start_coords[0], start_coords[1], end_coords[0], end_coords[1])

        # Traffic factor (scaled 0.5 to 1.5)
        traffic_factor = current_traffic / 100 + 0.5

        # Pick a representative route
        if inter_city:
            optimal_route = "Highway Route"
        else:
            optimal_route = np.random.choice(self.city_routes.get(city, ['Main Road']))

        # Emissions
        optimal_emission = self.calculate_route_emissions(distance, vehicle_type, traffic_factor * 0.85)
        alternative_emission = self.calculate_route_emissions(distance * 1.2, vehicle_type, traffic_factor)
        savings = round(alternative_emission - optimal_emission, 2)

        # Travel time
        travel_time = self.estimate_travel_time(distance, traffic_factor, highway=inter_city)

        # Route coordinates for visualization
        route_coords = self.get_route_coordinates(
            start_coords[0], start_coords[1],
            end_coords[0], end_coords[1]
        )

        return {
            'route': optimal_route,
            'distance': distance,
            'emission': optimal_emission,
            'alternative_emission': alternative_emission,
            'savings': savings,
            'savings_percentage': round((savings / alternative_emission) * 100, 1) if alternative_emission > 0 else 0,
            'travel_time': travel_time,
            'coords': route_coords
        }
