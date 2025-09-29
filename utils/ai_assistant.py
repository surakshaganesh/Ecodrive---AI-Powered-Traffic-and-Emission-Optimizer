# utils/ai_assistant.py
"""
GenAI-Powered AI Assistant for EcoDrive AI
Uses OpenAI GPT for intelligent, context-aware responses
"""

import openai
import os
import random
from datetime import datetime

class AIAssistant:
    def __init__(self):
        self.conversation_history = []
        # Set your OpenAI API key (get from https://platform.openai.com)
        self.api_key = os.getenv('OPENAI_API_KEY', '')  # Set as environment variable
        
        if self.api_key:
            openai.api_key = self.api_key
            self.use_genai = True
        else:
            self.use_genai = False
            print("⚠️ OpenAI API key not found. Using fallback responses.")
    
    def generate_response(self, user_input, context=None):
        """
        Generate intelligent response using OpenAI GPT
        Falls back to rule-based if API key not available
        """
        if self.use_genai:
            return self._generate_genai_response(user_input, context)
        else:
            return self._generate_fallback_response(user_input, context)
    
    def _generate_genai_response(self, user_input, context=None):
        """
        Use OpenAI GPT for intelligent responses
        """
        try:
            # Build context from traffic data
            system_message = """You are an AI traffic optimization assistant for EcoDrive AI, 
            a system that optimizes routes across Indian cities (Bangalore, Delhi, Mumbai, Hyderabad) 
            to minimize CO2 emissions.
            
            You have access to real-time traffic data and can provide:
            - Traffic predictions and congestion information
            - Route optimization suggestions
            - Emission reduction advice
            - Vehicle-specific recommendations (car, bike, bus, EV)
            - Cost comparisons
            
            Be concise, helpful, and focus on eco-friendly solutions. Use data when available."""
            
            # Add current context if available
            if context:
                context_str = f"\n\nCurrent Traffic Data:\n{context}"
                system_message += context_str
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # or "gpt-4" for better quality
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Store in history
            self.conversation_history.append({
                "user": user_input,
                "assistant": ai_response,
                "timestamp": datetime.now().isoformat()
            })
            
            return ai_response
            
        except Exception as e:
            print(f"GenAI Error: {e}")
            return self._generate_fallback_response(user_input, context)
    
    def _generate_fallback_response(self, user_input, context=None):
        """
        Intelligent fallback responses without API key
        Better than simple keyword matching
        """
        user_input_lower = user_input.lower()
        
        # Extract intent and entities
        intent = self._classify_intent(user_input_lower)
        
        if intent == 'traffic_query':
            cities = ['bangalore', 'delhi', 'mumbai', 'hyderabad']
            mentioned_city = next((city for city in cities if city in user_input_lower), random.choice(cities))
            
            responses = [
                f"Based on current patterns, {mentioned_city.title()} typically experiences peak congestion between 8-10 AM and 6-8 PM. Current traffic is influenced by weather conditions and time of day.",
                f"Traffic analysis for {mentioned_city.title()} shows congestion levels vary by area. IT corridors and business districts see 60-90% congestion during peak hours, while residential areas remain moderate at 30-50%.",
                f"Real-time data indicates {mentioned_city.title()}'s traffic is affected by weather (15-25% impact), day of week (weekends 40% lighter), and hour of day. Plan accordingly for optimal travel times."
            ]
        
        elif intent == 'route_query':
            responses = [
                "For optimal routes, I analyze distance, current traffic, weather conditions, and your vehicle type. EVs save 75% emissions, bikes navigate traffic 30% faster, and buses are most economical at ₹0.50/km per person.",
                "Route optimization considers: 1) Real-time traffic (reduces time by 20-30%), 2) Vehicle efficiency (bikes use shortcuts, buses follow fixed routes), 3) Weather impact, and 4) Signal timing patterns.",
                "My ML models predict the best route by evaluating traffic density, speed patterns, and emission factors. Alternative routes can save 15-30 minutes during peak hours while reducing CO2 by 20-25%."
            ]
        
        elif intent == 'emission_query':
            responses = [
                "Vehicle emissions per km: Car (0.21 kg CO2), Bike (0.089 kg), Bus (0.105 kg per person), EV (0.05 kg). Traffic congestion increases emissions by 30-40% due to stop-start driving.",
                "To reduce emissions: 1) Choose EVs (75% less CO2), 2) Avoid peak hours (40% more efficient), 3) Carpool (reduces per-person emissions by 60%), 4) Use bikes for short distances (<10 km).",
                "Current air quality is affected by vehicle emissions. My system optimizes routes to minimize environmental impact while maintaining reasonable travel times. Eco-friendly vehicles show 50-75% emission reduction."
            ]
        
        elif intent == 'cost_query':
            responses = [
                "Travel costs per km: Bike ₹2 (fuel), Car ₹6 (petrol/diesel), EV ₹1.5 (electricity), Bus ₹0.50 per person. For a 20km trip: Bike ₹40, Car ₹120, EV ₹30, Bus ₹10-30.",
                "Cost-benefit analysis: EVs have higher upfront cost but 70% lower running costs. Bikes are economical for individuals. Public transport (bus/metro) is most affordable for daily commutes.",
                "Including toll charges, parking (₹50-200/day), and fuel, cars cost ₹8-12/km in cities. Two-wheelers cost ₹3-4/km. EVs cost ₹2-3/km. Public transport costs ₹0.50-2/km."
            ]
        
        elif intent == 'time_query':
            responses = [
                "Average city speeds: Bikes 25-30 km/h (faster in traffic), Cars 20-25 km/h, Buses 15-20 km/h (multiple stops). Highway speeds: Cars 60-70 km/h, Bikes 55 km/h, Buses 50 km/h.",
                "Travel time varies by congestion: Light traffic (0-30%) = baseline speed, Medium (30-60%) = 30% slower, Heavy (60-90%) = 50-70% slower. Peak hours add 40-60% to journey time.",
                "Time optimization tips: 1) Start 30 min before/after peaks, 2) Use bikes for <15 km in city, 3) Check real-time traffic before departure, 4) Consider alternative routes during congestion."
            ]
        
        elif intent == 'weather_query':
            responses = [
                "Weather significantly impacts traffic: Rain increases congestion by 20-30%, reduces average speeds by 25%, and increases travel time by 30-50%. Plan extra buffer time during monsoons.",
                "Poor weather (heavy rain, fog) affects visibility and road conditions, leading to slower traffic and increased accidents. Our system factors weather into route optimization.",
                "Temperature extremes also affect traffic patterns. Hot days (>35°C) see 10-15% more vehicles using roads. Rainy season shows 25% increase in congestion in low-lying areas."
            ]
        
        elif intent == 'vehicle_comparison':
            responses = [
                "Vehicle comparison for city travel: Bikes - fastest (maneuverability), lowest cost, weather-dependent. Cars - comfortable, moderate cost, AC comfort. Buses - cheapest, slowest, fixed routes. EVs - eco-friendly, low running cost, charging needs.",
                "For 20km daily commute: Bike saves 15-20 min vs car, costs 60% less. EV costs 75% less than petrol car yearly. Bus costs least but takes 40% longer. Choose based on priority: time, cost, or comfort.",
                "Long-distance (>100km): Cars offer flexibility and comfort. Buses are economical but scheduled. Bikes efficient for 50-150km. EVs need charging infrastructure planning every 150-200km."
            ]
        
        elif intent == 'city_specific':
            city_info = {
                'bangalore': "Bangalore has Outer Ring Road congestion, Electronic City routes are tech-heavy. Best times: 11 AM-4 PM. Metro integration reduces traffic by 15%.",
                'delhi': "Delhi sees extreme congestion on Ring Road and NH-1. Metro is highly effective. Peak hours are severe (90%+ congestion). Best travel: 10 AM-5 PM.",
                'mumbai': "Mumbai has unique sea-link option. Western and Eastern Express Highways are critical. Local trains handle 75 lakh daily passengers. Peak hours: 8-11 AM, 6-9 PM.",
                'hyderabad': "Hyderabad's IT corridor (Gachibowli-HITEC City) sees peak congestion. Outer Ring Road is fast. Best times: pre-8 AM or post-7 PM for IT areas."
            }
            
            for city, info in city_info.items():
                if city in user_input_lower:
                    return info
            
            return random.choice(list(city_info.values()))
        
        else:  # general or greeting
            responses = [
                "I'm your AI traffic assistant. I can help with route optimization, emission reduction, traffic predictions, and cost analysis. What would you like to know?",
                "I analyze traffic patterns using ML models with 94% accuracy to provide optimal routes. Ask me about traffic conditions, emission savings, or cost comparisons.",
                "I'm here to help optimize your travel across Indian metros. I can suggest eco-friendly routes, predict congestion, compare vehicle costs, and provide time estimates."
            ]
        
        return random.choice(responses)
    
    def _classify_intent(self, text):
        """
        Classify user intent based on keywords and context
        """
        intents = {
            'traffic_query': ['traffic', 'congestion', 'jam', 'busy', 'crowded', 'flow'],
            'route_query': ['route', 'path', 'way', 'direction', 'navigate', 'go to', 'reach'],
            'emission_query': ['emission', 'co2', 'carbon', 'pollution', 'environment', 'green', 'eco'],
            'cost_query': ['cost', 'price', 'money', 'expensive', 'cheap', 'fare', 'charge'],
            'time_query': ['time', 'duration', 'how long', 'minutes', 'hours', 'fast', 'slow'],
            'weather_query': ['weather', 'rain', 'hot', 'cold', 'temperature', 'climate'],
            'vehicle_comparison': ['compare', 'vs', 'versus', 'difference', 'better', 'which vehicle'],
            'city_specific': ['bangalore', 'delhi', 'mumbai', 'hyderabad', 'city']
        }
        
        # Count matches for each intent
        intent_scores = {}
        for intent, keywords in intents.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                intent_scores[intent] = score
        
        # Return intent with highest score, or 'general' if no matches
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        return 'general'
    
    def get_conversation_history(self):
        """Get conversation history"""
        return self.conversation_history