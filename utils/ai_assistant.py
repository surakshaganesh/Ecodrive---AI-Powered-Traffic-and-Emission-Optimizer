import random
import openai

class AIAssistant:
    def __init__(self, api_key=None):
        self.api_key = api_key
        if api_key:
            openai.api_key = api_key

        self.fallback_responses = [
            "Based on current traffic patterns, plan your trip during off-peak hours for better efficiency.",
            "EVs can reduce your carbon footprint by up to 75% in urban traffic.",
            "Carpooling or using buses can cut emissions by nearly 50%.",
            "Heavy congestion increases emissions. Consider eco-optimized routes.",
        ]

    def get_response(self, user_message, context_data=None):
        """Get AI response"""
        if self.api_key:
            return self.get_openai_response(user_message, context_data)
        else:
            return self.get_fallback_response(user_message)

    def get_openai_response(self, user_message, context_data):
        try:
            context = f"Context: {context_data}" if context_data else ""
            prompt = f"""
            You are an AI traffic and emission optimization assistant for EcoDrive AI.

            {context}
            User: {user_message}

            Give specific, short advice (under 80 words) about:
            - Traffic optimization
            - Route recommendations
            - Emission reduction
            - Eco-friendly transport
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except:
            return self.get_fallback_response(user_message)

    def get_fallback_response(self, user_message):
        return random.choice(self.fallback_responses)

    def generate_route_explanation(self, route_data):
        """Explain why a route was chosen"""
        return f"This route via {route_data.get('route', 'main road')} reduces emissions by {route_data.get('savings', 0)}kg CO₂."
