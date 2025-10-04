""" Here's everything we should need to use SerpAPI to search for local plants.
    This allows a user to enter a city name and then searches for local plants.
    I put the code here until we figure out how we want to implement it.
    The imports already exist in app.py. 
    We can make changes as needed. """

# Already in app.py
from flask import Flask, render_template, request, jsonify
import requests

# Already in app.py
app = Flask(__name__)

# SerpAPI key and base URL
API_KEY = '5eb5f47bd48d32d2a2af5b36799fbb7b4c782b833d6c1c9d9048be5c9afd9066'
SERPAPI_URL = 'https://serpapi.com/search'

# Function to get local plants for a given city
def get_local_plants(city_name):
    query = f"local plants in {city_name}"
    params = {
        'q': query,
        'location': city_name,
        'api_key': API_KEY,
        'engine': 'google'
    }

    response = requests.get(SERPAPI_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('organic_results', []) # organic_results returns non-paid results
    else:
        return []

@app.route('/')
def index():
    return render_template('index.html')

# Flask route that accepts a city as a query parameter.
# The frontend HTML will make a request to this route and display the results.
@app.route('/get-plants', methods=['GET'])
def get_plants():
    city = request.args.get('city')
    if city:
        plants = get_local_plants(city)
        return jsonify(plants)
    else:
        return jsonify({"error": "City is required"}), 400

# This exists in app.py
if __name__ == '__main__':
    app.run(debug=True)
