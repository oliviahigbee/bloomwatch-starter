from flask import Flask, render_template, request, jsonify
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# NASA API configuration
NASA_API_KEY = os.getenv('NASA_API_KEY', 'DEMO_KEY')
LANDSAT_API_URL = "https://api.nasa.gov/planetary/earth/assets"
MODIS_API_URL = "https://api.nasa.gov/planetary/earth/assets"

class BloomMonitor:
    def __init__(self):
        self.vegetation_indices = {
            'NDVI': self.calculate_ndvi,
            'EVI': self.calculate_evi,
            'SAVI': self.calculate_savi,
            'GNDVI': self.calculate_gndvi
        }
    
    def calculate_ndvi(self, red, nir):
        """Calculate Normalized Difference Vegetation Index"""
        return (nir - red) / (nir + red + 1e-8)
    
    def calculate_evi(self, red, nir, blue):
        """Calculate Enhanced Vegetation Index"""
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    
    def calculate_savi(self, red, nir, l=0.5):
        """Calculate Soil Adjusted Vegetation Index"""
        return ((nir - red) / (nir + red + l)) * (1 + l)
    
    def calculate_gndvi(self, green, nir):
        """Calculate Green Normalized Difference Vegetation Index"""
        return (nir - green) / (nir + green + 1e-8)
    
    def detect_bloom_events(self, vegetation_data, threshold=0.3):
        """Detect bloom events based on vegetation index thresholds"""
        bloom_events = []
        for idx, value in enumerate(vegetation_data):
            if value > threshold:
                bloom_events.append({
                    'index': idx,
                    'value': value,
                    'intensity': 'high' if value > 0.6 else 'medium' if value > 0.4 else 'low'
                })
        return bloom_events
    
    def analyze_temporal_trends(self, time_series_data):
        """Analyze temporal trends in bloom patterns"""
        if len(time_series_data) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate trend
        x = np.arange(len(time_series_data))
        y = np.array(time_series_data)
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            trend = 'increasing'
        elif slope < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': slope,
            'mean_value': np.mean(y),
            'std_value': np.std(y),
            'peak_season': self.identify_peak_season(time_series_data)
        }
    
    def identify_peak_season(self, time_series_data):
        """Identify peak blooming season"""
        if len(time_series_data) < 12:  # Need at least a year of data
            return 'insufficient_data'
        
        # Find month with highest average vegetation index
        monthly_avg = []
        for month in range(12):
            month_values = [time_series_data[i] for i in range(month, len(time_series_data), 12)]
            if month_values:
                monthly_avg.append(np.mean(month_values))
            else:
                monthly_avg.append(0)
        
        peak_month = np.argmax(monthly_avg)
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        return seasons[peak_month // 3]

# Initialize bloom monitor
bloom_monitor = BloomMonitor()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/bloom-data')
def get_bloom_data():
    """Get bloom data for a specific location and time range"""
    lat = request.args.get('lat', 40.7128)  # Default to NYC
    lon = request.args.get('lon', -74.0060)
    start_date = request.args.get('start_date', '2023-01-01')
    end_date = request.args.get('end_date', '2023-12-31')
    
    # Simulate NASA data retrieval (in real implementation, this would call NASA APIs)
    bloom_data = simulate_nasa_data(lat, lon, start_date, end_date)
    
    return jsonify(bloom_data)

@app.route('/api/cities')
def get_cities():
    """Get list of available cities for bloom monitoring"""
    cities = [
        {
            'name': 'New York City',
            'country': 'United States',
            'continent': 'North America',
            'lat': 40.7128,
            'lon': -74.0060,
            'description': 'Major metropolitan area with diverse urban vegetation'
        },
        {
            'name': 'London',
            'country': 'United Kingdom',
            'continent': 'Europe',
            'lat': 51.5074,
            'lon': -0.1278,
            'description': 'Historic city with extensive park systems'
        },
        {
            'name': 'Tokyo',
            'country': 'Japan',
            'continent': 'Asia',
            'lat': 35.6762,
            'lon': 139.6503,
            'description': 'Mega-city with seasonal cherry blossom monitoring'
        },
        {
            'name': 'São Paulo',
            'country': 'Brazil',
            'continent': 'South America',
            'lat': -23.5505,
            'lon': -46.6333,
            'description': 'Tropical urban environment with rich biodiversity'
        },
        {
            'name': 'Sydney',
            'country': 'Australia',
            'continent': 'Oceania',
            'lat': -33.8688,
            'lon': 151.2093,
            'description': 'Coastal city with unique Australian flora'
        },
        {
            'name': 'Cape Town',
            'country': 'South Africa',
            'continent': 'Africa',
            'lat': -33.9249,
            'lon': 18.4241,
            'description': 'Mediterranean climate with fynbos vegetation'
        },
        {
            'name': 'Mumbai',
            'country': 'India',
            'continent': 'Asia',
            'lat': 19.0760,
            'lon': 72.8777,
            'description': 'Tropical coastal city with monsoon bloom patterns'
        },
        {
            'name': 'Paris',
            'country': 'France',
            'continent': 'Europe',
            'lat': 48.8566,
            'lon': 2.3522,
            'description': 'Temperate climate with seasonal garden blooms'
        },
        {
            'name': 'Los Angeles',
            'country': 'United States',
            'continent': 'North America',
            'lat': 34.0522,
            'lon': -118.2437,
            'description': 'Mediterranean climate with year-round vegetation'
        },
        {
            'name': 'Buenos Aires',
            'country': 'Argentina',
            'continent': 'South America',
            'lat': -34.6118,
            'lon': -58.3960,
            'description': 'Subtropical climate with diverse urban parks'
        },
        {
            'name': 'Cairo',
            'country': 'Egypt',
            'continent': 'Africa',
            'lat': 30.0444,
            'lon': 31.2357,
            'description': 'Desert climate with Nile delta vegetation'
        },
        {
            'name': 'Moscow',
            'country': 'Russia',
            'continent': 'Europe',
            'lat': 55.7558,
            'lon': 37.6176,
            'description': 'Continental climate with distinct seasonal patterns'
        }
    ]
    
    return jsonify({
        'cities': cities,
        'total': len(cities),
        'description': 'Global cities for plant bloom monitoring using NASA Earth observation data'
    })

@app.route('/api/global-bloom-map')
def get_global_bloom_map():
    """Get global bloom map data"""
    # Simulate global bloom data
    global_data = simulate_global_bloom_data()
    return jsonify(global_data)

@app.route('/api/trends')
def get_trends():
    """Get temporal trends analysis"""
    location = request.args.get('location', 'global')
    years = int(request.args.get('years', 5))
    
    # Simulate trend analysis
    trends = simulate_trend_analysis(location, years)
    return jsonify(trends)

@app.route('/api/conservation-insights', methods=['GET', 'POST'])
def get_conservation_insights():
    """Get conservation insights and recommendations"""
    location = request.args.get('location', 'global')
    bloom_data = request.get_json() if request.method == 'POST' else {}
    
    insights = generate_conservation_insights(location, bloom_data)
    return jsonify(insights)

@lru_cache(maxsize=10)
def simulate_nasa_data(lat, lon, start_date, end_date):
    """Simulate NASA satellite data retrieval - ultra fast"""
    # Minimal data points for instant loading
    dates = pd.date_range(start=start_date, end=end_date, freq='90D')  # Quarterly data only
    
    # Pre-calculated seasonal pattern
    base_intensity = 0.5
    seasonal_variation = 0.3
    
    # Create minimal data set
    bloom_data = []
    for i, date in enumerate(dates):
        # Simple seasonal pattern
        intensity = base_intensity + seasonal_variation * np.sin(2 * np.pi * i / len(dates))
        intensity = max(0.1, min(0.9, intensity))  # Clamp values
        
        bloom_data.append({
            'date': date.isoformat(),
            'ndvi': round(intensity, 3),
            'evi': round(intensity * 0.9, 3),
            'savi': round(intensity * 1.1, 3),
            'gndvi': round(intensity * 0.8, 3),
            'latitude': round(float(lat), 2),
            'longitude': round(float(lon), 2),
            'bloom_probability': round(min(1.0, intensity * 1.5), 3)
        })
    
    return {
        'location': {'lat': lat, 'lon': lon},
        'time_range': {'start': start_date, 'end': end_date},
        'data': bloom_data,
        'summary': {
            'total_observations': len(bloom_data),
            'avg_bloom_intensity': round(np.mean([d['ndvi'] for d in bloom_data]), 3),
            'peak_bloom_date': bloom_data[0]['date']  # Simplified
        }
    }

@lru_cache(maxsize=1)
def simulate_global_bloom_map():
    """Simulate global bloom map data - ultra fast minimal data"""
    # Ultra minimal grid for instant loading
    lats = np.linspace(-60, 60, 8)  # Very small grid
    lons = np.linspace(-180, 180, 16)  # Very small grid
    
    # Simple pattern generation
    global_data = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # Simple intensity based on latitude
            intensity = 0.3 + 0.4 * (lat + 60) / 120  # 0.3 to 0.7 range
            intensity = max(0.1, min(0.9, intensity))
            
            global_data.append({
                'lat': round(float(lat), 1),
                'lon': round(float(lon), 1),
                'bloom_intensity': round(intensity, 3),
                'bloom_status': 'active' if intensity > 0.4 else 'dormant'
            })
    
    return {
        'data': global_data,
        'timestamp': datetime.now().isoformat(),
        'total_locations': len(global_data)
    }

def simulate_global_bloom_data():
    """Alias for simulate_global_bloom_map for compatibility"""
    return simulate_global_bloom_map()

@lru_cache(maxsize=10)
def simulate_trend_analysis(location, years):
    """Simulate temporal trend analysis - ultra fast"""
    # Simple trend simulation
    trend_types = ['increasing', 'decreasing', 'stable']
    trend = trend_types[hash(location) % 3]
    
    # Simple yearly data
    yearly_data = {}
    for i in range(years):
        year = 2020 + i
        year_data = [0.3 + 0.2 * np.sin(2 * np.pi * month / 12) for month in range(1, 13)]
        yearly_data[str(year)] = [round(val, 3) for val in year_data]
    
    return {
        'location': location,
        'years_analyzed': years,
        'trends': {
            'trend': trend,
            'slope': 0.01 if trend == 'increasing' else -0.01 if trend == 'decreasing' else 0,
            'mean_value': 0.5,
            'std_value': 0.1,
            'peak_season': 'Spring'
        },
        'yearly_data': yearly_data,
        'recommendations': [
            f"Trend analysis shows {trend} bloom patterns",
            "Continue monitoring for conservation insights",
            "Consider seasonal management strategies"
        ]
    }

@lru_cache(maxsize=10)
def generate_conservation_insights(location, bloom_data):
    """Generate conservation insights and recommendations - ultra fast"""
    # Simple location-based insights
    priority_levels = ['low', 'medium', 'high']
    priority = priority_levels[hash(location) % 3]
    
    return {
        'location': location,
        'insights': [
            "NASA satellite data shows active vegetation monitoring",
            "Bloom patterns indicate seasonal ecosystem health",
            "Multi-year trends provide conservation insights"
        ],
        'recommendations': [
            "Continue monitoring with NASA Earth observation data",
            "Coordinate with local conservation organizations",
            "Track seasonal bloom variations for management planning",
            "Consider pollinator-friendly habitat enhancement"
        ],
        'conservation_priority': priority
    }

def generate_trend_recommendations(trends):
    """Generate recommendations based on trend analysis"""
    recommendations = []
    
    if trends['trend'] == 'increasing':
        recommendations.append("Positive trend detected - continue current management practices")
    elif trends['trend'] == 'decreasing':
        recommendations.append("Declining trend - investigate causes and implement conservation measures")
    else:
        recommendations.append("Stable trend - maintain monitoring and consider enhancement opportunities")
    
    if trends.get('peak_season') != 'insufficient_data':
        recommendations.append(f"Peak bloom season: {trends['peak_season']} - plan conservation activities accordingly")
    
    return recommendations

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)