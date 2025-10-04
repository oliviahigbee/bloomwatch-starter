from flask import Flask, render_template, request, jsonify
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
from functools import lru_cache
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# NASA API configuration
NASA_API_KEY = os.getenv('NASA_API_KEY', 'DEMO_KEY')
LANDSAT_API_URL = "https://api.nasa.gov/planetary/earth/assets"
MODIS_API_URL = "https://api.nasa.gov/planetary/earth/assets"
EARTHDATA_API_URL = "https://cmr.earthdata.nasa.gov/search"
CLIMATE_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

class BloomPredictor:
    """AI-powered bloom prediction and anomaly detection"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.anomaly_threshold = 0.15
        
    def train_model(self, historical_data):
        """Train ML model on historical bloom data"""
        try:
            # Prepare training data
            X, y = self._prepare_training_data(historical_data)
            
            if len(X) < 10:  # Need minimum data for training
                return False
                
            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Model training failed: {e}")
            return False
    
    def _prepare_training_data(self, data):
        """Prepare features and targets for ML training"""
        features = []
        targets = []
        
        for i in range(1, len(data)):
            if i >= 12:  # Need at least a year of data
                # Features: previous 12 months of data + seasonal features
                prev_12_months = [d['ndvi'] for d in data[i-12:i]]
                month = i % 12
                season = month // 3
                
                feature_vector = prev_12_months + [month, season, np.mean(prev_12_months), np.std(prev_12_months)]
                features.append(feature_vector)
                targets.append(data[i]['ndvi'])
        
        return np.array(features), np.array(targets)
    
    def predict_bloom(self, recent_data, days_ahead=30):
        """Predict future bloom intensity"""
        if not self.is_trained or len(recent_data) < 12:
            return self._fallback_prediction(recent_data, days_ahead)
        
        try:
            # Use last 12 months for prediction
            last_12_months = [d['ndvi'] for d in recent_data[-12:]]
            month = len(recent_data) % 12
            season = month // 3
            
            feature_vector = last_12_months + [month, season, np.mean(last_12_months), np.std(last_12_months)]
            X = np.array([feature_vector])
            X_scaled = self.scaler.transform(X)
            
            prediction = self.model.predict(X_scaled)[0]
            confidence = min(0.95, max(0.1, 1.0 - np.std(last_12_months)))
            
            return {
                'predicted_intensity': float(prediction),
                'confidence': float(confidence),
                'days_ahead': days_ahead,
                'model_used': 'RandomForest'
            }
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            return self._fallback_prediction(recent_data, days_ahead)
    
    def _fallback_prediction(self, recent_data, days_ahead):
        """Fallback prediction using simple trend analysis"""
        if len(recent_data) < 3:
            return {'predicted_intensity': 0.5, 'confidence': 0.1, 'days_ahead': days_ahead, 'model_used': 'Fallback'}
        
        recent_values = [d['ndvi'] for d in recent_data[-3:]]
        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        predicted = recent_values[-1] + trend * (days_ahead / 30)
        
        return {
            'predicted_intensity': float(max(0, min(1, predicted))),
            'confidence': 0.3,
            'days_ahead': days_ahead,
            'model_used': 'Trend'
        }
    
    def detect_anomalies(self, data):
        """Detect anomalous bloom patterns"""
        if len(data) < 12:
            return []
        
        anomalies = []
        values = [d['ndvi'] for d in data]
        
        # Calculate rolling statistics
        window = 12
        for i in range(window, len(values)):
            window_data = values[i-window:i]
            mean_val = np.mean(window_data)
            std_val = np.std(window_data)
            
            current_val = values[i]
            z_score = abs(current_val - mean_val) / (std_val + 1e-8)
            
            if z_score > 2.0:  # Statistical anomaly
                anomalies.append({
                    'date': data[i]['date'],
                    'value': current_val,
                    'expected_range': [mean_val - 2*std_val, mean_val + 2*std_val],
                    'anomaly_score': float(z_score),
                    'type': 'high' if current_val > mean_val else 'low'
                })
        
        return anomalies

class ClimateAnalyzer:
    """Analyze climate data correlation with bloom patterns"""
    
    def __init__(self):
        self.climate_cache = {}
    
    def get_climate_data(self, lat, lon, start_date, end_date):
        """Fetch climate data from NASA POWER API"""
        cache_key = f"{lat}_{lon}_{start_date}_{end_date}"
        if cache_key in self.climate_cache:
            return self.climate_cache[cache_key]
        
        try:
            # NASA POWER API parameters
            params = {
                'parameters': 'T2M,PRECTOT,ALLSKY_SFC_SW_DWN',
                'community': 'RE',
                'longitude': lon,
                'latitude': lat,
                'start': start_date,
                'end': end_date,
                'format': 'JSON'
            }
            
            response = requests.get(CLIMATE_API_URL, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.climate_cache[cache_key] = data
                return data
        except Exception as e:
            print(f"Climate data fetch failed: {e}")
        
        return None
    
    def correlate_climate_bloom(self, climate_data, bloom_data):
        """Correlate climate variables with bloom patterns"""
        if not climate_data or not bloom_data:
            return {}
        
        try:
            # Extract climate variables
            temp_data = climate_data.get('properties', {}).get('parameter', {}).get('T2M', {})
            precip_data = climate_data.get('properties', {}).get('parameter', {}).get('PRECTOT', {})
            solar_data = climate_data.get('properties', {}).get('parameter', {}).get('ALLSKY_SFC_SW_DWN', {})
            
            # Calculate correlations
            correlations = {
                'temperature': self._calculate_correlation(temp_data, bloom_data),
                'precipitation': self._calculate_correlation(precip_data, bloom_data),
                'solar_radiation': self._calculate_correlation(solar_data, bloom_data)
            }
            
            return correlations
        except Exception as e:
            print(f"Climate correlation failed: {e}")
            return {}
    
    def _calculate_correlation(self, climate_series, bloom_data):
        """Calculate correlation between climate and bloom data"""
        if not climate_series or not bloom_data:
            return 0.0
        
        try:
            # Align dates and calculate correlation
            climate_values = []
            bloom_values = []
            
            for bloom_point in bloom_data:
                date_str = bloom_point['date'][:10]  # YYYY-MM-DD
                if date_str in climate_series:
                    climate_values.append(climate_series[date_str])
                    bloom_values.append(bloom_point['ndvi'])
            
            if len(climate_values) > 3:
                correlation = np.corrcoef(climate_values, bloom_values)[0, 1]
                return float(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            print(f"Correlation calculation failed: {e}")
        
        return 0.0

class BloomMonitor:
    def __init__(self):
        self.vegetation_indices = {
            'NDVI': self.calculate_ndvi,
            'EVI': self.calculate_evi,
            'SAVI': self.calculate_savi,
            'GNDVI': self.calculate_gndvi
        }
        self.predictor = BloomPredictor()
        self.climate_analyzer = ClimateAnalyzer()
    
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
    
    # Convert bloom_data to a hashable string for caching
    bloom_data_str = str(sorted(bloom_data.items())) if bloom_data else 'empty'
    insights = generate_conservation_insights(location, bloom_data_str)
    return jsonify(insights)

@app.route('/api/predict-bloom', methods=['POST'])
def predict_bloom():
    """AI-powered bloom prediction endpoint"""
    try:
        data = request.get_json()
        location = data.get('location', 'global')
        days_ahead = data.get('days_ahead', 30)
        
        # Get historical data for the location
        if location != 'global':
            # Extract coordinates from location or use default
            lat, lon = 40.7128, -74.0060  # Default to NYC
            if ',' in location:
                try:
                    coords = location.split(',')
                    lat, lon = float(coords[0]), float(coords[1])
                except:
                    pass
        else:
            lat, lon = 20.0, 0.0  # Global center
        
        # Get historical data
        historical_data = simulate_nasa_data(lat, lon, '2020-01-01', '2024-12-31')
        
        # Train model and predict
        bloom_monitor.predictor.train_model(historical_data['data'])
        prediction = bloom_monitor.predictor.predict_bloom(historical_data['data'], days_ahead)
        
        return jsonify({
            'location': location,
            'prediction': prediction,
            'model_status': 'trained' if bloom_monitor.predictor.is_trained else 'fallback',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect-anomalies', methods=['POST'])
def detect_anomalies():
    """Detect anomalous bloom patterns"""
    try:
        data = request.get_json()
        location = data.get('location', 'global')
        
        # Get historical data
        if location != 'global':
            lat, lon = 40.7128, -74.0060
        else:
            lat, lon = 20.0, 0.0
            
        historical_data = simulate_nasa_data(lat, lon, '2020-01-01', '2024-12-31')
        
        # Detect anomalies
        anomalies = bloom_monitor.predictor.detect_anomalies(historical_data['data'])
        
        return jsonify({
            'location': location,
            'anomalies': anomalies,
            'total_anomalies': len(anomalies),
            'analysis_period': '2020-2024',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/climate-correlation', methods=['POST'])
def get_climate_correlation():
    """Get climate data correlation with bloom patterns"""
    try:
        data = request.get_json()
        lat = float(data.get('lat', 40.7128))
        lon = float(data.get('lon', -74.0060))
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2023-12-31')
        
        # Get climate data
        climate_data = bloom_monitor.climate_analyzer.get_climate_data(lat, lon, start_date, end_date)
        
        # Get bloom data
        bloom_data = simulate_nasa_data(lat, lon, start_date, end_date)
        
        # Calculate correlations
        correlations = bloom_monitor.climate_analyzer.correlate_climate_bloom(
            climate_data, bloom_data['data']
        )
        
        return jsonify({
            'location': {'lat': lat, 'lon': lon},
            'time_range': {'start': start_date, 'end': end_date},
            'correlations': correlations,
            'climate_data_available': climate_data is not None,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/citizen-science', methods=['GET', 'POST'])
def citizen_science():
    """Citizen science data collection endpoint"""
    if request.method == 'GET':
        # Return recent citizen science observations
        observations = [
            {
                'id': 1,
                'location': {'lat': 40.7128, 'lon': -74.0060},
                'species': 'Cherry Blossom',
                'bloom_status': 'full_bloom',
                'observer': 'Citizen Scientist',
                'date': '2024-04-15',
                'confidence': 0.9,
                'photo_url': '/static/images/cherry_blossom_sample.jpg'
            },
            {
                'id': 2,
                'location': {'lat': 51.5074, 'lon': -0.1278},
                'species': 'Daffodil',
                'bloom_status': 'early_bloom',
                'observer': 'Nature Enthusiast',
                'date': '2024-03-20',
                'confidence': 0.8,
                'photo_url': '/static/images/daffodil_sample.jpg'
            }
        ]
        
        return jsonify({
            'observations': observations,
            'total_observations': len(observations),
            'contribution_message': 'Help us track blooms worldwide! Submit your observations.'
        })
    
    elif request.method == 'POST':
        # Accept new citizen science observations
        data = request.get_json()
        
        # Validate and store observation (in real implementation, save to database)
        observation = {
            'id': len(data) + 1,  # Simple ID generation
            'location': data.get('location'),
            'species': data.get('species'),
            'bloom_status': data.get('bloom_status'),
            'observer': data.get('observer', 'Anonymous'),
            'date': datetime.now().isoformat(),
            'confidence': data.get('confidence', 0.5),
            'notes': data.get('notes', '')
        }
        
        return jsonify({
            'message': 'Observation recorded successfully!',
            'observation': observation,
            'contribution_points': 10
        })

@app.route('/api/3d-globe-data')
def get_3d_globe_data():
    """Get data for 3D globe visualization"""
    try:
        # Generate 3D globe data with time-lapse capability
        globe_data = []
        
        # Create a grid of points around the globe
        for lat in range(-60, 61, 10):
            for lon in range(-180, 181, 20):
                # Simulate bloom intensity based on latitude and season
                seasonal_factor = 0.5 + 0.3 * np.sin(2 * np.pi * lat / 180)
                bloom_intensity = max(0.1, min(0.9, seasonal_factor + 0.1 * np.random.random()))
                
                globe_data.append({
                    'lat': lat,
                    'lon': lon,
                    'bloom_intensity': round(bloom_intensity, 3),
                    'elevation': 0,  # Sea level
                    'timestamp': datetime.now().isoformat()
                })
        
        return jsonify({
            'globe_data': globe_data,
            'total_points': len(globe_data),
            'time_lapse_available': True,
            'animation_speed': '1_day_per_second'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@lru_cache(maxsize=10)
def simulate_nasa_data(lat, lon, start_date, end_date):
    """Simulate NASA satellite data retrieval - ultra fast with location-specific patterns"""
    # Minimal data points for instant loading
    dates = pd.date_range(start=start_date, end=end_date, freq='90D')  # Quarterly data only
    
    # Location-specific base intensity based on latitude and longitude
    # Higher latitudes (colder) have lower base intensity, tropical regions have higher
    lat_factor = 1.0 - abs(float(lat)) / 90.0  # 0 at poles, 1 at equator
    lon_factor = 0.8 + 0.4 * np.sin(float(lon) * np.pi / 180)  # Vary by longitude
    
    base_intensity = 0.3 + 0.4 * lat_factor * lon_factor
    seasonal_variation = 0.2 + 0.1 * lat_factor  # More seasonal variation at higher latitudes
    
    # Create minimal data set with location-specific patterns
    bloom_data = []
    for i, date in enumerate(dates):
        # Seasonal pattern varies by latitude (hemisphere effect)
        hemisphere_factor = 1.0 if float(lat) >= 0 else -1.0
        seasonal_phase = hemisphere_factor * 2 * np.pi * i / len(dates)
        
        intensity = base_intensity + seasonal_variation * np.sin(seasonal_phase)
        intensity = max(0.1, min(0.9, intensity))  # Clamp values
        
        # Add some location-specific noise
        location_noise = 0.05 * np.sin(float(lat) * np.pi / 180) * np.cos(float(lon) * np.pi / 180)
        intensity += location_noise
        intensity = max(0.1, min(0.9, intensity))
        
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
def generate_conservation_insights(location, bloom_data_str):
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