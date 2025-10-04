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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import xgboost as xgb
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
    """AI-powered bloom prediction and anomaly detection with regional specificity"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.anomaly_threshold = 0.15
        self.regional_weights = {}
        self.climate_zones = {}
        self.vegetation_types = {}
        
    def train_model(self, historical_data, lat=None, lon=None):
        """Train ensemble ML models on historical bloom data with regional features"""
        try:
            # Prepare training data with regional features
            X, y = self._prepare_training_data(historical_data, lat, lon)
            
            if len(X) < 10:  # Need minimum data for training
                return False
                
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train ensemble of models
            model_scores = {}
            for name, model in self.models.items():
                try:
                    # Cross-validation to assess model performance
                    scores = cross_val_score(model, X_scaled, y, cv=3, scoring='neg_mean_squared_error')
                    model_scores[name] = -scores.mean()
                    
                    # Train the model
                    model.fit(X_scaled, y)
                except Exception as e:
                    print(f"Failed to train {name}: {e}")
                    model_scores[name] = float('inf')
            
            # Calculate ensemble weights based on performance
            if model_scores:
                total_score = sum(1/score for score in model_scores.values() if score != float('inf'))
                self.regional_weights = {
                    name: (1/score)/total_score if score != float('inf') else 0
                    for name, score in model_scores.items()
                }
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Model training failed: {e}")
            return False
    
    def _prepare_training_data(self, data, lat=None, lon=None):
        """Prepare enhanced features and targets for ML training with regional specificity"""
        features = []
        targets = []
        
        for i in range(1, len(data)):
            if i >= 12:  # Need at least a year of data
                # Basic temporal features
                prev_12_months = [d['ndvi'] for d in data[i-12:i]]
                month = i % 12
                season = month // 3
                
                # Enhanced vegetation indices
                prev_evi = [d.get('evi', d['ndvi'] * 0.9) for d in data[i-12:i]]
                prev_savi = [d.get('savi', d['ndvi'] * 1.1) for d in data[i-12:i]]
                prev_gndvi = [d.get('gndvi', d['ndvi'] * 0.8) for d in data[i-12:i]]
                
                # Statistical features
                ndvi_mean = np.mean(prev_12_months)
                ndvi_std = np.std(prev_12_months)
                ndvi_trend = np.polyfit(range(12), prev_12_months, 1)[0]
                ndvi_peak = np.max(prev_12_months)
                ndvi_min = np.min(prev_12_months)
                
                # Seasonal features
                spring_avg = np.mean(prev_12_months[2:5])  # March-May
                summer_avg = np.mean(prev_12_months[5:8])  # June-August
                fall_avg = np.mean(prev_12_months[8:11])   # September-November
                winter_avg = np.mean([prev_12_months[11]] + prev_12_months[0:2])  # December-February
                
                # Regional features
                regional_features = self._get_regional_features(lat, lon, month)
                
                # Climate zone features
                climate_features = self._get_climate_features(lat, lon, month)
                
                # Vegetation type features
                vegetation_features = self._get_vegetation_features(lat, lon)
                
                # Combine all features
                feature_vector = (
                    prev_12_months +  # 12 NDVI values
                    prev_evi +        # 12 EVI values
                    prev_savi +       # 12 SAVI values
                    prev_gndvi +      # 12 GNDVI values
                    [month, season, ndvi_mean, ndvi_std, ndvi_trend, ndvi_peak, ndvi_min] +  # 7 statistical features
                    [spring_avg, summer_avg, fall_avg, winter_avg] +  # 4 seasonal averages
                    regional_features +  # Regional features
                    climate_features +   # Climate features
                    vegetation_features  # Vegetation features
                )
                
                features.append(feature_vector)
                targets.append(data[i]['ndvi'])
        
        return np.array(features), np.array(targets)
    
    def _get_regional_features(self, lat, lon, month):
        """Get regional-specific features based on latitude, longitude, and month"""
        if lat is None or lon is None:
            return [0] * 8  # Default features
        
        # Hemisphere features
        hemisphere = 1 if lat >= 0 else -1
        
        # Distance from equator (affects seasonality)
        equator_distance = abs(lat)
        
        # Continental vs maritime influence (distance from coast approximation)
        # Simplified: distance from major coastlines
        continental_factor = min(1.0, max(0.0, 1.0 - abs(lon) / 180.0))
        
        # Elevation approximation (simplified based on latitude)
        elevation_factor = max(0.0, min(1.0, (abs(lat) - 30) / 60))  # Higher at poles
        
        # Urban vs rural (simplified based on longitude patterns)
        urban_factor = 0.5 + 0.3 * np.sin(lon * np.pi / 180)  # Simplified urban distribution
        
        # Seasonal day length effect
        day_length_factor = 0.5 + 0.5 * np.cos(2 * np.pi * (month - 6) / 12) * np.cos(lat * np.pi / 180)
        
        # Temperature seasonality
        temp_seasonality = np.sin(2 * np.pi * (month - 3) / 12) * hemisphere
        
        # Precipitation seasonality (simplified)
        precip_seasonality = 0.5 + 0.3 * np.sin(2 * np.pi * (month - 6) / 12)
        
        return [
            hemisphere, equator_distance, continental_factor, elevation_factor,
            urban_factor, day_length_factor, temp_seasonality, precip_seasonality
        ]
    
    def _get_climate_features(self, lat, lon, month):
        """Get climate zone features based on Köppen classification approximation"""
        if lat is None or lon is None:
            return [0] * 6  # Default features
        
        # Simplified Köppen climate classification
        if abs(lat) < 10:
            climate_zone = 1  # Tropical
        elif abs(lat) < 25:
            climate_zone = 2  # Subtropical
        elif abs(lat) < 40:
            climate_zone = 3  # Temperate
        elif abs(lat) < 60:
            climate_zone = 4  # Continental
        else:
            climate_zone = 5  # Polar
        
        # Temperature features
        temp_range = max(0, min(1, (abs(lat) - 20) / 40))  # Temperature range increases with latitude
        
        # Precipitation features
        if abs(lat) < 10:
            precip_pattern = 1  # High year-round
        elif abs(lat) < 30:
            precip_pattern = 2  # Seasonal
        else:
            precip_pattern = 3  # Moderate
        
        # Growing season length (approximation)
        growing_season = max(0, min(1, (60 - abs(lat)) / 60))
        
        # Frost risk
        frost_risk = max(0, min(1, (abs(lat) - 30) / 30))
        
        # Drought risk (simplified)
        drought_risk = 0.3 + 0.4 * abs(np.sin(lon * np.pi / 180))
        
        return [climate_zone, temp_range, precip_pattern, growing_season, frost_risk, drought_risk]
    
    def _get_vegetation_features(self, lat, lon):
        """Get vegetation type features based on location"""
        if lat is None or lon is None:
            return [0] * 5  # Default features
        
        # Simplified vegetation classification
        if abs(lat) < 10:
            vegetation_type = 1  # Tropical forest
        elif abs(lat) < 25:
            vegetation_type = 2  # Subtropical forest/grassland
        elif abs(lat) < 40:
            vegetation_type = 3  # Temperate forest
        elif abs(lat) < 60:
            vegetation_type = 4  # Boreal forest/tundra
        else:
            vegetation_type = 5  # Arctic tundra
        
        # Vegetation density (approximation)
        if abs(lat) < 20:
            density = 0.9  # High density in tropics
        elif abs(lat) < 40:
            density = 0.7  # Medium-high density
        elif abs(lat) < 60:
            density = 0.5  # Medium density
        else:
            density = 0.2  # Low density in polar regions
        
        # Deciduous vs evergreen (simplified)
        deciduous_ratio = max(0, min(1, (40 - abs(lat)) / 40))
        
        # Grassland vs forest ratio
        grassland_ratio = 0.3 + 0.4 * abs(np.sin(lon * np.pi / 180))
        
        # Agricultural land use (simplified)
        agricultural_ratio = 0.2 + 0.3 * abs(np.cos(lon * np.pi / 180))
        
        return [vegetation_type, density, deciduous_ratio, grassland_ratio, agricultural_ratio]
    
    def predict_bloom(self, recent_data, days_ahead=30, lat=None, lon=None):
        """Predict future bloom intensity with detailed regional analysis"""
        if not self.is_trained or len(recent_data) < 12:
            return self._fallback_prediction(recent_data, days_ahead, lat, lon)
        
        try:
            # Prepare enhanced features
            last_12_months = [d['ndvi'] for d in recent_data[-12:]]
            month = len(recent_data) % 12
            season = month // 3
            
            # Enhanced vegetation indices
            last_evi = [d.get('evi', d['ndvi'] * 0.9) for d in recent_data[-12:]]
            last_savi = [d.get('savi', d['ndvi'] * 1.1) for d in recent_data[-12:]]
            last_gndvi = [d.get('gndvi', d['ndvi'] * 0.8) for d in recent_data[-12:]]
            
            # Statistical features
            ndvi_mean = np.mean(last_12_months)
            ndvi_std = np.std(last_12_months)
            ndvi_trend = np.polyfit(range(12), last_12_months, 1)[0]
            ndvi_peak = np.max(last_12_months)
            ndvi_min = np.min(last_12_months)
            
            # Seasonal features
            spring_avg = np.mean(last_12_months[2:5])
            summer_avg = np.mean(last_12_months[5:8])
            fall_avg = np.mean(last_12_months[8:11])
            winter_avg = np.mean([last_12_months[11]] + last_12_months[0:2])
            
            # Regional features
            regional_features = self._get_regional_features(lat, lon, month)
            climate_features = self._get_climate_features(lat, lon, month)
            vegetation_features = self._get_vegetation_features(lat, lon)
            
            # Combine all features
            feature_vector = (
                last_12_months + last_evi + last_savi + last_gndvi +
                [month, season, ndvi_mean, ndvi_std, ndvi_trend, ndvi_peak, ndvi_min] +
                [spring_avg, summer_avg, fall_avg, winter_avg] +
                regional_features + climate_features + vegetation_features
            )
            
            X = np.array([feature_vector])
            X_scaled = self.scaler.transform(X)
            
            # Ensemble prediction
            predictions = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(X_scaled)[0]
                    predictions[name] = float(pred)
                except:
                    predictions[name] = 0.5
            
            # Weighted ensemble prediction
            if self.regional_weights:
                ensemble_prediction = sum(
                    predictions[name] * self.regional_weights.get(name, 0)
                    for name in predictions.keys()
                )
            else:
                ensemble_prediction = np.mean(list(predictions.values()))
            
            # Calculate confidence based on model agreement and data quality
            model_agreement = 1.0 - np.std(list(predictions.values()))
            data_quality = 1.0 - min(1.0, ndvi_std)
            confidence = min(0.95, max(0.1, (model_agreement + data_quality) / 2))
            
            # Generate detailed prediction breakdown
            prediction_details = self._generate_prediction_details(
                ensemble_prediction, predictions, recent_data, lat, lon, month
            )
            
            return {
                'predicted_intensity': float(ensemble_prediction),
                'confidence': float(confidence),
                'days_ahead': days_ahead,
                'model_used': 'Ensemble',
                'individual_predictions': predictions,
                'model_weights': self.regional_weights,
                'prediction_details': prediction_details,
                'regional_analysis': self._get_regional_analysis(lat, lon, month),
                'risk_factors': self._assess_risk_factors(recent_data, lat, lon),
                'uncertainty_range': self._calculate_uncertainty_range(predictions, confidence)
            }
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            return self._fallback_prediction(recent_data, days_ahead, lat, lon)
    
    def _fallback_prediction(self, recent_data, days_ahead, lat=None, lon=None):
        """Enhanced fallback prediction using simple trend analysis with regional adjustments"""
        if len(recent_data) < 3:
            return {
                'predicted_intensity': 0.5, 
                'confidence': 0.1, 
                'days_ahead': days_ahead, 
                'model_used': 'Fallback',
                'prediction_details': {'method': 'insufficient_data'},
                'regional_analysis': self._get_regional_analysis(lat, lon, 0),
                'risk_factors': []
            }
        
        recent_values = [d['ndvi'] for d in recent_data[-3:]]
        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        predicted = recent_values[-1] + trend * (days_ahead / 30)
        
        # Regional adjustment
        if lat is not None and lon is not None:
            regional_features = self._get_regional_features(lat, lon, 0)
            regional_adjustment = 0.1 * regional_features[1]  # Adjust based on distance from equator
            predicted += regional_adjustment
        
        return {
            'predicted_intensity': float(max(0, min(1, predicted))),
            'confidence': 0.3,
            'days_ahead': days_ahead,
            'model_used': 'Trend',
            'prediction_details': {'method': 'trend_analysis', 'trend': float(trend)},
            'regional_analysis': self._get_regional_analysis(lat, lon, 0),
            'risk_factors': self._assess_risk_factors(recent_data, lat, lon)
        }
    
    def _generate_prediction_details(self, ensemble_prediction, individual_predictions, recent_data, lat, lon, month):
        """Generate detailed breakdown of prediction factors"""
        details = {
            'ensemble_method': 'weighted_average',
            'primary_factors': [],
            'seasonal_influence': self._get_seasonal_influence(month, lat),
            'trend_analysis': self._analyze_trend(recent_data),
            'peak_timing': self._predict_peak_timing(ensemble_prediction, month, lat),
            'intensity_curve': self._generate_intensity_curve(ensemble_prediction, month, lat)
        }
        
        # Identify primary contributing factors
        if lat is not None:
            if abs(lat) < 10:
                details['primary_factors'].append('tropical_climate')
            elif abs(lat) < 30:
                details['primary_factors'].append('subtropical_seasonality')
            elif abs(lat) < 50:
                details['primary_factors'].append('temperate_seasonality')
            else:
                details['primary_factors'].append('high_latitude_constraints')
        
        # Model agreement analysis
        model_std = np.std(list(individual_predictions.values()))
        if model_std < 0.05:
            details['primary_factors'].append('high_model_agreement')
        elif model_std > 0.15:
            details['primary_factors'].append('model_uncertainty')
        
        return details
    
    def _get_regional_analysis(self, lat, lon, month):
        """Get detailed regional analysis for the prediction"""
        if lat is None or lon is None:
            return {'region_type': 'unknown', 'characteristics': []}
        
        analysis = {
            'hemisphere': 'northern' if lat >= 0 else 'southern',
            'latitude_zone': self._get_latitude_zone(lat),
            'climate_characteristics': self._get_climate_characteristics(lat, lon),
            'vegetation_characteristics': self._get_vegetation_characteristics(lat, lon),
            'seasonal_patterns': self._get_seasonal_patterns(lat, month),
            'dominant_factors': self._get_dominant_factors(lat, lon, month)
        }
        
        return analysis
    
    def _assess_risk_factors(self, recent_data, lat, lon):
        """Assess risk factors that could affect bloom prediction accuracy"""
        risks = []
        
        if len(recent_data) < 12:
            risks.append({'type': 'insufficient_data', 'severity': 'high', 'description': 'Limited historical data'})
        
        if recent_data:
            recent_std = np.std([d['ndvi'] for d in recent_data[-6:]])
            if recent_std > 0.2:
                risks.append({'type': 'high_variability', 'severity': 'medium', 'description': 'High recent variability'})
        
        if lat is not None:
            if abs(lat) > 60:
                risks.append({'type': 'extreme_latitude', 'severity': 'medium', 'description': 'Extreme latitude conditions'})
            elif abs(lat) < 5:
                risks.append({'type': 'tropical_complexity', 'severity': 'low', 'description': 'Complex tropical patterns'})
        
        return risks
    
    def _calculate_uncertainty_range(self, predictions, confidence):
        """Calculate uncertainty range for predictions"""
        if not predictions:
            return {'lower': 0.0, 'upper': 1.0}
        
        values = list(predictions.values())
        mean_pred = np.mean(values)
        std_pred = np.std(values)
        
        # Adjust uncertainty based on confidence
        uncertainty_factor = 1.0 - confidence
        
        return {
            'lower': max(0.0, mean_pred - std_pred - uncertainty_factor * 0.1),
            'upper': min(1.0, mean_pred + std_pred + uncertainty_factor * 0.1),
            'standard_deviation': float(std_pred)
        }
    
    def _get_latitude_zone(self, lat):
        """Get latitude zone classification"""
        if abs(lat) < 10:
            return 'tropical'
        elif abs(lat) < 25:
            return 'subtropical'
        elif abs(lat) < 40:
            return 'temperate'
        elif abs(lat) < 60:
            return 'continental'
        else:
            return 'polar'
    
    def _get_climate_characteristics(self, lat, lon):
        """Get climate characteristics for the region"""
        characteristics = []
        
        if abs(lat) < 10:
            characteristics.extend(['high_temperature', 'high_humidity', 'minimal_seasonality'])
        elif abs(lat) < 30:
            characteristics.extend(['moderate_seasonality', 'warm_temperatures'])
        elif abs(lat) < 50:
            characteristics.extend(['distinct_seasons', 'moderate_temperatures'])
        else:
            characteristics.extend(['extreme_seasonality', 'cold_temperatures'])
        
        return characteristics
    
    def _get_vegetation_characteristics(self, lat, lon):
        """Get vegetation characteristics for the region"""
        characteristics = []
        
        if abs(lat) < 10:
            characteristics.extend(['tropical_forest', 'high_biodiversity', 'year_round_growth'])
        elif abs(lat) < 30:
            characteristics.extend(['mixed_vegetation', 'seasonal_variation'])
        elif abs(lat) < 50:
            characteristics.extend(['temperate_forest', 'deciduous_dominant'])
        else:
            characteristics.extend(['boreal_forest', 'coniferous_dominant', 'short_growing_season'])
        
        return characteristics
    
    def _get_seasonal_patterns(self, lat, month):
        """Get seasonal patterns for the region"""
        if abs(lat) < 10:
            return {'pattern': 'minimal_seasonality', 'peak_months': 'year_round'}
        elif abs(lat) < 30:
            return {'pattern': 'subtropical', 'peak_months': 'spring_autumn'}
        elif abs(lat) < 50:
            return {'pattern': 'temperate', 'peak_months': 'spring_summer'}
        else:
            return {'pattern': 'high_latitude', 'peak_months': 'summer'}
    
    def _get_dominant_factors(self, lat, lon, month):
        """Get dominant factors influencing bloom prediction"""
        factors = []
        
        if abs(lat) < 10:
            factors.extend(['temperature', 'precipitation', 'humidity'])
        elif abs(lat) < 30:
            factors.extend(['seasonal_temperature', 'precipitation_patterns'])
        elif abs(lat) < 50:
            factors.extend(['temperature_seasonality', 'day_length', 'frost_risk'])
        else:
            factors.extend(['extreme_temperature_variation', 'day_length_variation', 'growing_season_length'])
        
        return factors
    
    def _get_seasonal_influence(self, month, lat):
        """Get seasonal influence on bloom prediction"""
        if lat is None:
            return {'influence': 'unknown'}
        
        hemisphere = 1 if lat >= 0 else -1
        seasonal_phase = (month * hemisphere) % 12
        
        if seasonal_phase in [2, 3, 4]:  # Spring
            return {'influence': 'spring_growth', 'strength': 'high'}
        elif seasonal_phase in [5, 6, 7]:  # Summer
            return {'influence': 'summer_peak', 'strength': 'high'}
        elif seasonal_phase in [8, 9, 10]:  # Fall
            return {'influence': 'autumn_decline', 'strength': 'medium'}
        else:  # Winter
            return {'influence': 'winter_dormancy', 'strength': 'low'}
    
    def _analyze_trend(self, recent_data):
        """Analyze trend in recent data"""
        if len(recent_data) < 6:
            return {'trend': 'insufficient_data'}
        
        values = [d['ndvi'] for d in recent_data[-6:]]
        trend = np.polyfit(range(len(values)), values, 1)[0]
        
        if trend > 0.01:
            return {'trend': 'increasing', 'strength': abs(trend)}
        elif trend < -0.01:
            return {'trend': 'decreasing', 'strength': abs(trend)}
        else:
            return {'trend': 'stable', 'strength': abs(trend)}
    
    def _predict_peak_timing(self, prediction, month, lat):
        """Predict peak bloom timing"""
        if lat is None:
            return {'peak_month': 'unknown'}
        
        # Simplified peak timing based on latitude and current month
        if abs(lat) < 10:
            return {'peak_month': 'year_round', 'next_peak': 'ongoing'}
        elif abs(lat) < 30:
            peak_months = [3, 4, 9, 10]  # Spring and fall
            next_peak = min([m for m in peak_months if m > month], default=peak_months[0])
            return {'peak_month': 'spring_autumn', 'next_peak': next_peak}
        elif abs(lat) < 50:
            peak_months = [4, 5, 6, 7]  # Spring to summer
            next_peak = min([m for m in peak_months if m > month], default=peak_months[0])
            return {'peak_month': 'spring_summer', 'next_peak': next_peak}
        else:
            peak_months = [6, 7, 8]  # Summer only
            next_peak = min([m for m in peak_months if m > month], default=peak_months[0])
            return {'peak_month': 'summer', 'next_peak': next_peak}
    
    def _generate_intensity_curve(self, prediction, month, lat):
        """Generate predicted intensity curve over time"""
        if lat is None:
            return {'curve': 'unknown'}
        
        # Generate a simple intensity curve based on seasonal patterns
        months = list(range(12))
        intensities = []
        
        for m in months:
            if abs(lat) < 10:  # Tropical - relatively constant
                intensity = prediction * (0.8 + 0.2 * np.sin(2 * np.pi * m / 12))
            elif abs(lat) < 30:  # Subtropical - two peaks
                intensity = prediction * (0.6 + 0.4 * np.sin(2 * np.pi * m / 6))
            elif abs(lat) < 50:  # Temperate - one main peak
                intensity = prediction * (0.3 + 0.7 * np.sin(2 * np.pi * (m - 3) / 12))
            else:  # High latitude - sharp summer peak
                intensity = prediction * max(0.1, np.sin(2 * np.pi * (m - 6) / 12))
            
            intensities.append(max(0, min(1, intensity)))
        
        return {
            'monthly_intensities': intensities,
            'peak_intensity': max(intensities),
            'peak_month': months[intensities.index(max(intensities))]
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
        
        # Train model and predict with regional features
        bloom_monitor.predictor.train_model(historical_data['data'], lat, lon)
        prediction = bloom_monitor.predictor.predict_bloom(historical_data['data'], days_ahead, lat, lon)
        
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