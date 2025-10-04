// BloomWatch Application JavaScript

class BloomWatchApp {
    constructor() {
        this.map = null;
        this.chart = null;
        this.currentData = null;
        this.currentLocation = 'global';
        this.currentCity = 'global';
        this.currentTimeRange = 5;
        this.currentVegetationIndex = 'ndvi';
        this.cities = {};
        this.predictions = null;
        this.anomalies = [];
        this.citizenObservations = [];
        this.climateCorrelations = {};
        
        this.init();
    }
    
    init() {
        this.initializeMap();
        this.initializeChart();
        this.loadCities();
        this.loadInitialData();
        this.loadCitizenObservations();
        this.setupEventListeners();
    }
    
    initializeMap() {
        // Initialize Leaflet map
        this.map = L.map('globalMap').setView([20, 0], 2);
        
        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map);
        
        // Add NASA attribution
        this.map.attributionControl.addAttribution('NASA Earth Observation Data');
    }
    
    initializeChart() {
        const ctx = document.getElementById('timeSeriesChart').getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Bloom Intensity',
                    data: [],
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#28a745',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#28a745',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Date'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Bloom Intensity'
                        },
                        min: 0,
                        max: 1,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }
    
    setupEventListeners() {
        // Add smooth scrolling for navigation buttons
        window.scrollToMap = () => {
            document.getElementById('mapSection').scrollIntoView({ 
                behavior: 'smooth' 
            });
        };
        
        window.scrollToAnalysis = () => {
            document.getElementById('analysisSection').scrollIntoView({ 
                behavior: 'smooth' 
            });
        };
        
        // Global functions for form controls
        window.updateCity = () => {
            this.currentCity = document.getElementById('citySelect').value;
            console.log('City selected:', this.currentCity);
            this.updateLocationFromCity();
            this.refreshData();
        };
        
        window.updateTimeRange = () => {
            this.currentTimeRange = parseInt(document.getElementById('timeRange').value);
            this.refreshData();
        };
        
        // Global function to select a city from map popup
        window.selectCity = (cityKey) => {
            this.currentCity = cityKey;
            document.getElementById('citySelect').value = cityKey;
            this.updateLocationFromCity();
            
            // Immediately center and zoom to the selected city
            if (this.cities[cityKey]) {
                const city = this.cities[cityKey];
                this.map.setView([city.lat, city.lon], 10);
                
                // Update map title and description
                const mapTitle = document.getElementById('mapTitle');
                const mapDescription = document.getElementById('mapDescription');
                if (mapTitle) {
                    mapTitle.textContent = `${city.name} Bloom Map`;
                }
                if (mapDescription) {
                    mapDescription.textContent = `Bloom data for ${city.name}, ${city.country}`;
                }
            }
            
            this.refreshData();
            
            // Close any open popups
            this.map.closePopup();
        };
        
        window.updateVegetationIndex = () => {
            this.currentVegetationIndex = document.getElementById('vegetationIndex').value;
            this.refreshData();
        };
        
        window.refreshData = () => {
            this.loadDataForCurrentLocation();
        };
        
        // AI Prediction functions
        window.predictBloom = () => {
            this.predictFutureBloom();
        };
        
        window.detectAnomalies = () => {
            this.detectBloomAnomalies();
        };
        
        window.analyzeClimate = () => {
            this.analyzeClimateCorrelation();
        };
        
        // Citizen Science functions
        window.submitObservation = () => {
            this.submitCitizenObservation();
        };
        
        window.view3DGlobe = () => {
            this.show3DGlobe();
        };
    }
    
    async loadCities() {
        try {
            const response = await fetch('/api/cities');
            if (response.ok) {
                const data = await response.json();
                this.cities = {};
                data.cities.forEach(city => {
                    // Create a key that matches the option values in the HTML
                    let key = city.name.toLowerCase()
                        .replace(/\s+/g, '-')
                        .replace(/[^a-z0-9-]/g, '');
                    
                    // Special handling for specific cities to match HTML options
                    if (city.name === 'New York City') key = 'new-york';
                    else if (city.name === 'São Paulo') key = 'sao-paulo';
                    else if (city.name === 'Cape Town') key = 'cape-town';
                    else if (city.name === 'Los Angeles') key = 'los-angeles';
                    else if (city.name === 'Buenos Aires') key = 'buenos-aires';
                    
                    this.cities[key] = city;
                });
            }
        } catch (error) {
            console.warn('Could not load cities:', error);
        }
    }
    
    updateLocationFromCity() {
        if (this.currentCity === 'global') {
            this.currentLocation = 'global';
        } else if (this.cities[this.currentCity]) {
            const city = this.cities[this.currentCity];
            this.currentLocation = `${city.name}, ${city.country}`;
        }
    }
    
    async loadInitialData() {
        this.showLoading();
        
        try {
            // Load data in parallel for better performance
            const [bloomData, trends, insights] = await Promise.all([
                this.fetchBloomData(),
                this.fetchTrends(),
                this.fetchConservationInsights()
            ]);
            
            this.currentData = bloomData;
            
            // Update all components
            await this.updateMap(bloomData);
            this.updateChart(bloomData);
            this.updateMetrics(bloomData);
            this.updateTrends(trends);
            this.updateInsights(insights);
            
        } catch (error) {
            console.error('Error loading data:', error);
            this.showError('Failed to load data. Please try again.');
        } finally {
            this.hideLoading();
        }
    }
    
    async loadDataForCurrentLocation() {
        this.showLoading();
        
        try {
            // Update location from current city selection
            this.updateLocationFromCity();
            
            // Load data in parallel for better performance
            const [bloomData, trends, insights] = await Promise.all([
                this.fetchBloomData(),
                this.fetchTrends(),
                this.fetchConservationInsights()
            ]);
            
            this.currentData = bloomData;
            
            // Update all components
            await this.updateMap(bloomData);
            this.updateChart(bloomData);
            this.updateMetrics(bloomData);
            this.updateTrends(trends);
            this.updateInsights(insights);
            
        } catch (error) {
            console.error('Error loading data:', error);
            this.showError('Failed to load data. Please try again.');
        } finally {
            this.hideLoading();
        }
    }
    
    async fetchBloomData() {
        let url = '/api/bloom-data';
        
        // Add city coordinates if a specific city is selected
        if (this.currentCity !== 'global' && this.cities[this.currentCity]) {
            const city = this.cities[this.currentCity];
            url += `?lat=${city.lat}&lon=${city.lon}`;
        }
        
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error('Failed to fetch bloom data');
        }
        return await response.json();
    }
    
    async fetchGlobalBloomMap() {
        const response = await fetch('/api/global-bloom-map');
        if (!response.ok) {
            throw new Error('Failed to fetch global bloom map');
        }
        return await response.json();
    }
    
    async fetchTrends() {
        const location = this.currentLocation || 'global';
        const response = await fetch(`/api/trends?location=${location}&years=${this.currentTimeRange}`);
        if (!response.ok) {
            throw new Error('Failed to fetch trends');
        }
        return await response.json();
    }
    
    async fetchConservationInsights() {
        const location = this.currentLocation || 'global';
        const response = await fetch(`/api/conservation-insights?location=${location}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(this.currentData)
        });
        if (!response.ok) {
            throw new Error('Failed to fetch conservation insights');
        }
        return await response.json();
    }
    
    async updateMap(bloomData) {
        // Clear existing markers
        this.map.eachLayer(layer => {
            if (layer instanceof L.CircleMarker) {
                this.map.removeLayer(layer);
            }
        });
        
        // Center map on selected city or keep global view
        if (this.currentCity !== 'global' && this.cities[this.currentCity]) {
            const city = this.cities[this.currentCity];
            this.map.setView([city.lat, city.lon], 10);
            
            // Add a prominent marker for the selected city
            const cityMarker = L.circleMarker([city.lat, city.lon], {
                radius: 12,
                fillColor: '#e91e63',
                color: '#fff',
                weight: 3,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(this.map);
            
            cityMarker.bindPopup(`
                <div class="text-center">
                    <h6><strong>${city.name}</strong></h6>
                    <p class="mb-1">${city.country}</p>
                    <small class="text-muted">${city.description}</small>
                </div>
            `);
            
            // Update map title and description
            const mapTitle = document.getElementById('mapTitle');
            const mapDescription = document.getElementById('mapDescription');
            if (mapTitle) {
                mapTitle.textContent = `${city.name} Bloom Map`;
            }
            if (mapDescription) {
                mapDescription.textContent = `Bloom data for ${city.name}, ${city.country}`;
            }
        } else {
            this.map.setView([20, 0], 2);
            
            // Add markers for all available cities in global view
            Object.entries(this.cities).forEach(([cityKey, city]) => {
                const cityMarker = L.circleMarker([city.lat, city.lon], {
                    radius: 8,
                    fillColor: '#e91e63',
                    color: '#fff',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.7
                }).addTo(this.map);
                
                cityMarker.bindPopup(`
                    <div class="text-center">
                        <h6><strong>${city.name}</strong></h6>
                        <p class="mb-1">${city.country}</p>
                        <small class="text-muted">${city.description}</small>
                        <br><br>
                        <button class="btn btn-sm btn-success" onclick="selectCity('${cityKey}')">
                            View ${city.name}
                        </button>
                    </div>
                `);
            });
            
            // Update map title and description
            const mapTitle = document.getElementById('mapTitle');
            const mapDescription = document.getElementById('mapDescription');
            if (mapTitle) {
                mapTitle.textContent = 'Global Bloom Map - All Cities';
            }
            if (mapDescription) {
                mapDescription.textContent = 'Click on pink city markers to view detailed bloom data';
            }
        }
        
        // Add bloom intensity markers
        if (bloomData.data && bloomData.data.length > 0) {
            bloomData.data.forEach(point => {
                const intensity = point[this.currentVegetationIndex] || point.ndvi;
                const color = this.getIntensityColor(intensity);
                const radius = Math.max(3, intensity * 10);
                
                L.circleMarker([point.latitude, point.longitude], {
                    radius: radius,
                    fillColor: color,
                    color: '#ffffff',
                    weight: 2,
                    opacity: 0.8,
                    fillOpacity: 0.6
                }).addTo(this.map).bindPopup(`
                    <div class="popup-content">
                        <h6>Bloom Data</h6>
                        <p><strong>Date:</strong> ${new Date(point.date).toLocaleDateString()}</p>
                        <p><strong>NDVI:</strong> ${point.ndvi.toFixed(3)}</p>
                        <p><strong>EVI:</strong> ${point.evi.toFixed(3)}</p>
                        <p><strong>Bloom Probability:</strong> ${(point.bloom_probability * 100).toFixed(1)}%</p>
                    </div>
                `);
            });
        }
        
        // Add global bloom data if available
        try {
            const globalData = await this.fetchGlobalBloomMap();
            if (globalData.data) {
                globalData.data.forEach(point => {
                    const color = this.getIntensityColor(point.bloom_intensity);
                    const radius = Math.max(2, point.bloom_intensity * 8);
                    
                    L.circleMarker([point.lat, point.lon], {
                        radius: radius,
                        fillColor: color,
                        color: '#ffffff',
                        weight: 1,
                        opacity: 0.6,
                        fillOpacity: 0.4
                    }).addTo(this.map).bindPopup(`
                        <div class="popup-content">
                            <h6>Global Bloom Status</h6>
                            <p><strong>Intensity:</strong> ${point.bloom_intensity.toFixed(3)}</p>
                            <p><strong>Status:</strong> <span class="bloom-status-${point.bloom_status}">${point.bloom_status}</span></p>
                        </div>
                    `);
                });
            }
        } catch (error) {
            console.warn('Could not load global bloom data:', error);
        }
    }
    
    updateChart(bloomData) {
        if (!bloomData.data || bloomData.data.length === 0) return;
        
        const labels = bloomData.data.map(point => new Date(point.date).toLocaleDateString());
        const data = bloomData.data.map(point => point[this.currentVegetationIndex] || point.ndvi);
        
        this.chart.data.labels = labels;
        this.chart.data.datasets[0].data = data;
        this.chart.data.datasets[0].label = `${this.currentVegetationIndex.toUpperCase()} Bloom Intensity`;
        this.chart.update('active');
    }
    
    updateMetrics(bloomData) {
        if (!bloomData.data || bloomData.data.length === 0) return;
        
        const latestData = bloomData.data[bloomData.data.length - 1];
        const avgIntensity = bloomData.summary.avg_bloom_intensity;
        const peakDate = new Date(bloomData.summary.peak_bloom_date).toLocaleDateString();
        
        // Update current intensity
        const currentIntensity = latestData[this.currentVegetationIndex];
        document.getElementById('currentIntensity').textContent = currentIntensity.toFixed(3);
        document.getElementById('intensityBar').style.width = `${currentIntensity * 100}%`;
        
        // Update peak date
        document.getElementById('peakDate').textContent = peakDate;
        
        // Update location display
        const locationElement = document.getElementById('currentLocationDisplay');
        if (locationElement) {
            if (this.currentLocation !== 'global') {
                locationElement.innerHTML = `<i class="fas fa-map-marker-alt me-1"></i>${this.currentLocation}`;
            } else {
                locationElement.innerHTML = `<i class="fas fa-satellite me-1"></i>Powered by NASA Earth Observation Data`;
            }
        }
    }
    
    updateTrends(trends) {
        if (!trends.trends) return;
        
        const trendElement = document.getElementById('trend');
        const trend = trends.trends.trend;
        
        trendElement.textContent = trend;
        trendElement.className = `fw-bold trend-${trend}`;
        
        // Update peak season
        const peakSeason = trends.trends.peak_season || 'Unknown';
        document.getElementById('peakSeason').textContent = peakSeason;
    }
    
    updateInsights(insights) {
        const insightsList = document.getElementById('insightsList');
        const recommendationsList = document.getElementById('recommendationsList');
        
        // Clear existing content
        insightsList.innerHTML = '';
        recommendationsList.innerHTML = '';
        
        // Add insights
        if (insights.insights && insights.insights.length > 0) {
            insights.insights.forEach(insight => {
                const li = document.createElement('li');
                li.innerHTML = `<i class="fas fa-lightbulb me-2"></i>${insight}`;
                insightsList.appendChild(li);
            });
        } else {
            insightsList.innerHTML = '<li><i class="fas fa-info-circle me-2"></i>No specific insights available for this location.</li>';
        }
        
        // Add recommendations
        if (insights.recommendations && insights.recommendations.length > 0) {
            insights.recommendations.forEach(recommendation => {
                const li = document.createElement('li');
                li.innerHTML = `<i class="fas fa-check-circle me-2"></i>${recommendation}`;
                recommendationsList.appendChild(li);
            });
        } else {
            recommendationsList.innerHTML = '<li><i class="fas fa-info-circle me-2"></i>No specific recommendations available.</li>';
        }
    }
    
    getIntensityColor(intensity) {
        if (intensity > 0.6) return '#28a745'; // Green - High
        if (intensity > 0.4) return '#ffc107'; // Yellow - Medium
        if (intensity > 0.2) return '#fd7e14'; // Orange - Low
        return '#dc3545'; // Red - Very Low
    }
    
    showLoading() {
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        modal.show();
        
        // Auto-hide loading after 2 seconds maximum
        setTimeout(() => {
            this.hideLoading();
        }, 2000);
    }
    
    hideLoading() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
        if (modal) {
            modal.hide();
        }
    }
    
    showError(message) {
        // Create and show error toast
        const toast = document.createElement('div');
        toast.className = 'toast align-items-center text-white bg-danger border-0';
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-exclamation-triangle me-2"></i>${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        document.body.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // Remove toast after it's hidden
        toast.addEventListener('hidden.bs.toast', () => {
            document.body.removeChild(toast);
        });
    }
    
    // AI Prediction Methods
    async predictFutureBloom() {
        try {
            this.showLoading();
            
            const response = await fetch('/api/predict-bloom', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    location: this.currentLocation,
                    days_ahead: 30
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.predictions = data.prediction;
                this.displayPrediction(data);
                this.showSuccess('Bloom prediction generated successfully!');
            } else {
                throw new Error('Failed to generate prediction');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            this.showError('Failed to generate bloom prediction');
        } finally {
            this.hideLoading();
        }
    }
    
    displayPrediction(predictionData) {
        const prediction = predictionData.prediction;
        const predictionCard = document.getElementById('predictionCard');
        
        if (predictionCard) {
            // Generate detailed prediction display
            const details = prediction.prediction_details || {};
            const regional = prediction.regional_analysis || {};
            const risks = prediction.risk_factors || [];
            const uncertainty = prediction.uncertainty_range || {};
            
            // Individual model predictions
            const individualPreds = prediction.individual_predictions || {};
            const modelPredsHtml = Object.entries(individualPreds).map(([model, value]) => 
                `<div class="d-flex justify-content-between">
                    <span class="text-capitalize">${model.replace('_', ' ')}</span>
                    <span class="badge bg-secondary">${(value * 100).toFixed(1)}%</span>
                </div>`
            ).join('');
            
            // Risk factors
            const riskHtml = risks.map(risk => 
                `<div class="alert alert-${risk.severity === 'high' ? 'danger' : risk.severity === 'medium' ? 'warning' : 'info'} alert-sm">
                    <i class="fas fa-exclamation-triangle me-1"></i>
                    <strong>${risk.type.replace('_', ' ')}:</strong> ${risk.description}
                </div>`
            ).join('');
            
            // Regional characteristics
            const climateChars = regional.climate_characteristics || [];
            const vegetationChars = regional.vegetation_characteristics || [];
            const dominantFactors = regional.dominant_factors || [];
            
            predictionCard.innerHTML = `
                <div class="card-header bg-primary text-white">
                    <h6 class="mb-0"><i class="fas fa-brain me-2"></i>Enhanced AI Bloom Prediction</h6>
                </div>
                <div class="card-body">
                    <!-- Main Prediction -->
                    <div class="row mb-3">
                        <div class="col-4">
                            <h4 class="text-primary">${(prediction.predicted_intensity * 100).toFixed(1)}%</h4>
                            <small class="text-muted">Predicted Intensity</small>
                        </div>
                        <div class="col-4">
                            <h4 class="text-success">${(prediction.confidence * 100).toFixed(1)}%</h4>
                            <small class="text-muted">Confidence</small>
                        </div>
                        <div class="col-4">
                            <h4 class="text-info">${prediction.days_ahead}</h4>
                            <small class="text-muted">Days Ahead</small>
                        </div>
                    </div>
                    
                    <!-- Uncertainty Range -->
                    <div class="mb-3">
                        <label class="form-label">Uncertainty Range:</label>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar bg-warning" style="width: ${uncertainty.lower * 100}%"></div>
                            <div class="progress-bar bg-primary" style="width: ${(uncertainty.upper - uncertainty.lower) * 100}%"></div>
                        </div>
                        <small class="text-muted">${(uncertainty.lower * 100).toFixed(1)}% - ${(uncertainty.upper * 100).toFixed(1)}%</small>
                    </div>
                    
                    <!-- Model Details -->
                    <div class="mb-3">
                        <h6><i class="fas fa-cogs me-1"></i>Model Details</h6>
                        <p><strong>Ensemble Method:</strong> ${prediction.model_used}</p>
                        <div class="model-predictions">
                            <small class="text-muted">Individual Model Predictions:</small>
                            ${modelPredsHtml}
                        </div>
                    </div>
                    
                    <!-- Regional Analysis -->
                    <div class="mb-3">
                        <h6><i class="fas fa-globe me-1"></i>Regional Analysis</h6>
                        <div class="row">
                            <div class="col-6">
                                <p><strong>Hemisphere:</strong> ${regional.hemisphere || 'Unknown'}</p>
                                <p><strong>Latitude Zone:</strong> ${regional.latitude_zone || 'Unknown'}</p>
                            </div>
                            <div class="col-6">
                                <p><strong>Climate:</strong> ${climateChars.slice(0, 2).join(', ') || 'Unknown'}</p>
                                <p><strong>Vegetation:</strong> ${vegetationChars.slice(0, 2).join(', ') || 'Unknown'}</p>
                            </div>
                        </div>
                        <div class="mt-2">
                            <small class="text-muted"><strong>Dominant Factors:</strong> ${dominantFactors.join(', ')}</small>
                        </div>
                    </div>
                    
                    <!-- Prediction Details -->
                    <div class="mb-3">
                        <h6><i class="fas fa-chart-line me-1"></i>Prediction Analysis</h6>
                        <div class="row">
                            <div class="col-6">
                                <p><strong>Seasonal Influence:</strong> ${details.seasonal_influence?.influence || 'Unknown'}</p>
                                <p><strong>Trend:</strong> ${details.trend_analysis?.trend || 'Unknown'}</p>
                            </div>
                            <div class="col-6">
                                <p><strong>Peak Timing:</strong> ${details.peak_timing?.peak_month || 'Unknown'}</p>
                                <p><strong>Next Peak:</strong> ${details.peak_timing?.next_peak || 'Unknown'}</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Risk Factors -->
                    ${risks.length > 0 ? `
                    <div class="mb-3">
                        <h6><i class="fas fa-exclamation-triangle me-1"></i>Risk Assessment</h6>
                        ${riskHtml}
                    </div>
                    ` : ''}
                    
                    <!-- Intensity Curve Preview -->
                    ${details.intensity_curve ? `
                    <div class="mb-3">
                        <h6><i class="fas fa-chart-area me-1"></i>Seasonal Intensity Pattern</h6>
                        <div class="intensity-preview">
                            <canvas id="intensityChart" width="300" height="100"></canvas>
                        </div>
                    </div>
                    ` : ''}
                </div>
            `;
            predictionCard.style.display = 'block';
            
            // Draw intensity curve if available
            if (details.intensity_curve && details.intensity_curve.monthly_intensities) {
                this.drawIntensityCurve(details.intensity_curve.monthly_intensities);
            }
        }
    }
    
    drawIntensityCurve(intensities) {
        const canvas = document.getElementById('intensityChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw axes
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(20, height - 20);
        ctx.lineTo(width - 20, height - 20);
        ctx.moveTo(20, 20);
        ctx.lineTo(20, height - 20);
        ctx.stroke();
        
        // Draw curve
        ctx.strokeStyle = '#28a745';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const stepX = (width - 40) / (intensities.length - 1);
        for (let i = 0; i < intensities.length; i++) {
            const x = 20 + i * stepX;
            const y = height - 20 - (intensities[i] * (height - 40));
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        
        // Add month labels
        ctx.fillStyle = '#666';
        ctx.font = '10px Arial';
        const months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'];
        for (let i = 0; i < intensities.length; i += 2) {
            const x = 20 + i * stepX;
            ctx.fillText(months[i], x - 3, height - 5);
        }
    }
    
    async detectBloomAnomalies() {
        try {
            this.showLoading();
            
            const response = await fetch('/api/detect-anomalies', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    location: this.currentLocation
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.anomalies = data.anomalies;
                this.displayAnomalies(data);
                this.showSuccess(`Found ${data.total_anomalies} bloom anomalies!`);
            } else {
                throw new Error('Failed to detect anomalies');
            }
        } catch (error) {
            console.error('Anomaly detection error:', error);
            this.showError('Failed to detect bloom anomalies');
        } finally {
            this.hideLoading();
        }
    }
    
    displayAnomalies(anomalyData) {
        const anomaliesCard = document.getElementById('anomaliesCard');
        
        if (anomaliesCard && anomalyData.anomalies.length > 0) {
            const anomaliesList = anomalyData.anomalies.map(anomaly => `
                <div class="anomaly-item mb-2 p-2 border rounded">
                    <div class="d-flex justify-content-between">
                        <span><strong>${new Date(anomaly.date).toLocaleDateString()}</strong></span>
                        <span class="badge bg-${anomaly.type === 'high' ? 'danger' : 'warning'}">${anomaly.type}</span>
                    </div>
                    <small class="text-muted">Value: ${anomaly.value.toFixed(3)} | Score: ${anomaly.anomaly_score.toFixed(2)}</small>
                </div>
            `).join('');
            
            anomaliesCard.innerHTML = `
                <div class="card-header bg-warning text-dark">
                    <h6 class="mb-0"><i class="fas fa-exclamation-triangle me-2"></i>Bloom Anomalies Detected</h6>
                </div>
                <div class="card-body">
                    <p class="mb-3">Found <strong>${anomalyData.total_anomalies}</strong> anomalies in ${anomalyData.analysis_period}</p>
                    <div class="anomalies-list" style="max-height: 300px; overflow-y: auto;">
                        ${anomaliesList}
                    </div>
                </div>
            `;
            anomaliesCard.style.display = 'block';
        }
    }
    
    async analyzeClimateCorrelation() {
        try {
            this.showLoading();
            
            // Get current city coordinates
            let lat = 40.7128, lon = -74.0060; // Default to NYC
            if (this.currentCity !== 'global' && this.cities[this.currentCity]) {
                const city = this.cities[this.currentCity];
                lat = city.lat;
                lon = city.lon;
            }
            
            const response = await fetch('/api/climate-correlation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    lat: lat,
                    lon: lon,
                    start_date: '2023-01-01',
                    end_date: '2023-12-31'
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.climateCorrelations = data.correlations;
                this.displayClimateCorrelation(data);
                this.showSuccess('Climate correlation analysis completed!');
            } else {
                throw new Error('Failed to analyze climate correlation');
            }
        } catch (error) {
            console.error('Climate analysis error:', error);
            this.showError('Failed to analyze climate correlation');
        } finally {
            this.hideLoading();
        }
    }
    
    displayClimateCorrelation(climateData) {
        const climateCard = document.getElementById('climateCard');
        
        if (climateCard) {
            const correlations = climateData.correlations;
            const correlationItems = Object.entries(correlations).map(([key, value]) => `
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="text-capitalize">${key.replace('_', ' ')}</span>
                    <span class="badge bg-${Math.abs(value) > 0.5 ? 'primary' : 'secondary'}">
                        ${value.toFixed(3)}
                    </span>
                </div>
            `).join('');
            
            climateCard.innerHTML = `
                <div class="card-header bg-info text-white">
                    <h6 class="mb-0"><i class="fas fa-thermometer-half me-2"></i>Climate Correlation</h6>
                </div>
                <div class="card-body">
                    <p class="mb-3">Bloom pattern correlation with climate variables:</p>
                    ${correlationItems}
                    <hr>
                    <small class="text-muted">
                        <i class="fas fa-info-circle me-1"></i>
                        Values closer to ±1.0 indicate stronger correlation
                    </small>
                </div>
            `;
            climateCard.style.display = 'block';
        }
    }
    
    // Citizen Science Methods
    async loadCitizenObservations() {
        try {
            const response = await fetch('/api/citizen-science');
            if (response.ok) {
                const data = await response.json();
                this.citizenObservations = data.observations;
                this.displayCitizenObservations(data);
            }
        } catch (error) {
            console.error('Failed to load citizen observations:', error);
        }
    }
    
    displayCitizenObservations(data) {
        const citizenCard = document.getElementById('citizenCard');
        
        if (citizenCard && data.observations.length > 0) {
            const observationsList = data.observations.map(obs => `
                <div class="observation-item mb-3 p-3 border rounded">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h6 class="mb-1">${obs.species}</h6>
                            <p class="mb-1 text-muted">${obs.bloom_status.replace('_', ' ')}</p>
                            <small class="text-muted">by ${obs.observer} on ${new Date(obs.date).toLocaleDateString()}</small>
                        </div>
                        <span class="badge bg-success">${(obs.confidence * 100).toFixed(0)}%</span>
                    </div>
                </div>
            `).join('');
            
            citizenCard.innerHTML = `
                <div class="card-header bg-success text-white">
                    <h6 class="mb-0"><i class="fas fa-users me-2"></i>Citizen Science Observations</h6>
                </div>
                <div class="card-body">
                    <p class="mb-3">${data.contribution_message}</p>
                    <div class="observations-list" style="max-height: 400px; overflow-y: auto;">
                        ${observationsList}
                    </div>
                    <button class="btn btn-outline-success btn-sm mt-3" onclick="showObservationForm()">
                        <i class="fas fa-plus me-1"></i>Submit Observation
                    </button>
                </div>
            `;
            citizenCard.style.display = 'block';
        }
    }
    
    async submitCitizenObservation() {
        // This would open a form modal in a real implementation
        this.showSuccess('Citizen science observation form would open here!');
    }
    
    async show3DGlobe() {
        try {
            this.showLoading();
            
            const response = await fetch('/api/3d-globe-data');
            if (response.ok) {
                const data = await response.json();
                
                // Show the 3D globe modal
                const modal = new bootstrap.Modal(document.getElementById('globeModal'));
                modal.show();
                
                // Initialize the 3D globe after modal is shown
                setTimeout(() => {
                    if (!window.globeApp) {
                        window.globeApp = new Globe3D('globeContainer', data);
                    } else {
                        window.globeApp.updateData(data);
                    }
                }, 300);
                
                this.showSuccess('3D Globe visualization loaded!');
            }
        } catch (error) {
            console.error('Failed to load 3D globe data:', error);
            this.showError('Failed to load 3D globe data');
        } finally {
            this.hideLoading();
        }
    }
    
    display3DGlobeInfo(data) {
        const globeCard = document.getElementById('globeCard');
        
        if (globeCard) {
            globeCard.innerHTML = `
                <div class="card-header bg-dark text-white">
                    <h6 class="mb-0"><i class="fas fa-globe me-2"></i>3D Globe Visualization</h6>
                </div>
                <div class="card-body">
                    <p class="mb-3">Interactive 3D Earth with bloom data visualization</p>
                    <div class="row text-center">
                        <div class="col-4">
                            <h4 class="text-primary">${data.total_points}</h4>
                            <small class="text-muted">Data Points</small>
                        </div>
                        <div class="col-4">
                            <h4 class="text-success">${data.time_lapse_available ? 'Yes' : 'No'}</h4>
                            <small class="text-muted">Time-lapse</small>
                        </div>
                        <div class="col-4">
                            <h4 class="text-info">${data.animation_speed}</h4>
                            <small class="text-muted">Speed</small>
                        </div>
                    </div>
                    <hr>
                    <p class="text-muted">
                        <i class="fas fa-info-circle me-1"></i>
                        Full 3D WebGL visualization would be implemented here with Three.js
                    </p>
                </div>
            `;
            globeCard.style.display = 'block';
        }
    }
    
    showSuccess(message) {
        const toast = document.createElement('div');
        toast.className = 'toast align-items-center text-white bg-success border-0';
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-check-circle me-2"></i>${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        document.body.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        toast.addEventListener('hidden.bs.toast', () => {
            document.body.removeChild(toast);
        });
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new BloomWatchApp();
});

// Add some utility functions
function formatDate(dateString) {
    return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

function formatNumber(number, decimals = 2) {
    return parseFloat(number).toFixed(decimals);
}

// Add animation classes when elements come into view
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
        }
    });
}, observerOptions);

// Observe all cards for animation
document.addEventListener('DOMContentLoaded', () => {
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        observer.observe(card);
    });
});

// 3D Globe Visualization Class
class Globe3D {
    constructor(containerId, data) {
        this.container = document.getElementById(containerId);
        this.data = data;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.globe = null;
        this.bloomPoints = [];
        this.animationId = null;
        this.isAnimating = false;
        this.showBloomData = false;
        
        // Layer system
        this.layers = {
            temperature: { active: false, objects: [] },
            wind: { active: false, objects: [] },
            vegetation: { active: false, objects: [] },
            precipitation: { active: false, objects: [] },
            elevation: { active: false, objects: [] }
        };
        
        this.init();
    }
    
    init() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000011);
        
        // Create camera
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.position.set(0, 0, 3);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);
        
        // Add controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.enableZoom = true;
        this.controls.enablePan = false;
        this.controls.minDistance = 1.5;
        this.controls.maxDistance = 5;
        
        // Create globe
        this.createGlobe();
        
        // Add lighting
        this.addLighting();
        
        // Add bloom data points
        this.addBloomData();
        
        // Start render loop
        this.animate();
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Update info display
        this.updateInfo();
    }
    
    createGlobe() {
        // Create Earth geometry
        const geometry = new THREE.SphereGeometry(1, 64, 64);
        
        // Load world map texture with fallback
        const textureLoader = new THREE.TextureLoader();
        const worldTexture = textureLoader.load(
            'https://unpkg.com/three-globe@2.32.0/example/img/earth-blue-marble.jpg',
            // Success callback
            (texture) => {
                console.log('World map texture loaded successfully');
            },
            // Progress callback
            undefined,
            // Error callback - use fallback
            (error) => {
                console.warn('Failed to load world map texture, using fallback:', error);
                // Create a simple procedural texture as fallback
                this.createFallbackTexture();
            }
        );
        
        // Create Earth material with world map texture
        const material = new THREE.MeshPhongMaterial({
            map: worldTexture,
            transparent: false,
            shininess: 100
        });
        
        this.globe = new THREE.Mesh(geometry, material);
        this.globe.castShadow = true;
        this.globe.receiveShadow = true;
        this.scene.add(this.globe);
        
        // Add atmosphere
        const atmosphereGeometry = new THREE.SphereGeometry(1.05, 32, 32);
        const atmosphereMaterial = new THREE.MeshPhongMaterial({
            color: 0x87ceeb,
            transparent: true,
            opacity: 0.1,
            side: THREE.BackSide
        });
        const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
        this.scene.add(atmosphere);
    }
    
    createFallbackTexture() {
        // Create a simple procedural texture as fallback
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        
        // Create a simple world-like pattern
        const gradient = ctx.createLinearGradient(0, 0, 0, 512);
        gradient.addColorStop(0, '#1e3a8a'); // Dark blue (ocean)
        gradient.addColorStop(0.3, '#3b82f6'); // Blue (ocean)
        gradient.addColorStop(0.7, '#10b981'); // Green (land)
        gradient.addColorStop(1, '#059669'); // Dark green (land)
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 1024, 512);
        
        // Add some simple landmass shapes
        ctx.fillStyle = '#16a34a';
        ctx.beginPath();
        ctx.ellipse(200, 200, 80, 60, 0, 0, 2 * Math.PI); // North America
        ctx.fill();
        
        ctx.beginPath();
        ctx.ellipse(300, 300, 60, 40, 0, 0, 2 * Math.PI); // South America
        ctx.fill();
        
        ctx.beginPath();
        ctx.ellipse(500, 150, 70, 50, 0, 0, 2 * Math.PI); // Europe/Africa
        ctx.fill();
        
        ctx.beginPath();
        ctx.ellipse(800, 200, 60, 45, 0, 0, 2 * Math.PI); // Asia
        ctx.fill();
        
        const fallbackTexture = new THREE.CanvasTexture(canvas);
        if (this.globe && this.globe.material) {
            this.globe.material.map = fallbackTexture;
            this.globe.material.needsUpdate = true;
        }
    }
    
    addLighting() {
        // Ambient light - increased for better texture visibility
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        // Directional light (sun) - positioned to illuminate the globe nicely
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
        directionalLight.position.set(3, 2, 3);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);
        
        // Additional fill light for better texture visibility
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-2, 1, -2);
        this.scene.add(fillLight);
        
        // Point light for bloom points
        const pointLight = new THREE.PointLight(0x28a745, 0.5, 10);
        pointLight.position.set(0, 0, 0);
        this.scene.add(pointLight);
    }
    
    addBloomData() {
        if (!this.data || !this.data.globe_data) return;
        
        // Clear existing bloom points
        this.bloomPoints.forEach(point => {
            this.scene.remove(point);
        });
        this.bloomPoints = [];
        
        // Add bloom data points
        this.data.globe_data.forEach(point => {
            const bloomPoint = this.createBloomPoint(point);
            this.bloomPoints.push(bloomPoint);
            this.scene.add(bloomPoint);
        });
    }
    
    createBloomPoint(point) {
        // Convert lat/lon to 3D coordinates
        const lat = point.lat * Math.PI / 180;
        const lon = point.lon * Math.PI / 180;
        const radius = 1.02; // Slightly above the globe surface
        
        const x = radius * Math.cos(lat) * Math.cos(lon);
        const y = radius * Math.sin(lat);
        const z = radius * Math.cos(lat) * Math.sin(lon);
        
        // Create point geometry
        const geometry = new THREE.SphereGeometry(0.01, 8, 8);
        
        // Color based on bloom intensity
        const intensity = point.bloom_intensity;
        let color = 0x28a745; // Green
        if (intensity > 0.6) color = 0x28a745; // High - Green
        else if (intensity > 0.4) color = 0xffc107; // Medium - Yellow
        else if (intensity > 0.2) color = 0xfd7e14; // Low - Orange
        else color = 0xdc3545; // Very Low - Red
        
        const material = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: this.showBloomData ? 0.8 : 0
        });
        
        const bloomPoint = new THREE.Mesh(geometry, material);
        bloomPoint.position.set(x, y, z);
        bloomPoint.userData = point;
        
        return bloomPoint;
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        // Rotate globe slowly
        if (this.globe) {
            this.globe.rotation.y += 0.002;
        }
        
        // Animate wind layer if active
        if (this.layers.wind.active && this.layers.wind.objects.length > 0) {
            const windPoints = this.layers.wind.objects[0];
            if (windPoints && windPoints.userData && windPoints.userData.originalPositions) {
                const positions = windPoints.geometry.attributes.position.array;
                const time = Date.now() * 0.001;
                
                for (let i = 0; i < positions.length; i += 3) {
                    const originalX = windPoints.userData.originalPositions[i];
                    const originalY = windPoints.userData.originalPositions[i + 1];
                    const originalZ = windPoints.userData.originalPositions[i + 2];
                    
                    const windType = windPoints.userData.windTypes[i / 3];
                    let speed, amplitude;
                    
                    // Different speeds and amplitudes for different wind types
                    switch (windType) {
                        case 'trade':
                            speed = 0.5;
                            amplitude = 0.008;
                            break;
                        case 'westerly':
                            speed = 1.0;
                            amplitude = 0.012;
                            break;
                        case 'polar':
                            speed = 0.3;
                            amplitude = 0.006;
                            break;
                        case 'jet':
                            speed = 2.0;
                            amplitude = 0.015;
                            break;
                        default:
                            speed = 0.8;
                            amplitude = 0.01;
                    }
                    
                    // Add wind-like movement with different patterns
                    positions[i] = originalX + Math.sin(time * speed + i * 0.05) * amplitude;
                    positions[i + 1] = originalY + Math.cos(time * speed * 0.7 + i * 0.05) * amplitude * 0.5;
                    positions[i + 2] = originalZ + Math.sin(time * speed * 0.5 + i * 0.05) * amplitude;
                }
                
                windPoints.geometry.attributes.position.needsUpdate = true;
            }
        }
        
        // Update controls
        this.controls.update();
        
        // Render
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    resetView() {
        this.camera.position.set(0, 0, 3);
        this.controls.reset();
    }
    
    toggleAnimation() {
        this.isAnimating = !this.isAnimating;
        const icon = document.getElementById('animationIcon');
        const status = document.getElementById('globeAnimationStatus');
        
        if (this.isAnimating) {
            icon.className = 'fas fa-pause';
            status.textContent = 'Playing';
            // Globe rotation is always on, but we could add more animation here
        } else {
            icon.className = 'fas fa-play';
            status.textContent = 'Paused';
        }
    }
    
    toggleBloomData() {
        this.showBloomData = !this.showBloomData;
        const icon = document.getElementById('bloomIcon');
        const status = document.getElementById('globeBloomStatus');
        
        // Update bloom points visibility
        this.bloomPoints.forEach(point => {
            point.material.opacity = this.showBloomData ? 0.8 : 0;
        });
        
        if (this.showBloomData) {
            icon.className = 'fas fa-seedling';
            status.textContent = 'Visible';
        } else {
            icon.className = 'fas fa-seedling';
            status.textContent = 'Hidden';
        }
    }
    
    updateData(newData) {
        this.data = newData;
        this.addBloomData();
        this.updateInfo();
    }
    
    updateInfo() {
        const dataCount = document.getElementById('globeDataCount');
        if (dataCount && this.data) {
            dataCount.textContent = this.data.total_points || 0;
        }
        
        // Update active layers display
        const activeLayers = document.getElementById('activeLayers');
        if (activeLayers) {
            const activeLayerNames = Object.keys(this.layers).filter(layer => this.layers[layer].active);
            activeLayers.textContent = activeLayerNames.length > 0 ? activeLayerNames.join(', ') : 'None';
        }
    }
    
    toggleLayer(layerName) {
        if (!this.layers[layerName]) return;
        
        this.layers[layerName].active = !this.layers[layerName].active;
        
        if (this.layers[layerName].active) {
            this.createLayer(layerName);
        } else {
            this.removeLayer(layerName);
        }
        
        this.updateInfo();
    }
    
    createLayer(layerName) {
        switch (layerName) {
            case 'temperature':
                this.createTemperatureLayer();
                break;
            case 'wind':
                this.createWindLayer();
                break;
            case 'vegetation':
                this.createVegetationLayer();
                break;
            case 'precipitation':
                this.createPrecipitationLayer();
                break;
            case 'elevation':
                this.createElevationLayer();
                break;
        }
    }
    
    removeLayer(layerName) {
        if (this.layers[layerName] && this.layers[layerName].objects) {
            this.layers[layerName].objects.forEach(obj => {
                this.scene.remove(obj);
            });
            this.layers[layerName].objects = [];
        }
    }
    
    createTemperatureLayer() {
        // Create temperature visualization as colored overlay
        const geometry = new THREE.SphereGeometry(1.01, 32, 32);
        const material = new THREE.MeshBasicMaterial({
            transparent: true,
            opacity: 0.6,
            side: THREE.DoubleSide
        });
        
        // Create temperature gradient texture
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        
        // Create realistic temperature zones
        // Arctic regions (very cold)
        ctx.fillStyle = '#000080'; // Deep blue
        ctx.fillRect(0, 0, 1024, 60);
        ctx.fillRect(0, 452, 1024, 60);
        
        // Subarctic regions
        ctx.fillStyle = '#4169E1'; // Royal blue
        ctx.fillRect(0, 60, 1024, 40);
        ctx.fillRect(0, 412, 1024, 40);
        
        // Temperate regions
        ctx.fillStyle = '#00BFFF'; // Deep sky blue
        ctx.fillRect(0, 100, 1024, 30);
        ctx.fillRect(0, 382, 1024, 30);
        
        // Subtropical regions
        ctx.fillStyle = '#32CD32'; // Lime green
        ctx.fillRect(0, 130, 1024, 25);
        ctx.fillRect(0, 357, 1024, 25);
        
        // Tropical regions (hot)
        ctx.fillStyle = '#FF4500'; // Orange red
        ctx.fillRect(0, 155, 1024, 30);
        ctx.fillRect(0, 327, 1024, 30);
        
        // Equatorial region (hottest)
        ctx.fillStyle = '#FF0000'; // Red
        ctx.fillRect(0, 185, 1024, 20);
        ctx.fillRect(0, 307, 1024, 20);
        
        // Add continental temperature variations
        // North America - continental effect
        ctx.fillStyle = 'rgba(255, 100, 0, 0.4)';
        ctx.beginPath();
        ctx.moveTo(150, 80);
        ctx.bezierCurveTo(200, 60, 250, 70, 300, 90);
        ctx.bezierCurveTo(350, 110, 400, 120, 450, 100);
        ctx.bezierCurveTo(500, 80, 550, 60, 600, 80);
        ctx.bezierCurveTo(650, 100, 700, 120, 750, 100);
        ctx.bezierCurveTo(800, 80, 850, 60, 900, 80);
        ctx.lineTo(900, 120);
        ctx.bezierCurveTo(850, 100, 800, 120, 750, 140);
        ctx.bezierCurveTo(700, 160, 650, 180, 600, 160);
        ctx.bezierCurveTo(550, 140, 500, 160, 450, 180);
        ctx.bezierCurveTo(400, 200, 350, 190, 300, 170);
        ctx.bezierCurveTo(250, 150, 200, 160, 150, 140);
        ctx.closePath();
        ctx.fill();
        
        // Europe - maritime influence
        ctx.fillStyle = 'rgba(0, 150, 255, 0.3)';
        ctx.beginPath();
        ctx.moveTo(450, 120);
        ctx.bezierCurveTo(500, 100, 550, 110, 600, 130);
        ctx.bezierCurveTo(650, 150, 700, 160, 750, 140);
        ctx.bezierCurveTo(800, 120, 850, 100, 900, 120);
        ctx.lineTo(900, 160);
        ctx.bezierCurveTo(850, 140, 800, 160, 750, 180);
        ctx.bezierCurveTo(700, 200, 650, 190, 600, 170);
        ctx.bezierCurveTo(550, 150, 500, 160, 450, 180);
        ctx.closePath();
        ctx.fill();
        
        // Asia - continental extremes
        ctx.fillStyle = 'rgba(255, 50, 0, 0.5)';
        ctx.beginPath();
        ctx.moveTo(700, 60);
        ctx.bezierCurveTo(750, 40, 800, 50, 850, 70);
        ctx.bezierCurveTo(900, 90, 950, 100, 1000, 80);
        ctx.lineTo(1000, 120);
        ctx.bezierCurveTo(950, 100, 900, 120, 850, 140);
        ctx.bezierCurveTo(800, 160, 750, 150, 700, 130);
        ctx.closePath();
        ctx.fill();
        
        // Add ocean current influences
        ctx.fillStyle = 'rgba(0, 100, 200, 0.2)';
        for (let i = 0; i < 8; i++) {
            const x = i * 128;
            const y = 256 + Math.sin(i * 0.5) * 50;
            ctx.beginPath();
            ctx.ellipse(x, y, 60, 20, 0, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        const texture = new THREE.CanvasTexture(canvas);
        material.map = texture;
        
        const temperatureMesh = new THREE.Mesh(geometry, material);
        this.scene.add(temperatureMesh);
        this.layers.temperature.objects.push(temperatureMesh);
    }
    
    createWindLayer() {
        // Create realistic wind patterns
        const windGeometry = new THREE.BufferGeometry();
        const windPositions = [];
        const windColors = [];
        
        // Create major wind patterns
        // Trade winds (easterlies) - 0-30 degrees latitude
        for (let lat = -30; lat <= 30; lat += 5) {
            for (let lon = 0; lon < 360; lon += 15) {
                const latRad = (lat * Math.PI) / 180;
                const lonRad = (lon * Math.PI) / 180;
                const radius = 1.02;
                
                const x = radius * Math.cos(latRad) * Math.cos(lonRad);
                const y = radius * Math.sin(latRad);
                const z = radius * Math.cos(latRad) * Math.sin(lonRad);
                
                windPositions.push(x, y, z);
                // Trade winds - light blue
                windColors.push(0.3, 0.7, 1.0, 0.7);
            }
        }
        
        // Westerlies - 30-60 degrees latitude
        for (let lat = -60; lat <= -30; lat += 5) {
            for (let lon = 0; lon < 360; lon += 20) {
                const latRad = (lat * Math.PI) / 180;
                const lonRad = (lon * Math.PI) / 180;
                const radius = 1.02;
                
                const x = radius * Math.cos(latRad) * Math.cos(lonRad);
                const y = radius * Math.sin(latRad);
                const z = radius * Math.cos(latRad) * Math.sin(lonRad);
                
                windPositions.push(x, y, z);
                // Westerlies - cyan
                windColors.push(0.0, 0.8, 1.0, 0.6);
            }
        }
        
        for (let lat = 30; lat <= 60; lat += 5) {
            for (let lon = 0; lon < 360; lon += 20) {
                const latRad = (lat * Math.PI) / 180;
                const lonRad = (lon * Math.PI) / 180;
                const radius = 1.02;
                
                const x = radius * Math.cos(latRad) * Math.cos(lonRad);
                const y = radius * Math.sin(latRad);
                const z = radius * Math.cos(latRad) * Math.sin(lonRad);
                
                windPositions.push(x, y, z);
                // Westerlies - cyan
                windColors.push(0.0, 0.8, 1.0, 0.6);
            }
        }
        
        // Polar easterlies - 60-90 degrees latitude
        for (let lat = -90; lat <= -60; lat += 8) {
            for (let lon = 0; lon < 360; lon += 25) {
                const latRad = (lat * Math.PI) / 180;
                const lonRad = (lon * Math.PI) / 180;
                const radius = 1.02;
                
                const x = radius * Math.cos(latRad) * Math.cos(lonRad);
                const y = radius * Math.sin(latRad);
                const z = radius * Math.cos(latRad) * Math.sin(lonRad);
                
                windPositions.push(x, y, z);
                // Polar easterlies - white
                windColors.push(0.8, 0.9, 1.0, 0.5);
            }
        }
        
        for (let lat = 60; lat <= 90; lat += 8) {
            for (let lon = 0; lon < 360; lon += 25) {
                const latRad = (lat * Math.PI) / 180;
                const lonRad = (lon * Math.PI) / 180;
                const radius = 1.02;
                
                const x = radius * Math.cos(latRad) * Math.cos(lonRad);
                const y = radius * Math.sin(latRad);
                const z = radius * Math.cos(latRad) * Math.sin(lonRad);
                
                windPositions.push(x, y, z);
                // Polar easterlies - white
                windColors.push(0.8, 0.9, 1.0, 0.5);
            }
        }
        
        // Jet streams - high altitude winds
        for (let lat = -40; lat <= 40; lat += 10) {
            for (let lon = 0; lon < 360; lon += 30) {
                const latRad = (lat * Math.PI) / 180;
                const lonRad = (lon * Math.PI) / 180;
                const radius = 1.03; // Slightly higher altitude
                
                const x = radius * Math.cos(latRad) * Math.cos(lonRad);
                const y = radius * Math.sin(latRad);
                const z = radius * Math.cos(latRad) * Math.sin(lonRad);
                
                windPositions.push(x, y, z);
                // Jet streams - bright white
                windColors.push(1.0, 1.0, 1.0, 0.8);
            }
        }
        
        windGeometry.setAttribute('position', new THREE.Float32BufferAttribute(windPositions, 3));
        windGeometry.setAttribute('color', new THREE.Float32BufferAttribute(windColors, 4));
        
        const windMaterial = new THREE.PointsMaterial({
            size: 0.015,
            vertexColors: true,
            transparent: true,
            opacity: 0.7
        });
        
        const windPoints = new THREE.Points(windGeometry, windMaterial);
        this.scene.add(windPoints);
        this.layers.wind.objects.push(windPoints);
        
        // Store reference for animation with different speeds for different wind types
        windPoints.userData = { 
            originalPositions: [...windPositions],
            windTypes: windPositions.map((pos, i) => {
                const y = pos[1];
                if (Math.abs(y) < 0.5) return 'trade'; // Trade winds
                if (Math.abs(y) < 0.8) return 'westerly'; // Westerlies
                if (Math.abs(y) < 0.95) return 'polar'; // Polar easterlies
                return 'jet'; // Jet streams
            })
        };
    }
    
    createVegetationLayer() {
        // Create vegetation density visualization
        const geometry = new THREE.SphereGeometry(1.01, 32, 32);
        const material = new THREE.MeshBasicMaterial({
            transparent: true,
            opacity: 0.5,
            side: THREE.DoubleSide
        });
        
        // Create vegetation texture
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        
        // Base ocean color
        ctx.fillStyle = '#000080';
        ctx.fillRect(0, 0, 1024, 512);
        
        // North America - Boreal Forest
        ctx.fillStyle = '#006400'; // Dark green
        ctx.beginPath();
        ctx.moveTo(100, 80);
        ctx.bezierCurveTo(150, 60, 200, 70, 250, 90);
        ctx.bezierCurveTo(300, 110, 350, 120, 400, 100);
        ctx.bezierCurveTo(450, 80, 500, 60, 550, 80);
        ctx.bezierCurveTo(600, 100, 650, 120, 700, 100);
        ctx.bezierCurveTo(750, 80, 800, 60, 850, 80);
        ctx.lineTo(850, 120);
        ctx.bezierCurveTo(800, 100, 750, 120, 700, 140);
        ctx.bezierCurveTo(650, 160, 600, 180, 550, 160);
        ctx.bezierCurveTo(500, 140, 450, 160, 400, 180);
        ctx.bezierCurveTo(350, 200, 300, 190, 250, 170);
        ctx.bezierCurveTo(200, 150, 150, 160, 100, 140);
        ctx.closePath();
        ctx.fill();
        
        // North America - Temperate Forest
        ctx.fillStyle = '#228B22'; // Forest green
        ctx.beginPath();
        ctx.moveTo(150, 140);
        ctx.bezierCurveTo(200, 120, 250, 130, 300, 150);
        ctx.bezierCurveTo(350, 170, 400, 180, 450, 160);
        ctx.bezierCurveTo(500, 140, 550, 120, 600, 140);
        ctx.bezierCurveTo(650, 160, 700, 180, 750, 160);
        ctx.bezierCurveTo(800, 140, 850, 120, 900, 140);
        ctx.lineTo(900, 180);
        ctx.bezierCurveTo(850, 160, 800, 180, 750, 200);
        ctx.bezierCurveTo(700, 220, 650, 240, 600, 220);
        ctx.bezierCurveTo(550, 200, 500, 220, 450, 240);
        ctx.bezierCurveTo(400, 260, 350, 250, 300, 230);
        ctx.bezierCurveTo(250, 210, 200, 220, 150, 200);
        ctx.closePath();
        ctx.fill();
        
        // Europe - Mixed Forest
        ctx.fillStyle = '#32CD32'; // Lime green
        ctx.beginPath();
        ctx.moveTo(450, 120);
        ctx.bezierCurveTo(500, 100, 550, 110, 600, 130);
        ctx.bezierCurveTo(650, 150, 700, 160, 750, 140);
        ctx.bezierCurveTo(800, 120, 850, 100, 900, 120);
        ctx.lineTo(900, 160);
        ctx.bezierCurveTo(850, 140, 800, 160, 750, 180);
        ctx.bezierCurveTo(700, 200, 650, 190, 600, 170);
        ctx.bezierCurveTo(550, 150, 500, 160, 450, 180);
        ctx.closePath();
        ctx.fill();
        
        // Asia - Taiga and Temperate Forest
        ctx.fillStyle = '#006400'; // Dark green for taiga
        ctx.beginPath();
        ctx.moveTo(700, 60);
        ctx.bezierCurveTo(750, 40, 800, 50, 850, 70);
        ctx.bezierCurveTo(900, 90, 950, 100, 1000, 80);
        ctx.lineTo(1000, 120);
        ctx.bezierCurveTo(950, 100, 900, 120, 850, 140);
        ctx.bezierCurveTo(800, 160, 750, 150, 700, 130);
        ctx.closePath();
        ctx.fill();
        
        // Asia - Temperate Forest
        ctx.fillStyle = '#228B22'; // Forest green
        ctx.beginPath();
        ctx.moveTo(750, 130);
        ctx.bezierCurveTo(800, 110, 850, 120, 900, 140);
        ctx.bezierCurveTo(950, 160, 1000, 170, 1024, 150);
        ctx.lineTo(1024, 190);
        ctx.bezierCurveTo(1000, 170, 950, 190, 900, 210);
        ctx.bezierCurveTo(850, 230, 800, 220, 750, 200);
        ctx.closePath();
        ctx.fill();
        
        // South America - Amazon Rainforest
        ctx.fillStyle = '#006400'; // Dark green
        ctx.beginPath();
        ctx.moveTo(200, 300);
        ctx.bezierCurveTo(250, 280, 300, 290, 350, 310);
        ctx.bezierCurveTo(400, 330, 450, 340, 500, 320);
        ctx.bezierCurveTo(550, 300, 600, 280, 650, 300);
        ctx.lineTo(650, 340);
        ctx.bezierCurveTo(600, 320, 550, 340, 500, 360);
        ctx.bezierCurveTo(450, 380, 400, 370, 350, 350);
        ctx.bezierCurveTo(300, 330, 250, 340, 200, 360);
        ctx.closePath();
        ctx.fill();
        
        // Africa - Tropical and Subtropical
        ctx.fillStyle = '#228B22'; // Forest green
        ctx.beginPath();
        ctx.moveTo(500, 250);
        ctx.bezierCurveTo(550, 230, 600, 240, 650, 260);
        ctx.bezierCurveTo(700, 280, 750, 290, 800, 270);
        ctx.bezierCurveTo(850, 250, 900, 230, 950, 250);
        ctx.lineTo(950, 290);
        ctx.bezierCurveTo(900, 270, 850, 290, 800, 310);
        ctx.bezierCurveTo(750, 330, 700, 320, 650, 300);
        ctx.bezierCurveTo(600, 280, 550, 290, 500, 310);
        ctx.closePath();
        ctx.fill();
        
        // Australia - Eucalyptus and Desert
        ctx.fillStyle = '#9ACD32'; // Yellow green
        ctx.beginPath();
        ctx.moveTo(800, 350);
        ctx.bezierCurveTo(850, 330, 900, 340, 950, 360);
        ctx.bezierCurveTo(1000, 380, 1024, 390, 1024, 370);
        ctx.lineTo(1024, 410);
        ctx.bezierCurveTo(1000, 390, 950, 410, 900, 430);
        ctx.bezierCurveTo(850, 450, 800, 440, 800, 420);
        ctx.closePath();
        ctx.fill();
        
        // Add grassland areas
        ctx.fillStyle = '#9ACD32'; // Yellow green
        // North American Great Plains
        ctx.beginPath();
        ctx.moveTo(200, 200);
        ctx.bezierCurveTo(250, 180, 300, 190, 350, 210);
        ctx.bezierCurveTo(400, 230, 450, 240, 500, 220);
        ctx.bezierCurveTo(550, 200, 600, 180, 650, 200);
        ctx.lineTo(650, 240);
        ctx.bezierCurveTo(600, 220, 550, 240, 500, 260);
        ctx.bezierCurveTo(450, 280, 400, 270, 350, 250);
        ctx.bezierCurveTo(300, 230, 250, 240, 200, 260);
        ctx.closePath();
        ctx.fill();
        
        // African Savanna
        ctx.beginPath();
        ctx.moveTo(500, 320);
        ctx.bezierCurveTo(550, 300, 600, 310, 650, 330);
        ctx.bezierCurveTo(700, 350, 750, 360, 800, 340);
        ctx.bezierCurveTo(850, 320, 900, 300, 950, 320);
        ctx.lineTo(950, 360);
        ctx.bezierCurveTo(900, 340, 850, 360, 800, 380);
        ctx.bezierCurveTo(750, 400, 700, 390, 650, 370);
        ctx.bezierCurveTo(600, 350, 550, 360, 500, 380);
        ctx.closePath();
        ctx.fill();
        
        const texture = new THREE.CanvasTexture(canvas);
        material.map = texture;
        
        const vegetationMesh = new THREE.Mesh(geometry, material);
        this.scene.add(vegetationMesh);
        this.layers.vegetation.objects.push(vegetationMesh);
    }
    
    createPrecipitationLayer() {
        // Create precipitation visualization
        const geometry = new THREE.SphereGeometry(1.01, 32, 32);
        const material = new THREE.MeshBasicMaterial({
            transparent: true,
            opacity: 0.4,
            side: THREE.DoubleSide
        });
        
        // Create precipitation texture
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        
        // Base transparent
        ctx.clearRect(0, 0, 1024, 512);
        
        // Tropical Rain Belt (ITCZ)
        ctx.fillStyle = 'rgba(0, 100, 255, 0.7)'; // Blue for rain
        ctx.beginPath();
        ctx.moveTo(0, 200);
        ctx.bezierCurveTo(200, 180, 400, 190, 600, 200);
        ctx.bezierCurveTo(800, 210, 1000, 200, 1024, 190);
        ctx.lineTo(1024, 230);
        ctx.bezierCurveTo(1000, 220, 800, 230, 600, 220);
        ctx.bezierCurveTo(400, 210, 200, 220, 0, 230);
        ctx.closePath();
        ctx.fill();
        
        // Monsoon regions
        // Indian Monsoon
        ctx.fillStyle = 'rgba(0, 150, 255, 0.6)';
        ctx.beginPath();
        ctx.moveTo(600, 150);
        ctx.bezierCurveTo(650, 130, 700, 140, 750, 160);
        ctx.bezierCurveTo(800, 180, 850, 190, 900, 170);
        ctx.bezierCurveTo(950, 150, 1000, 130, 1024, 150);
        ctx.lineTo(1024, 190);
        ctx.bezierCurveTo(1000, 170, 950, 190, 900, 210);
        ctx.bezierCurveTo(850, 230, 800, 220, 750, 200);
        ctx.bezierCurveTo(700, 180, 650, 190, 600, 210);
        ctx.closePath();
        ctx.fill();
        
        // Southeast Asian Monsoon
        ctx.fillStyle = 'rgba(0, 120, 255, 0.5)';
        ctx.beginPath();
        ctx.moveTo(800, 200);
        ctx.bezierCurveTo(850, 180, 900, 190, 950, 210);
        ctx.bezierCurveTo(1000, 230, 1024, 240, 1024, 220);
        ctx.lineTo(1024, 260);
        ctx.bezierCurveTo(1000, 250, 950, 260, 900, 280);
        ctx.bezierCurveTo(850, 300, 800, 290, 800, 270);
        ctx.closePath();
        ctx.fill();
        
        // West African Monsoon
        ctx.fillStyle = 'rgba(0, 130, 255, 0.5)';
        ctx.beginPath();
        ctx.moveTo(400, 200);
        ctx.bezierCurveTo(450, 180, 500, 190, 550, 210);
        ctx.bezierCurveTo(600, 230, 650, 240, 700, 220);
        ctx.bezierCurveTo(750, 200, 800, 180, 850, 200);
        ctx.lineTo(850, 240);
        ctx.bezierCurveTo(800, 220, 750, 240, 700, 260);
        ctx.bezierCurveTo(650, 280, 600, 270, 550, 250);
        ctx.bezierCurveTo(500, 230, 450, 240, 400, 260);
        ctx.closePath();
        ctx.fill();
        
        // Amazon Rainforest precipitation
        ctx.fillStyle = 'rgba(0, 100, 255, 0.6)';
        ctx.beginPath();
        ctx.moveTo(200, 300);
        ctx.bezierCurveTo(250, 280, 300, 290, 350, 310);
        ctx.bezierCurveTo(400, 330, 450, 340, 500, 320);
        ctx.bezierCurveTo(550, 300, 600, 280, 650, 300);
        ctx.lineTo(650, 340);
        ctx.bezierCurveTo(600, 320, 550, 340, 500, 360);
        ctx.bezierCurveTo(450, 380, 400, 370, 350, 350);
        ctx.bezierCurveTo(300, 330, 250, 340, 200, 360);
        ctx.closePath();
        ctx.fill();
        
        // Mid-latitude storm tracks
        ctx.fillStyle = 'rgba(0, 80, 200, 0.4)';
        // North Atlantic storm track
        ctx.beginPath();
        ctx.moveTo(300, 100);
        ctx.bezierCurveTo(400, 80, 500, 90, 600, 110);
        ctx.bezierCurveTo(700, 130, 800, 140, 900, 120);
        ctx.lineTo(900, 160);
        ctx.bezierCurveTo(800, 140, 700, 150, 600, 170);
        ctx.bezierCurveTo(500, 190, 400, 180, 300, 160);
        ctx.closePath();
        ctx.fill();
        
        // North Pacific storm track
        ctx.beginPath();
        ctx.moveTo(100, 120);
        ctx.bezierCurveTo(200, 100, 300, 110, 400, 130);
        ctx.bezierCurveTo(500, 150, 600, 160, 700, 140);
        ctx.lineTo(700, 180);
        ctx.bezierCurveTo(600, 160, 500, 170, 400, 190);
        ctx.bezierCurveTo(300, 210, 200, 200, 100, 180);
        ctx.closePath();
        ctx.fill();
        
        // Southern Ocean storm tracks
        ctx.fillStyle = 'rgba(0, 60, 180, 0.3)';
        ctx.beginPath();
        ctx.moveTo(0, 400);
        ctx.bezierCurveTo(200, 380, 400, 390, 600, 410);
        ctx.bezierCurveTo(800, 430, 1000, 440, 1024, 420);
        ctx.lineTo(1024, 460);
        ctx.bezierCurveTo(1000, 440, 800, 450, 600, 470);
        ctx.bezierCurveTo(400, 490, 200, 480, 0, 460);
        ctx.closePath();
        ctx.fill();
        
        // Add some scattered precipitation areas
        for (let i = 0; i < 30; i++) {
            const x = Math.random() * 1024;
            const y = Math.random() * 512;
            const size = Math.random() * 40 + 15;
            const intensity = Math.random() * 0.4 + 0.1;
            
            ctx.fillStyle = `rgba(0, 100, 255, ${intensity})`;
            ctx.beginPath();
            ctx.ellipse(x, y, size, size * 0.6, 0, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        const texture = new THREE.CanvasTexture(canvas);
        material.map = texture;
        
        const precipitationMesh = new THREE.Mesh(geometry, material);
        this.scene.add(precipitationMesh);
        this.layers.precipitation.objects.push(precipitationMesh);
    }
    
    createElevationLayer() {
        // Create elevation visualization
        const geometry = new THREE.SphereGeometry(1.01, 32, 32);
        const material = new THREE.MeshBasicMaterial({
            transparent: true,
            opacity: 0.6,
            side: THREE.DoubleSide
        });
        
        // Create elevation texture
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        
        // Base ocean color
        ctx.fillStyle = '#000080';
        ctx.fillRect(0, 0, 1024, 512);
        
        // Rocky Mountains (North America)
        ctx.fillStyle = '#8B4513'; // Saddle brown
        ctx.beginPath();
        ctx.moveTo(200, 100);
        ctx.bezierCurveTo(250, 80, 300, 90, 350, 110);
        ctx.bezierCurveTo(400, 130, 450, 140, 500, 120);
        ctx.bezierCurveTo(550, 100, 600, 80, 650, 100);
        ctx.lineTo(650, 140);
        ctx.bezierCurveTo(600, 120, 550, 140, 500, 160);
        ctx.bezierCurveTo(450, 180, 400, 170, 350, 150);
        ctx.bezierCurveTo(300, 130, 250, 140, 200, 160);
        ctx.closePath();
        ctx.fill();
        
        // Appalachian Mountains
        ctx.fillStyle = '#A0522D'; // Sienna
        ctx.beginPath();
        ctx.moveTo(250, 150);
        ctx.bezierCurveTo(300, 130, 350, 140, 400, 160);
        ctx.bezierCurveTo(450, 180, 500, 190, 550, 170);
        ctx.bezierCurveTo(600, 150, 650, 130, 700, 150);
        ctx.lineTo(700, 190);
        ctx.bezierCurveTo(650, 170, 600, 190, 550, 210);
        ctx.bezierCurveTo(500, 230, 450, 220, 400, 200);
        ctx.bezierCurveTo(350, 180, 300, 190, 250, 210);
        ctx.closePath();
        ctx.fill();
        
        // Alps (Europe)
        ctx.fillStyle = '#8B4513'; // Saddle brown
        ctx.beginPath();
        ctx.moveTo(500, 120);
        ctx.bezierCurveTo(550, 100, 600, 110, 650, 130);
        ctx.bezierCurveTo(700, 150, 750, 160, 800, 140);
        ctx.bezierCurveTo(850, 120, 900, 100, 950, 120);
        ctx.lineTo(950, 160);
        ctx.bezierCurveTo(900, 140, 850, 160, 800, 180);
        ctx.bezierCurveTo(750, 200, 700, 190, 650, 170);
        ctx.bezierCurveTo(600, 150, 550, 160, 500, 180);
        ctx.closePath();
        ctx.fill();
        
        // Himalayas and Tibetan Plateau
        ctx.fillStyle = '#654321'; // Dark brown
        ctx.beginPath();
        ctx.moveTo(700, 100);
        ctx.bezierCurveTo(750, 80, 800, 90, 850, 110);
        ctx.bezierCurveTo(900, 130, 950, 140, 1000, 120);
        ctx.lineTo(1000, 160);
        ctx.bezierCurveTo(950, 140, 900, 160, 850, 180);
        ctx.bezierCurveTo(800, 200, 750, 190, 700, 170);
        ctx.closePath();
        ctx.fill();
        
        // Ural Mountains
        ctx.fillStyle = '#A0522D'; // Sienna
        ctx.beginPath();
        ctx.moveTo(600, 80);
        ctx.bezierCurveTo(650, 60, 700, 70, 750, 90);
        ctx.bezierCurveTo(800, 110, 850, 120, 900, 100);
        ctx.lineTo(900, 140);
        ctx.bezierCurveTo(850, 120, 800, 140, 750, 160);
        ctx.bezierCurveTo(700, 180, 650, 170, 600, 150);
        ctx.closePath();
        ctx.fill();
        
        // Andes Mountains (South America)
        ctx.fillStyle = '#8B4513'; // Saddle brown
        ctx.beginPath();
        ctx.moveTo(150, 250);
        ctx.bezierCurveTo(200, 230, 250, 240, 300, 260);
        ctx.bezierCurveTo(350, 280, 400, 290, 450, 270);
        ctx.bezierCurveTo(500, 250, 550, 230, 600, 250);
        ctx.lineTo(600, 290);
        ctx.bezierCurveTo(550, 270, 500, 290, 450, 310);
        ctx.bezierCurveTo(400, 330, 350, 320, 300, 300);
        ctx.bezierCurveTo(250, 280, 200, 290, 150, 310);
        ctx.closePath();
        ctx.fill();
        
        // African Highlands
        ctx.fillStyle = '#A0522D'; // Sienna
        ctx.beginPath();
        ctx.moveTo(500, 200);
        ctx.bezierCurveTo(550, 180, 600, 190, 650, 210);
        ctx.bezierCurveTo(700, 230, 750, 240, 800, 220);
        ctx.bezierCurveTo(850, 200, 900, 180, 950, 200);
        ctx.lineTo(950, 240);
        ctx.bezierCurveTo(900, 220, 850, 240, 800, 260);
        ctx.bezierCurveTo(750, 280, 700, 270, 650, 250);
        ctx.bezierCurveTo(600, 230, 550, 240, 500, 260);
        ctx.closePath();
        ctx.fill();
        
        // Australian Great Dividing Range
        ctx.fillStyle = '#A0522D'; // Sienna
        ctx.beginPath();
        ctx.moveTo(800, 350);
        ctx.bezierCurveTo(850, 330, 900, 340, 950, 360);
        ctx.bezierCurveTo(1000, 380, 1024, 390, 1024, 370);
        ctx.lineTo(1024, 410);
        ctx.bezierCurveTo(1000, 390, 950, 410, 900, 430);
        ctx.bezierCurveTo(850, 450, 800, 440, 800, 420);
        ctx.closePath();
        ctx.fill();
        
        // Add some highland areas
        ctx.fillStyle = '#CD853F'; // Peru
        // Tibetan Plateau
        ctx.beginPath();
        ctx.moveTo(750, 120);
        ctx.bezierCurveTo(800, 100, 850, 110, 900, 130);
        ctx.bezierCurveTo(950, 150, 1000, 160, 1024, 140);
        ctx.lineTo(1024, 180);
        ctx.bezierCurveTo(1000, 160, 950, 180, 900, 200);
        ctx.bezierCurveTo(850, 220, 800, 210, 750, 190);
        ctx.closePath();
        ctx.fill();
        
        // Add some volcanic regions
        ctx.fillStyle = '#B22222'; // Fire brick
        // Ring of Fire (Pacific)
        for (let i = 0; i < 8; i++) {
            const x = 100 + i * 100;
            const y = 200 + Math.sin(i * 0.5) * 50;
            ctx.beginPath();
            ctx.ellipse(x, y, 15, 10, 0, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        const texture = new THREE.CanvasTexture(canvas);
        material.map = texture;
        
        const elevationMesh = new THREE.Mesh(geometry, material);
        this.scene.add(elevationMesh);
        this.layers.elevation.objects.push(elevationMesh);
    }
    
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.renderer) {
            this.container.removeChild(this.renderer.domElement);
        }
        
        window.removeEventListener('resize', this.onWindowResize);
    }
}
