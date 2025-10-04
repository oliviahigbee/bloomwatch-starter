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
            predictionCard.innerHTML = `
                <div class="card-header bg-primary text-white">
                    <h6 class="mb-0"><i class="fas fa-brain me-2"></i>AI Bloom Prediction</h6>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-6">
                            <h4 class="text-primary">${(prediction.predicted_intensity * 100).toFixed(1)}%</h4>
                            <small class="text-muted">Predicted Intensity</small>
                        </div>
                        <div class="col-6">
                            <h4 class="text-success">${(prediction.confidence * 100).toFixed(1)}%</h4>
                            <small class="text-muted">Confidence</small>
                        </div>
                    </div>
                    <hr>
                    <p><strong>Model:</strong> ${prediction.model_used}</p>
                    <p><strong>Timeframe:</strong> ${prediction.days_ahead} days ahead</p>
                    <div class="progress">
                        <div class="progress-bar bg-primary" style="width: ${prediction.predicted_intensity * 100}%"></div>
                    </div>
                </div>
            `;
            predictionCard.style.display = 'block';
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
