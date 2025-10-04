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
        
        this.init();
    }
    
    init() {
        this.initializeMap();
        this.initializeChart();
        this.loadCities();
        this.loadInitialData();
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
        
        window.updateVegetationIndex = () => {
            this.currentVegetationIndex = document.getElementById('vegetationIndex').value;
            this.refreshData();
        };
        
        window.refreshData = () => {
            this.loadInitialData();
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
        const response = await fetch(`/api/trends?location=${this.currentLocation}&years=${this.currentTimeRange}`);
        if (!response.ok) {
            throw new Error('Failed to fetch trends');
        }
        return await response.json();
    }
    
    async fetchConservationInsights() {
        const response = await fetch(`/api/conservation-insights?location=${this.currentLocation}`, {
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
        } else {
            this.map.setView([20, 0], 2);
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
        document.getElementById('currentIntensity').textContent = latestData[this.currentVegetationIndex].toFixed(3);
        document.getElementById('intensityBar').style.width = `${latestData[this.currentVegetationIndex] * 100}%`;
        
        // Update peak date
        document.getElementById('peakDate').textContent = peakDate;
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
