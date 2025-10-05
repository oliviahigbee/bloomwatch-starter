// BloomWatch Kids - Explore Page JavaScript 🌍

class ExploreApp {
    constructor() {
        this.map = null;
        this.currentData = null;
        this.currentLocation = 'global';
        this.currentTimeRange = 5;
        this.currentVegetationIndex = 'ndvi';
        this.funFacts = [
            "🌱 Did you know? Plants make their own food using sunlight!",
            "🌍 Our Earth has over 400,000 different types of plants!",
            "🚀 NASA satellites can see plants from space!",
            "🌸 Some flowers can bloom in just one day!",
            "🌳 Trees can live for thousands of years!",
            "🌿 Plants help clean the air we breathe!",
            "🦋 Many plants need animals to help them grow!",
            "🌧️ Rain helps plants grow big and strong!"
        ];
        this.currentFactIndex = 0;
        this.achievements = {
            worldExplorer: 0,
            mapNavigator: 0
        };
        
        this.init();
    }
    
    init() {
        this.initializeMap();
        this.loadFunFacts();
        this.startFactRotation();
        this.loadAchievements();
        this.initializeChart();
        this.loadInitialNASAData();
    }
    
    initializeMap() {
        // Initialize the map centered on the world
        this.map = L.map('globalMap').setView([20, 0], 2);
        
        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 18
        }).addTo(this.map);
        
        // Add some sample plant data points
        this.addSamplePlantData();
        
        // Add click event listener
        this.map.on('click', (e) => {
            this.showPlantInfo(e.latlng);
        });
    }
    
    addSamplePlantData() {
        const plantData = [
            { lat: 40.7128, lng: -74.0060, name: "New York Cherry Blossoms", type: "🌸 Cherry Blossom", health: "Good" },
            { lat: 51.5074, lng: -0.1278, name: "London Roses", type: "🌹 Rose Garden", health: "Excellent" },
            { lat: 35.6762, lng: 139.6503, name: "Tokyo Sakura", type: "🌸 Cherry Blossom", health: "Amazing" },
            { lat: -23.5505, lng: -46.6333, name: "São Paulo Orchids", type: "🌺 Orchids", health: "Good" },
            { lat: -33.8688, lng: 151.2093, name: "Sydney Eucalyptus", type: "🌳 Eucalyptus", health: "Excellent" },
            { lat: -33.9249, lng: 18.4241, name: "Cape Town Proteas", type: "🌸 Protea", health: "Good" },
            { lat: 19.0760, lng: 72.8777, name: "Mumbai Marigolds", type: "🌼 Marigold", health: "Excellent" },
            { lat: 48.8566, lng: 2.3522, name: "Paris Tulips", type: "🌷 Tulips", health: "Good" },
            { lat: 34.0522, lng: -118.2437, name: "LA Poppies", type: "🌺 California Poppy", health: "Amazing" },
            { lat: -34.6118, lng: -58.3960, name: "Buenos Aires Jacaranda", type: "💜 Jacaranda", health: "Excellent" }
        ];
        
        plantData.forEach(plant => {
            const color = this.getHealthColor(plant.health);
            const marker = L.circleMarker([plant.lat, plant.lng], {
                radius: 8,
                fillColor: color,
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(this.map);
            
            marker.bindPopup(`
                <div class="plant-popup">
                    <h6>${plant.type}</h6>
                    <p><strong>Location:</strong> ${plant.name}</p>
                    <p><strong>Health:</strong> ${plant.health}</p>
                    <p><strong>Status:</strong> 🌱 Growing well!</p>
                </div>
            `);
            
            // Add click event to marker
            marker.on('click', () => {
                this.updateAchievement('mapNavigator');
            });
        });
    }
    
    getHealthColor(health) {
        switch(health) {
            case 'Amazing': return '#28a745';
            case 'Excellent': return '#20c997';
            case 'Good': return '#17a2b8';
            case 'Fair': return '#ffc107';
            case 'Poor': return '#fd7e14';
            default: return '#6c757d';
        }
    }
    
    async showPlantInfo(latlng) {
        const lat = latlng.lat.toFixed(4);
        const lng = latlng.lng.toFixed(4);
        
        // Show loading popup
        const loadingPopup = L.popup()
            .setLatLng(latlng)
            .setContent(`
                <div class="plant-info-popup">
                    <h6>🛰️ Getting NASA Data...</h6>
                    <p><i class="fas fa-spinner fa-spin"></i> Loading real satellite data from space!</p>
                </div>
            `)
            .openOn(this.map);
        
        try {
            // Get real NASA data
            const response = await fetch(`/api/bloom-data?lat=${lat}&lon=${lng}`);
            const data = await response.json();
            
            // Create popup with real NASA data
            const popup = L.popup()
                .setLatLng(latlng)
                .setContent(`
                    <div class="plant-info-popup">
                        <h6>🛰️ NASA Satellite Data</h6>
                        <p><strong>Location:</strong> ${lat}, ${lng}</p>
                        <p><strong>Data Source:</strong> ${data.data_availability}</p>
                        ${data.satellite_imagery ? `
                            <p><strong>🛰️ Satellite:</strong> ${data.satellite_imagery.data_source}</p>
                            <p><strong>📅 Date:</strong> ${data.satellite_imagery.date}</p>
                        ` : ''}
                        ${data.data && data.data.length > 0 ? `
                            <p><strong>🌱 Plant Health:</strong> ${this.getHealthStatus(data.data[0])}</p>
                        ` : ''}
                        <p><strong>🚀 Fun Fact:</strong> This data comes from NASA satellites in space!</p>
                        <button class="btn btn-sm btn-success" onclick="exploreApp.addCustomMarker(${lat}, ${lng})">
                            🌱 Add Plant Here!
                        </button>
                    </div>
                `)
                .openOn(this.map);
            
            // Update achievements
            this.updateAchievement('worldExplorer');
            
            // Update the plant health scoreboard with real data
            this.updatePlantHealthScoreboard(lat, lng, data);
            
            // Show NASA data prominently
            this.showNASAData(data, lat, lng);
            
            // Update the chart with real NASA data
            this.updateChart(data);
            
        } catch (error) {
            console.error('Error fetching NASA data:', error);
            // Fallback popup
            const popup = L.popup()
                .setLatLng(latlng)
                .setContent(`
                    <div class="plant-info-popup">
                        <h6>🌱 Plant Discovery!</h6>
                        <p>You found a new location!</p>
                        <p><strong>Coordinates:</strong><br>
                        Latitude: ${lat}<br>
                        Longitude: ${lng}</p>
                        <button class="btn btn-sm btn-success" onclick="exploreApp.addCustomMarker(${lat}, ${lng})">
                            🌱 Add Plant Here!
                        </button>
                    </div>
                `)
                .openOn(this.map);
        }
    }
    
    getHealthStatus(dataPoint) {
        if (dataPoint.ndvi > 0.7) return '🌿 Excellent';
        if (dataPoint.ndvi > 0.5) return '🌱 Good';
        if (dataPoint.ndvi > 0.3) return '🍃 Fair';
        return '🌾 Needs Attention';
    }
    
    addCustomMarker(lat, lng) {
        const marker = L.marker([lat, lng]).addTo(this.map);
        marker.bindPopup(`
            <div class="custom-plant-popup">
                <h6>🌱 Your Plant Discovery!</h6>
                <p>Great job finding this location!</p>
                <p>Click "Share Photo" to add a real plant photo!</p>
                <a href="/share" class="btn btn-sm btn-primary">📸 Share Photo</a>
            </div>
        `);
        
        this.showNotification("🌱 Great discovery! You found a new plant location!", "success");
    }
    
    async updateCity() {
        const citySelect = document.getElementById('citySelect');
        this.currentLocation = citySelect.value;
        this.updateMapTitle();
        this.showNotification(`🌍 Now exploring ${citySelect.options[citySelect.selectedIndex].text}!`, "info");
        
        // Load NASA data for the selected city
        await this.loadNASADataForCity(this.currentLocation);
    }
    
    async loadNASADataForCity(city) {
        const cityCoordinates = {
            'global': { lat: 20, lon: 0 },
            'new-york': { lat: 40.7128, lon: -74.0060 },
            'london': { lat: 51.5074, lon: -0.1278 },
            'tokyo': { lat: 35.6762, lon: 139.6503 },
            'sao-paulo': { lat: -23.5505, lon: -46.6333 },
            'sydney': { lat: -33.8688, lon: 151.2093 },
            'cape-town': { lat: -33.9249, lon: 18.4241 },
            'mumbai': { lat: 19.0760, lon: 72.8777 },
            'paris': { lat: 48.8566, lon: 2.3522 },
            'los-angeles': { lat: 34.0522, lon: -118.2437 }
        };
        
        const coords = cityCoordinates[city] || cityCoordinates['new-york'];
        
        try {
            this.showNotification("🛰️ Loading NASA data for selected location...", "info");
            
            const response = await fetch(`/api/bloom-data?lat=${coords.lat}&lon=${coords.lon}`);
            const nasaData = await response.json();
            
            if (nasaData && nasaData.data && nasaData.data.length > 0) {
                this.updateChart(nasaData);
                this.updatePlantHealthScoreboard(coords.lat, coords.lon, nasaData);
                this.showNASAData(nasaData, coords.lat, coords.lon);
                
                this.showNotification(
                    `🛰️ NASA data loaded for ${city}! Source: ${nasaData.data_availability}`,
                    'success'
                );
            }
        } catch (error) {
            console.error('Error loading NASA data for city:', error);
            this.showNotification("⚠️ Error loading NASA data for selected city", "warning");
        }
    }
    
    updateTimeRange() {
        const timeSelect = document.getElementById('timeRange');
        this.currentTimeRange = parseInt(timeSelect.value);
        this.showNotification(`⏰ Looking at ${timeSelect.options[timeSelect.selectedIndex].text}!`, "info");
    }
    
    updateVegetationIndex() {
        const vegSelect = document.getElementById('vegetationIndex');
        this.currentVegetationIndex = vegSelect.value;
        this.showNotification(`🌿 Now showing ${vegSelect.options[vegSelect.selectedIndex].text}!`, "info");
    }
    
    updateMapTitle() {
        const citySelect = document.getElementById('citySelect');
        const selectedText = citySelect.options[citySelect.selectedIndex].text;
        document.getElementById('mapTitle').textContent = `🌍 ${selectedText} Plant Map`;
    }
    
    async refreshData() {
        this.showNotification("🔄 Getting fresh plant data from NASA satellites...", "info");
        
        try {
            // Reload NASA data for the current location
            await this.loadInitialNASAData();
            this.loadFunFacts();
            this.showNotification("✅ Fresh NASA satellite data loaded successfully!", "success");
        } catch (error) {
            console.error('Error refreshing data:', error);
            this.showNotification("⚠️ Error loading fresh data - using cached data", "warning");
        }
    }
    
    async loadFunFacts() {
        const container = document.getElementById('funFactsContainer');
        
        // Try to get NASA space facts
        try {
            const response = await fetch('/api/nasa-space-facts');
            const nasaFact = await response.json();
            
            if (nasaFact.data_availability === 'real_nasa_data') {
                container.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-rocket me-2"></i>
                        <strong>🚀 NASA Space Fact:</strong> ${nasaFact.space_fact.fact}
                        <br><small class="text-muted">${nasaFact.space_fact.emoji} ${nasaFact.space_fact.related_to_blooms ? 'Related to plants!' : ''}</small>
                    </div>
                `;
                return;
            }
        } catch (error) {
            console.log('NASA space facts not available, using local facts');
        }
        
        // Fallback to local facts
        const currentFact = this.funFacts[this.currentFactIndex];
        container.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-lightbulb me-2"></i>
                <strong>Fun Fact:</strong> ${currentFact}
            </div>
        `;
    }
    
    showNextFact() {
        this.currentFactIndex = (this.currentFactIndex + 1) % this.funFacts.length;
        this.loadFunFacts();
        
        // Add animation
        const container = document.getElementById('funFactsContainer');
        container.style.animation = 'slideIn 0.5s ease-in-out';
        setTimeout(() => {
            container.style.animation = '';
        }, 500);
    }
    
    startFactRotation() {
        setInterval(() => {
            this.showNextFact();
        }, 10000); // Rotate every 10 seconds
    }
    
    loadAchievements() {
        this.updateAchievementDisplay();
    }
    
    updateAchievement(achievementType) {
        if (achievementType === 'worldExplorer') {
            this.achievements.worldExplorer++;
            if (this.achievements.worldExplorer === 1) {
                this.showAchievementUnlocked("🌍 World Explorer", "You explored your first location!");
            }
        } else if (achievementType === 'mapNavigator') {
            this.achievements.mapNavigator++;
            if (this.achievements.mapNavigator === 3) {
                this.showAchievementUnlocked("🗺️ Map Navigator", "You clicked on 3 map locations!");
            }
        }
        
        this.updateAchievementDisplay();
    }
    
    updateAchievementDisplay() {
        const worldExplorerProgress = Math.min((this.achievements.worldExplorer / 1) * 100, 100);
        const mapNavigatorProgress = Math.min((this.achievements.mapNavigator / 3) * 100, 100);
        
        document.querySelector('.achievement-item:nth-child(1) .progress-bar').style.width = `${worldExplorerProgress}%`;
        document.querySelector('.achievement-item:nth-child(2) .progress-bar').style.width = `${mapNavigatorProgress}%`;
    }
    
    showAchievementUnlocked(title, description) {
        const notification = document.createElement('div');
        notification.className = 'alert alert-success alert-dismissible fade show position-fixed';
        notification.style.cssText = 'top: 100px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            <h6>🏆 Achievement Unlocked!</h6>
            <strong>${title}</strong><br>
            ${description}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
        
        // Add confetti effect
        this.createConfetti();
    }
    
    createConfetti() {
        const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'];
        const confettiCount = 50;
        
        for (let i = 0; i < confettiCount; i++) {
            setTimeout(() => {
                const confetti = document.createElement('div');
                confetti.style.cssText = `
                    position: fixed;
                    width: 10px;
                    height: 10px;
                    background: ${colors[Math.floor(Math.random() * colors.length)]};
                    top: -10px;
                    left: ${Math.random() * 100}%;
                    z-index: 10000;
                    animation: confettiFall 3s linear forwards;
                `;
                
                document.body.appendChild(confetti);
                
                setTimeout(() => {
                    confetti.remove();
                }, 3000);
            }, i * 10);
        }
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 3000);
    }
    
    updatePlantHealthScoreboard(lat, lng, nasaData = null) {
        // Update the plant health scoreboard with real NASA data
        if (nasaData && nasaData.data && nasaData.data.length > 0) {
            const latestData = nasaData.data[nasaData.data.length - 1];
            const healthScore = Math.round((latestData.ndvi || 0.5) * 100);
            
            // Update the intensity bar
            const intensityBar = document.getElementById('intensityBar');
            const currentIntensity = document.getElementById('currentIntensity');
            if (intensityBar && currentIntensity) {
                intensityBar.style.width = `${healthScore}%`;
                currentIntensity.textContent = `${healthScore}%`;
                
                // Update color based on health
                if (healthScore >= 70) {
                    intensityBar.className = 'progress-bar bg-success';
                } else if (healthScore >= 50) {
                    intensityBar.className = 'progress-bar bg-warning';
                } else {
                    intensityBar.className = 'progress-bar bg-danger';
                }
            }
            
            // Update peak date
            const peakDate = document.getElementById('peakDate');
            if (peakDate) {
                const peakData = nasaData.data.reduce((max, item) => item.ndvi > max.ndvi ? item : max);
                peakDate.textContent = peakData.date || 'Today';
            }
            
            // Update trend
            const trend = document.getElementById('trend');
            if (trend && nasaData.data.length > 1) {
                const first = nasaData.data[0].ndvi;
                const last = nasaData.data[nasaData.data.length - 1].ndvi;
                if (last > first) {
                    trend.textContent = '📈 Growing!';
                } else if (last < first) {
                    trend.textContent = '📉 Declining';
                } else {
                    trend.textContent = '➡️ Stable';
                }
            }
            
            // Update peak season
            const peakSeason = document.getElementById('peakSeason');
            if (peakSeason) {
                peakSeason.textContent = nasaData.satellite_imagery ? 
                    `🛰️ Real NASA Data (${nasaData.satellite_imagery.date})` : 
                    '🌱 Spring';
            }
            
            // Show NASA data source
            this.showNotification(
                `🛰️ Real NASA satellite data loaded for ${lat}, ${lng}! Data source: ${nasaData.data_availability}`,
                'success'
            );
        } else {
            // Fallback to default values
            const intensityBar = document.getElementById('intensityBar');
            const currentIntensity = document.getElementById('currentIntensity');
            if (intensityBar && currentIntensity) {
                intensityBar.style.width = '75%';
                currentIntensity.textContent = '75%';
                intensityBar.className = 'progress-bar bg-success';
            }
            
            // Set default values for other metrics
            const peakDate = document.getElementById('peakDate');
            if (peakDate) peakDate.textContent = 'Today';
            
            const trend = document.getElementById('trend');
            if (trend) trend.textContent = '📈 Growing!';
            
            const peakSeason = document.getElementById('peakSeason');
            if (peakSeason) peakSeason.textContent = '🌱 Spring';
        }
    }
    
    showNASAData(data, lat, lng) {
        // Show NASA data prominently on the page
        const nasaCard = document.getElementById('nasaDataCard');
        const nasaContent = document.getElementById('nasaDataContent');
        
        if (nasaCard && nasaContent) {
            let content = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>🛰️ NASA Data Source</h6>
                        <p><strong>Status:</strong> ${data.data_availability}</p>
                        <p><strong>Location:</strong> ${lat}, ${lng}</p>
                        ${data.satellite_imagery ? `
                            <p><strong>Satellite:</strong> ${data.satellite_imagery.data_source}</p>
                            <p><strong>Product:</strong> ${data.satellite_imagery.product}</p>
                            <p><strong>Date:</strong> ${data.satellite_imagery.date}</p>
                        ` : ''}
                    </div>
                    <div class="col-md-6">
                        <h6>🌱 Plant Health Data</h6>
                        ${data.data && data.data.length > 0 ? `
                            <p><strong>Latest NDVI:</strong> ${data.data[data.data.length - 1].ndvi.toFixed(3)}</p>
                            <p><strong>Health Status:</strong> ${this.getHealthStatus(data.data[data.data.length - 1])}</p>
                            <p><strong>Data Points:</strong> ${data.data.length}</p>
                        ` : '<p>No plant data available</p>'}
                    </div>
                </div>
            `;
            
            nasaContent.innerHTML = content;
            nasaCard.style.display = 'block';
            
            // Scroll to the NASA data section
            nasaCard.scrollIntoView({ behavior: 'smooth' });
        }
    }
    
    initializeChart() {
        // Initialize the time series chart with sample data
        const ctx = document.getElementById('timeSeriesChart');
        if (ctx) {
            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    datasets: [{
                        label: 'Plant Health (NDVI)',
                        data: [0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: '🛰️ NASA Satellite Plant Health Data',
                            font: {
                                size: 16,
                                weight: 'bold'
                            }
                        },
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            title: {
                                display: true,
                                text: 'NDVI (Plant Health Index)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Month'
                            }
                        }
                    }
                }
            });
            
            this.chart = chart;
        }
    }
    
    updateChart(nasaData) {
        // Update the chart with real NASA data
        if (this.chart && nasaData && nasaData.data && nasaData.data.length > 0) {
            const dates = nasaData.data.map(d => d.date || 'Unknown');
            const ndviValues = nasaData.data.map(d => d.ndvi || 0);
            
            this.chart.data.labels = dates;
            this.chart.data.datasets[0].data = ndviValues;
            this.chart.update();
        }
    }
    
    async loadInitialNASAData() {
        // Load initial NASA data for a default location to populate the chart
        try {
            // Use New York City as default location for initial data
            const defaultLat = 40.7128;
            const defaultLon = -74.0060;
            
            this.showNotification("🛰️ Loading real NASA satellite data...", "info");
            
            const response = await fetch(`/api/bloom-data?lat=${defaultLat}&lon=${defaultLon}`);
            const nasaData = await response.json();
            
            if (nasaData && nasaData.data && nasaData.data.length > 0) {
                // Update the chart with real NASA data
                this.updateChart(nasaData);
                
                // Update the plant health scoreboard
                this.updatePlantHealthScoreboard(defaultLat, defaultLon, nasaData);
                
                // Show NASA data prominently
                this.showNASAData(nasaData, defaultLat, defaultLon);
                
                this.showNotification(
                    `🛰️ Real NASA data loaded! Source: ${nasaData.data_availability}`,
                    'success'
                );
            } else {
                this.showNotification("⚠️ Using sample data - NASA API temporarily unavailable", "warning");
            }
        } catch (error) {
            console.error('Error loading initial NASA data:', error);
            this.showNotification("⚠️ Using sample data - NASA API temporarily unavailable", "warning");
        }
    }
}

// Global functions for HTML onclick events
function updateCity() {
    exploreApp.updateCity();
}

function updateTimeRange() {
    exploreApp.updateTimeRange();
}

function updateVegetationIndex() {
    exploreApp.updateVegetationIndex();
}

function refreshData() {
    exploreApp.refreshData();
}

function showNextFact() {
    exploreApp.showNextFact();
}

// Initialize the app when the page loads
let exploreApp;
document.addEventListener('DOMContentLoaded', () => {
    exploreApp = new ExploreApp();
    // Make exploreApp globally accessible for debugging and HTML onclick events
    window.exploreApp = exploreApp;
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes confettiFall {
        to {
            transform: translateY(100vh) rotate(360deg);
            opacity: 0;
        }
    }
    
    .plant-popup {
        font-family: 'Comic Neue', cursive;
    }
    
    .plant-popup h6 {
        color: #28a745;
        margin-bottom: 10px;
    }
    
    .custom-plant-popup {
        font-family: 'Comic Neue', cursive;
    }
    
    .custom-plant-popup h6 {
        color: #007bff;
        margin-bottom: 10px;
    }
    
    .achievement-item {
        transition: all 0.3s ease;
    }
    
    .achievement-item:hover {
        transform: translateY(-2px);
    }
`;
document.head.appendChild(style);
