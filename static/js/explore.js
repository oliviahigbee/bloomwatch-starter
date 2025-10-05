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
    
    showPlantInfo(latlng) {
        const popup = L.popup()
            .setLatLng(latlng)
            .setContent(`
                <div class="plant-info-popup">
                    <h6>🌱 Plant Discovery!</h6>
                    <p>You found a new location!</p>
                    <p><strong>Coordinates:</strong><br>
                    Latitude: ${latlng.lat.toFixed(4)}<br>
                    Longitude: ${latlng.lng.toFixed(4)}</p>
                    <button class="btn btn-sm btn-success" onclick="exploreApp.addCustomMarker(${latlng.lat}, ${latlng.lng})">
                        🌱 Add Plant Here!
                    </button>
                </div>
            `)
            .openOn(this.map);
            
        this.updateAchievement('worldExplorer');
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
    
    updateCity() {
        const citySelect = document.getElementById('citySelect');
        this.currentLocation = citySelect.value;
        this.updateMapTitle();
        this.showNotification(`🌍 Now exploring ${citySelect.options[citySelect.selectedIndex].text}!`, "info");
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
    
    refreshData() {
        this.showNotification("🔄 Getting fresh plant data from NASA satellites...", "info");
        
        // Simulate data loading
        setTimeout(() => {
            this.showNotification("✅ Fresh plant data loaded successfully!", "success");
            this.loadFunFacts();
        }, 2000);
    }
    
    loadFunFacts() {
        const container = document.getElementById('funFactsContainer');
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
