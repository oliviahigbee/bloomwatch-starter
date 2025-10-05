// BloomWatch Kids - Share Page JavaScript 📸

class ShareApp {
    constructor() {
        this.observations = [];
        this.achievements = {
            firstPhoto: false,
            globalExplorer: false,
            citizenScientist: false
        };
        this.photoCount = 0;
        this.photoGallery = [];
        
        this.init();
    }
    
    init() {
        this.loadAchievements();
        this.loadPhotoGallery();
        this.setupFormValidation();
    }
    
    setupFormValidation() {
        const form = document.getElementById('shareForm');
        if (form) {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.validateAndSubmitForm();
            });
        }
    }
    
    validateAndSubmitForm() {
        const requiredFields = ['plantName', 'plantType', 'plantPhoto', 'location'];
        let isValid = true;
        
        requiredFields.forEach(fieldId => {
            const field = document.getElementById(fieldId);
            if (!field || !field.value.trim()) {
                isValid = false;
                this.showFieldError(fieldId, 'This field is required!');
            } else {
                this.clearFieldError(fieldId);
            }
        });
        
        if (isValid) {
            this.submitObservation();
        } else {
            this.showNotification('Please fill in all required fields!', 'warning');
        }
    }
    
    showFieldError(fieldId, message) {
        const field = document.getElementById(fieldId);
        if (field) {
            field.classList.add('is-invalid');
            let errorDiv = field.parentNode.querySelector('.invalid-feedback');
            if (!errorDiv) {
                errorDiv = document.createElement('div');
                errorDiv.className = 'invalid-feedback';
                field.parentNode.appendChild(errorDiv);
            }
            errorDiv.textContent = message;
        }
    }
    
    clearFieldError(fieldId) {
        const field = document.getElementById(fieldId);
        if (field) {
            field.classList.remove('is-invalid');
            const errorDiv = field.parentNode.querySelector('.invalid-feedback');
            if (errorDiv) {
                errorDiv.remove();
            }
        }
    }
    
    previewPhoto(input) {
        const file = input.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const preview = document.getElementById('photoPreview');
                const previewImg = document.getElementById('previewImg');
                
                previewImg.src = e.target.result;
                preview.style.display = 'block';
                
                // Add some fun effects
                previewImg.style.animation = 'photoAppear 0.5s ease-in-out';
            };
            reader.readAsDataURL(file);
        }
    }
    
    removePhoto() {
        const fileInput = document.getElementById('plantPhoto');
        const preview = document.getElementById('photoPreview');
        
        if (fileInput) fileInput.value = '';
        if (preview) preview.style.display = 'none';
        
        this.showNotification('📸 Photo removed!', 'info');
    }
    
    submitObservation() {
        const formData = this.collectFormData();
        
        // Show loading state
        const submitBtn = document.querySelector('button[onclick="submitPhoto()"]');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>📸 Sharing...';
        submitBtn.disabled = true;
        
        // Simulate API call
        setTimeout(() => {
            this.processObservation(formData);
            this.resetForm();
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }, 2000);
    }
    
    collectFormData() {
        return {
            plantName: document.getElementById('plantName').value,
            plantType: document.getElementById('plantType').value,
            plantColor: document.getElementById('plantColor').value,
            yourName: document.getElementById('yourName').value || 'Anonymous Explorer',
            location: document.getElementById('location').value,
            plantStory: document.getElementById('plantStory').value,
            photo: document.getElementById('plantPhoto').files[0],
            timestamp: new Date().toISOString()
        };
    }
    
    processObservation(data) {
        // Add to local observations
        this.observations.push(data);
        
        // Update photo count
        this.photoCount++;
        
        // Update achievements
        this.updateAchievements();
        
        // Add to photo gallery
        this.addToPhotoGallery(data);
        
        // Show success message
        this.showSuccessMessage(data);
        
        // Save to localStorage
        this.saveObservations();
        this.saveAchievements();
    }
    
    updateAchievements() {
        if (!this.achievements.firstPhoto && this.photoCount >= 1) {
            this.achievements.firstPhoto = true;
            this.showAchievementUnlocked('📸 First Photo', 'You shared your first plant photo!');
        }
        
        if (!this.achievements.globalExplorer && this.photoCount >= 5) {
            this.achievements.globalExplorer = true;
            this.showAchievementUnlocked('🌍 Global Explorer', 'You shared 5 plant photos!');
        }
        
        if (!this.achievements.citizenScientist && this.photoCount >= 10) {
            this.achievements.citizenScientist = true;
            this.showAchievementUnlocked('🔬 Citizen Scientist', 'You shared 10 plant photos!');
        }
        
        this.updateAchievementDisplay();
    }
    
    addToPhotoGallery(data) {
        const photoData = {
            id: Date.now(),
            plantName: data.plantName,
            plantType: data.plantType,
            plantColor: data.plantColor,
            observerName: data.yourName,
            location: data.location,
            story: data.plantStory,
            timestamp: data.timestamp,
            imageUrl: data.photo ? URL.createObjectURL(data.photo) : this.getDefaultPlantImage(data.plantType)
        };
        
        this.photoGallery.unshift(photoData); // Add to beginning
        this.updatePhotoGalleryDisplay();
    }
    
    getDefaultPlantImage(plantType) {
        const defaultImages = {
            'flower': 'https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=300&h=200&fit=crop',
            'tree': 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=300&h=200&fit=crop',
            'bush': 'https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=300&h=200&fit=crop',
            'grass': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=300&h=200&fit=crop',
            'moss': 'https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=300&h=200&fit=crop',
            'fern': 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=300&h=200&fit=crop',
            'cactus': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=300&h=200&fit=crop',
            'other': 'https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=300&h=200&fit=crop'
        };
        return defaultImages[plantType] || defaultImages['other'];
    }
    
    updatePhotoGalleryDisplay() {
        const gallery = document.getElementById('photoGallery');
        if (!gallery) return;
        
        // Keep only the most recent 6 photos in display
        const recentPhotos = this.photoGallery.slice(0, 6);
        
        gallery.innerHTML = recentPhotos.map(photo => `
            <div class="col-md-6 col-lg-4 mb-4">
                <div class="card shadow photo-card">
                    <img src="${photo.imageUrl}" class="card-img-top" alt="${photo.plantName}" loading="lazy">
                    <div class="card-body">
                        <h6 class="card-title">${this.getPlantEmoji(photo.plantType)} ${photo.plantName}</h6>
                        <p class="card-text">${photo.story || 'Beautiful plant discovery!'}</p>
                        <div class="d-flex justify-content-between align-items-center">
                            <small class="text-muted">by ${photo.observerName}</small>
                            <small class="text-muted">📍 ${photo.location}</small>
                        </div>
                        <div class="mt-2">
                            <button class="btn btn-sm btn-outline-primary" onclick="shareApp.likePhoto(${photo.id})">
                                <i class="fas fa-heart me-1"></i>Like
                            </button>
                            <button class="btn btn-sm btn-outline-success" onclick="shareApp.sharePhoto(${photo.id})">
                                <i class="fas fa-share me-1"></i>Share
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    getPlantEmoji(plantType) {
        const emojis = {
            'flower': '🌸',
            'tree': '🌳',
            'bush': '🌿',
            'grass': '🌱',
            'moss': '🍀',
            'fern': '🌿',
            'cactus': '🌵',
            'other': '🌱'
        };
        return emojis[plantType] || '🌱';
    }
    
    likePhoto(photoId) {
        this.showNotification('❤️ Photo liked!', 'success');
        // In a real app, this would send a like to the server
    }
    
    sharePhoto(photoId) {
        const photo = this.photoGallery.find(p => p.id === photoId);
        if (photo) {
            if (navigator.share) {
                navigator.share({
                    title: `Check out this ${photo.plantName} I found!`,
                    text: `I found a beautiful ${photo.plantName} in ${photo.location}!`,
                    url: window.location.href
                });
            } else {
                // Fallback - copy to clipboard
                const shareText = `Check out this ${photo.plantName} I found in ${photo.location}! 🌱`;
                navigator.clipboard.writeText(shareText).then(() => {
                    this.showNotification('📋 Share text copied to clipboard!', 'info');
                });
            }
        }
    }
    
    loadMorePhotos() {
        this.showNotification('📸 Loading more photos...', 'info');
        // In a real app, this would load more photos from the server
        setTimeout(() => {
            this.showNotification('📸 More photos loaded!', 'success');
        }, 1500);
    }
    
    showSuccessMessage(data) {
        const message = `
            <div class="alert alert-success">
                <h5>🎉 Photo Shared Successfully!</h5>
                <p><strong>${data.plantName}</strong> has been shared with the world!</p>
                <p>📍 Location: ${data.location}</p>
                <p>👤 Shared by: ${data.yourName}</p>
                <p>Thank you for helping scientists learn about plants! 🌱✨</p>
            </div>
        `;
        
        // Show success message at the top of the form
        const form = document.getElementById('shareForm');
        const alertDiv = document.createElement('div');
        alertDiv.innerHTML = message;
        form.parentNode.insertBefore(alertDiv, form);
        
        // Remove the alert after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
        
        // Show confetti effect
        this.createConfetti();
    }
    
    createConfetti() {
        const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#28a745'];
        const confettiCount = 30;
        
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
    
    resetForm() {
        document.getElementById('shareForm').reset();
        document.getElementById('photoPreview').style.display = 'none';
        
        // Clear any validation errors
        const invalidFields = document.querySelectorAll('.is-invalid');
        invalidFields.forEach(field => {
            field.classList.remove('is-invalid');
        });
        
        const errorMessages = document.querySelectorAll('.invalid-feedback');
        errorMessages.forEach(error => error.remove());
    }
    
    loadPhotoGallery() {
        const saved = localStorage.getItem('bloomWatchPhotoGallery');
        if (saved) {
            this.photoGallery = JSON.parse(saved);
            this.updatePhotoGalleryDisplay();
        }
    }
    
    saveObservations() {
        localStorage.setItem('bloomWatchObservations', JSON.stringify(this.observations));
    }
    
    loadAchievements() {
        const saved = localStorage.getItem('bloomWatchShareAchievements');
        if (saved) {
            this.achievements = JSON.parse(saved);
        }
        this.updateAchievementDisplay();
    }
    
    saveAchievements() {
        localStorage.setItem('bloomWatchShareAchievements', JSON.stringify(this.achievements));
    }
    
    updateAchievementDisplay() {
        const achievements = document.querySelectorAll('.achievement-item');
        if (achievements.length >= 3) {
            // First Photo Achievement
            const firstPhotoProgress = achievements[0].querySelector('.progress-bar');
            const firstPhotoText = achievements[0].querySelector('small');
            if (this.achievements.firstPhoto) {
                firstPhotoProgress.style.width = '100%';
                firstPhotoText.textContent = 'Unlocked! 🎉';
                firstPhotoText.className = 'text-success';
            } else {
                firstPhotoProgress.style.width = '0%';
                firstPhotoText.textContent = '0/1 photos';
                firstPhotoText.className = 'text-muted';
            }
            
            // Global Explorer Achievement
            const globalExplorerProgress = achievements[1].querySelector('.progress-bar');
            const globalExplorerText = achievements[1].querySelector('small');
            if (this.achievements.globalExplorer) {
                globalExplorerProgress.style.width = '100%';
                globalExplorerText.textContent = 'Unlocked! 🎉';
                globalExplorerText.className = 'text-success';
            } else {
                const progress = Math.min((this.photoCount / 5) * 100, 100);
                globalExplorerProgress.style.width = `${progress}%`;
                globalExplorerText.textContent = `${this.photoCount}/5 photos`;
                globalExplorerText.className = 'text-muted';
            }
            
            // Citizen Scientist Achievement
            const citizenScientistProgress = achievements[2].querySelector('.progress-bar');
            const citizenScientistText = achievements[2].querySelector('small');
            if (this.achievements.citizenScientist) {
                citizenScientistProgress.style.width = '100%';
                citizenScientistText.textContent = 'Unlocked! 🎉';
                citizenScientistText.className = 'text-success';
            } else {
                const progress = Math.min((this.photoCount / 10) * 100, 100);
                citizenScientistProgress.style.width = `${progress}%`;
                citizenScientistText.textContent = `${this.photoCount}/10 photos`;
                citizenScientistText.className = 'text-muted';
            }
        }
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
        
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
        
        // Add confetti effect
        this.createConfetti();
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
        
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 3000);
    }
}

// Global functions for HTML onclick events
function previewPhoto(input) {
    shareApp.previewPhoto(input);
}

function removePhoto() {
    shareApp.removePhoto();
}

function submitPhoto() {
    shareApp.submitObservation();
}

function loadMorePhotos() {
    shareApp.loadMorePhotos();
}

// Initialize the app when the page loads
let shareApp;
document.addEventListener('DOMContentLoaded', () => {
    shareApp = new ShareApp();
});

// Add CSS animations and styles
const style = document.createElement('style');
style.textContent = `
    @keyframes photoAppear {
        from { opacity: 0; transform: scale(0.8); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes confettiFall {
        to {
            transform: translateY(100vh) rotate(360deg);
            opacity: 0;
        }
    }
    
    .photo-card {
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .photo-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15) !important;
    }
    
    .photo-card img {
        transition: transform 0.3s ease;
    }
    
    .photo-card:hover img {
        transform: scale(1.05);
    }
    
    .achievement-item {
        transition: all 0.3s ease;
    }
    
    .achievement-item:hover {
        transform: translateX(5px);
    }
    
    .form-control:focus {
        border-color: #28a745;
        box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.25);
    }
    
    .form-select:focus {
        border-color: #28a745;
        box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.25);
    }
    
    .btn-success:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(40, 167, 69, 0.3);
    }
    
    .card-img-top {
        object-fit: cover;
        height: 200px;
    }
    
    .invalid-feedback {
        display: block;
    }
    
    .is-invalid {
        border-color: #dc3545;
    }
    
    .is-invalid:focus {
        border-color: #dc3545;
        box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.25);
    }
`;
document.head.appendChild(style);
