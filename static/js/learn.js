// BloomWatch Kids - Learn Page JavaScript 🎓

class LearnApp {
    constructor() {
        this.currentLesson = null;
        this.lessons = {
            plants: {
                title: "🌱 What Are Plants?",
                content: `
                    <div class="lesson-content">
                        <h4>🌱 Amazing Plants All Around Us!</h4>
                        <p>Plants are living things that grow from the ground! They come in all shapes and sizes.</p>
                        
                        <div class="row mt-4">
                            <div class="col-md-6">
                                <h5>🌸 Types of Plants:</h5>
                                <ul class="list-group">
                                    <li class="list-group-item">🌹 Flowers - Beautiful and colorful!</li>
                                    <li class="list-group-item">🌳 Trees - Tall and strong!</li>
                                    <li class="list-group-item">🌿 Grass - Soft and green!</li>
                                    <li class="list-group-item">🌱 Bushes - Medium sized plants!</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h5>🌍 Where Plants Live:</h5>
                                <ul class="list-group">
                                    <li class="list-group-item">🏞️ In gardens and parks</li>
                                    <li class="list-group-item">🌲 In forests and woods</li>
                                    <li class="list-group-item">🏠 In our homes</li>
                                    <li class="list-group-item">🏔️ Even on mountains!</li>
                                </ul>
                            </div>
                        </div>
                        
                        <div class="alert alert-success mt-4">
                            <h5>💡 Fun Fact!</h5>
                            <p>Plants don't move around like animals, but they can grow toward sunlight!</p>
                        </div>
                    </div>
                `
            },
            photosynthesis: {
                title: "☀️ How Plants Make Food",
                content: `
                    <div class="lesson-content">
                        <h4>☀️ Plants Are Like Magic Factories!</h4>
                        <p>Plants make their own food using sunlight, water, and air! This is called photosynthesis.</p>
                        
                        <div class="row mt-4">
                            <div class="col-md-4">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h1 class="text-warning">☀️</h1>
                                        <h5>Sunlight</h5>
                                        <p>Plants use sunlight as energy!</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h1 class="text-info">💧</h1>
                                        <h5>Water</h5>
                                        <p>Plants drink water from the ground!</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h1 class="text-primary">💨</h1>
                                        <h5>Air (CO₂)</h5>
                                        <p>Plants breathe in carbon dioxide!</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="alert alert-info mt-4">
                            <h5>🌟 The Magic Process:</h5>
                            <ol>
                                <li>🌱 Plant takes in sunlight, water, and air</li>
                                <li>🍃 Leaves turn them into food (sugar)</li>
                                <li>🌿 Plant grows bigger and stronger</li>
                                <li>💨 Plant gives us fresh air to breathe!</li>
                            </ol>
                        </div>
                    </div>
                `
            },
            seasons: {
                title: "🌸 Seasons & Flowers",
                content: `
                    <div class="lesson-content">
                        <h4>🌸 Why Do Flowers Bloom at Different Times?</h4>
                        <p>Flowers are smart! They bloom when the weather is just right for them!</p>
                        
                        <div class="row mt-4">
                            <div class="col-md-3">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h1 class="text-success">🌸</h1>
                                        <h5>Spring</h5>
                                        <p>Tulips, Daffodils, Cherry Blossoms</p>
                                        <small class="text-muted">March - May</small>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h1 class="text-warning">🌻</h1>
                                        <h5>Summer</h5>
                                        <p>Sunflowers, Roses, Lilies</p>
                                        <small class="text-muted">June - August</small>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h1 class="text-danger">🍂</h1>
                                        <h5>Autumn</h5>
                                        <p>Chrysanthemums, Marigolds</p>
                                        <small class="text-muted">September - November</small>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h1 class="text-info">❄️</h1>
                                        <h5>Winter</h5>
                                        <p>Poinsettias, Winter Jasmine</p>
                                        <small class="text-muted">December - February</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="alert alert-warning mt-4">
                            <h5>🌡️ Temperature Matters!</h5>
                            <p>Some flowers like it warm, others like it cool. That's why they bloom at different times!</p>
                        </div>
                    </div>
                `
            },
            conservation: {
                title: "🌿 Protecting Nature",
                content: `
                    <div class="lesson-content">
                        <h4>🌿 How Can We Help Protect Plants?</h4>
                        <p>Plants are very important for our planet! Here's how we can help protect them:</p>
                        
                        <div class="row mt-4">
                            <div class="col-md-6">
                                <h5>🌱 Things We Can Do:</h5>
                                <ul class="list-group">
                                    <li class="list-group-item">🌱 Plant trees and flowers in our gardens</li>
                                    <li class="list-group-item">💧 Water plants regularly</li>
                                    <li class="list-group-item">🗑️ Don't litter - keep nature clean</li>
                                    <li class="list-group-item">♻️ Recycle paper to save trees</li>
                                    <li class="list-group-item">🚶 Walk instead of driving when possible</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h5>🚫 Things to Avoid:</h5>
                                <ul class="list-group">
                                    <li class="list-group-item">🚫 Don't pick flowers in parks</li>
                                    <li class="list-group-item">🚫 Don't step on plants</li>
                                    <li class="list-group-item">🚫 Don't waste water</li>
                                    <li class="list-group-item">🚫 Don't use too much plastic</li>
                                    <li class="list-group-item">🚫 Don't pollute the air</li>
                                </ul>
                            </div>
                        </div>
                        
                        <div class="alert alert-success mt-4">
                            <h5>🌍 Remember:</h5>
                            <p>When we protect plants, we protect our whole planet! Plants give us clean air, food, and beautiful places to play!</p>
                        </div>
                    </div>
                `
            },
            nasa: {
                title: "🚀 NASA & Space",
                content: `
                    <div class="lesson-content">
                        <h4>🚀 How NASA Studies Plants from Space!</h4>
                        <p>NASA uses special satellites to take pictures of Earth and see how plants are growing all around the world!</p>
                        
                        <div class="row mt-4">
                            <div class="col-md-6">
                                <h5>🛰️ NASA Satellites:</h5>
                                <ul class="list-group">
                                    <li class="list-group-item">🛰️ Landsat - Takes pictures every 16 days</li>
                                    <li class="list-group-item">🛰️ MODIS - Takes pictures every day</li>
                                    <li class="list-group-item">🛰️ VIIRS - Takes pictures at night too</li>
                                    <li class="list-group-item">🛰️ Sentinel - European satellite</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h5>📊 What They See:</h5>
                                <ul class="list-group">
                                    <li class="list-group-item">🌱 How green plants are</li>
                                    <li class="list-group-item">📈 How plants grow over time</li>
                                    <li class="list-group-item">🌍 Where plants are healthy</li>
                                    <li class="list-group-item">🔥 Where there might be problems</li>
                                </ul>
                            </div>
                        </div>
                        
                        <div class="alert alert-info mt-4">
                            <h5>🌟 Cool NASA Facts:</h5>
                            <ul>
                                <li>🛰️ Satellites orbit Earth 14 times per day!</li>
                                <li>📸 They can see plants as small as a car!</li>
                                <li>🌍 They take pictures of the whole Earth!</li>
                                <li>📊 Scientists use these pictures to help farmers!</li>
                            </ul>
                        </div>
                    </div>
                `
            },
            pollination: {
                title: "🦋 Pollination & Bugs",
                content: `
                    <div class="lesson-content">
                        <h4>🦋 How Bees and Butterflies Help Plants!</h4>
                        <p>Many plants need help from animals to make seeds and grow new plants. This is called pollination!</p>
                        
                        <div class="row mt-4">
                            <div class="col-md-4">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h1 class="text-warning">🐝</h1>
                                        <h5>Bees</h5>
                                        <p>Busy workers that carry pollen from flower to flower!</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h1 class="text-info">🦋</h1>
                                        <h5>Butterflies</h5>
                                        <p>Beautiful flyers that help spread pollen!</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h1 class="text-success">🐦</h1>
                                        <h5>Hummingbirds</h5>
                                        <p>Fast flyers that drink nectar and spread pollen!</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="alert alert-warning mt-4">
                            <h5>🌸 The Pollination Process:</h5>
                            <ol>
                                <li>🦋 Animal visits flower for food (nectar)</li>
                                <li>🌺 Pollen sticks to animal's body</li>
                                <li>🦋 Animal flies to another flower</li>
                                <li>🌺 Pollen falls off and helps make seeds</li>
                                <li>🌱 New plants can grow from seeds!</li>
                            </ol>
                        </div>
                        
                        <div class="alert alert-success mt-4">
                            <h5>💡 Fun Fact:</h5>
                            <p>Without bees and butterflies, we wouldn't have many of our favorite fruits like apples, strawberries, and oranges!</p>
                        </div>
                    </div>
                `
            }
        };
        
        this.quizQuestions = [
            {
                question: "What do plants need to make their own food?",
                options: ["A) Sunlight, water, and air", "B) Just water", "C) Only sunlight"],
                correct: "A",
                explanation: "Plants use sunlight, water, and carbon dioxide from air to make food!"
            },
            {
                question: "Which season do cherry blossoms usually bloom?",
                options: ["A) Winter", "B) Spring", "C) Summer"],
                correct: "B",
                explanation: "Cherry blossoms bloom in spring when the weather gets warmer!"
            },
            {
                question: "How do NASA satellites help study plants?",
                options: ["A) They water the plants", "B) They take pictures from space", "C) They plant seeds"],
                correct: "B",
                explanation: "NASA satellites take pictures of Earth to see how plants are growing!"
            },
            {
                question: "What is pollination?",
                options: ["A) When plants grow taller", "B) When animals help plants make seeds", "C) When plants make food"],
                correct: "B",
                explanation: "Pollination is when bees, butterflies, and other animals help plants make seeds!"
            },
            {
                question: "Why is it important to protect plants?",
                options: ["A) They give us clean air", "B) They make our world beautiful", "C) Both A and B"],
                correct: "C",
                explanation: "Plants give us clean air to breathe AND make our world beautiful!"
            }
        ];
        
        this.currentQuizQuestion = 0;
        this.quizScore = 0;
        this.plantGrowthData = this.generatePlantGrowthData();
        
        this.init();
    }
    
    init() {
        this.initializePlantGrowthChart();
        this.loadQuizQuestion();
    }
    
    showLesson(lessonType) {
        const lesson = this.lessons[lessonType];
        if (!lesson) return;
        
        this.currentLesson = lessonType;
        
        document.getElementById('lessonTitle').innerHTML = `
            <i class="fas fa-book me-2"></i>${lesson.title}
        `;
        document.getElementById('lessonBody').innerHTML = lesson.content;
        document.getElementById('lessonContent').style.display = 'block';
        
        // Scroll to lesson content
        document.getElementById('lessonContent').scrollIntoView({ behavior: 'smooth' });
        
        // Update progress
        this.updateProgress();
    }
    
    hideLesson() {
        document.getElementById('lessonContent').style.display = 'none';
        this.currentLesson = null;
    }
    
    initializePlantGrowthChart() {
        const ctx = document.getElementById('plantGrowthChart').getContext('2d');
        
        this.plantChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                datasets: [{
                    label: '🌱 Plant Growth',
                    data: this.plantGrowthData,
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
                    legend: {
                        labels: {
                            font: {
                                family: 'Comic Neue',
                                size: 14
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            font: {
                                family: 'Comic Neue'
                            }
                        }
                    },
                    x: {
                        ticks: {
                            font: {
                                family: 'Comic Neue'
                            }
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }
    
    generatePlantGrowthData() {
        // Simulate seasonal plant growth
        return [10, 15, 25, 45, 70, 85, 90, 85, 70, 45, 25, 15];
    }
    
    loadQuizQuestion() {
        if (this.currentQuizQuestion >= this.quizQuestions.length) {
            this.currentQuizQuestion = 0;
        }
        
        const question = this.quizQuestions[this.currentQuizQuestion];
        
        document.getElementById('quizQuestion').textContent = question.question;
        
        const buttons = document.querySelectorAll('#quizContainer .btn');
        question.options.forEach((option, index) => {
            buttons[index].textContent = option;
        });
        
        // Reset quiz result
        document.getElementById('quizResult').style.display = 'none';
        document.getElementById('nextQuizBtn').style.display = 'none';
        
        // Enable buttons
        buttons.forEach(btn => {
            btn.disabled = false;
            btn.className = 'btn btn-outline-primary';
        });
    }
    
    answerQuiz(answer) {
        const question = this.quizQuestions[this.currentQuizQuestion];
        const resultDiv = document.getElementById('quizResult');
        const buttons = document.querySelectorAll('#quizContainer .btn');
        
        // Disable all buttons
        buttons.forEach(btn => {
            btn.disabled = true;
        });
        
        // Check answer
        if (answer === question.correct) {
            this.quizScore++;
            resultDiv.innerHTML = `
                <div class="alert alert-success">
                    <h6>🎉 Correct!</h6>
                    <p>${question.explanation}</p>
                </div>
            `;
            
            // Highlight correct button
            buttons[question.correct.charCodeAt(0) - 65].className = 'btn btn-success';
        } else {
            resultDiv.innerHTML = `
                <div class="alert alert-danger">
                    <h6>😊 Not quite right!</h6>
                    <p>${question.explanation}</p>
                </div>
            `;
            
            // Highlight correct and wrong buttons
            buttons[question.correct.charCodeAt(0) - 65].className = 'btn btn-success';
            buttons[answer.charCodeAt(0) - 65].className = 'btn btn-danger';
        }
        
        resultDiv.style.display = 'block';
        document.getElementById('nextQuizBtn').style.display = 'block';
        
        // Update score display
        this.updateQuizScore();
    }
    
    nextQuizQuestion() {
        this.currentQuizQuestion++;
        this.loadQuizQuestion();
    }
    
    updateQuizScore() {
        // Update score in the progress section
        const scoreElement = document.querySelector('.text-center:nth-child(3) h3');
        if (scoreElement) {
            scoreElement.textContent = this.quizScore;
        }
    }
    
    updateProgress() {
        // Simulate progress update
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            let currentWidth = parseInt(progressBar.style.width) || 65;
            currentWidth = Math.min(currentWidth + 5, 100);
            progressBar.style.width = currentWidth + '%';
            progressBar.innerHTML = `<span class="fw-bold">${currentWidth}% Complete</span>`;
        }
        
        // Update lesson count
        const lessonCount = document.querySelector('.text-center:nth-child(1) h3');
        if (lessonCount) {
            let count = parseInt(lessonCount.textContent) || 5;
            lessonCount.textContent = count + 1;
        }
    }
}

// Global functions for HTML onclick events
function showLesson(lessonType) {
    learnApp.showLesson(lessonType);
}

function hideLesson() {
    learnApp.hideLesson();
}

function answerQuiz(answer) {
    learnApp.answerQuiz(answer);
}

function nextQuizQuestion() {
    learnApp.nextQuizQuestion();
}

// Initialize the app when the page loads
let learnApp;
document.addEventListener('DOMContentLoaded', () => {
    learnApp = new LearnApp();
});

// Add CSS styles for lesson content
const style = document.createElement('style');
style.textContent = `
    .lesson-content {
        font-family: 'Comic Neue', cursive;
        line-height: 1.8;
    }
    
    .lesson-content h4 {
        color: #28a745;
        margin-bottom: 20px;
    }
    
    .lesson-content h5 {
        color: #007bff;
        margin-bottom: 15px;
    }
    
    .learning-card {
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .learning-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15) !important;
        border-color: #28a745;
    }
    
    .achievement-item {
        transition: all 0.3s ease;
    }
    
    .achievement-item:hover {
        transform: translateX(5px);
    }
    
    .quiz-container {
        font-family: 'Comic Neue', cursive;
    }
    
    .progress {
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .progress-bar {
        transition: width 0.5s ease;
        border-radius: 10px;
    }
`;
document.head.appendChild(style);
