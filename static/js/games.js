// BloomWatch Kids - Games Page JavaScript 🎮

class GamesApp {
    constructor() {
        this.currentGame = null;
        this.gameScore = 0;
        this.gameTime = 0;
        this.gameTimer = null;
        this.highScores = {
            plantMemory: 0,
            plantQuiz: 0,
            seasonSort: 0,
            plantPuzzle: 0,
            gardenDesigner: 0,
            plantBingo: 0
        };
        this.achievements = {
            firstGame: false,
            highScore: false,
            memoryMaster: false
        };
        
        this.init();
    }
    
    init() {
        this.loadHighScores();
        this.loadAchievements();
        this.initializeColoringCanvas();
        this.loadStoryGenerator();
    }
    
    startGame(gameType) {
        this.currentGame = gameType;
        this.gameScore = 0;
        this.gameTime = 0;
        
        document.getElementById('gameArea').style.display = 'block';
        document.getElementById('gameTitle').innerHTML = `
            <i class="fas fa-gamepad me-2"></i>${this.getGameTitle(gameType)}
        `;
        
        this.updateGameDisplay();
        this.startGameTimer();
        
        switch(gameType) {
            case 'plantMemory':
                this.startMemoryGame();
                break;
            case 'plantQuiz':
                this.startQuizGame();
                break;
            case 'seasonSort':
                this.startSeasonSortGame();
                break;
            case 'plantPuzzle':
                this.startPuzzleGame();
                break;
            case 'gardenDesigner':
                this.startGardenDesigner();
                break;
            case 'plantBingo':
                this.startPlantBingo();
                break;
        }
        
        // Scroll to game area
        document.getElementById('gameArea').scrollIntoView({ behavior: 'smooth' });
        
        // Update achievement
        if (!this.achievements.firstGame) {
            this.achievements.firstGame = true;
            this.showAchievementUnlocked("🎮 First Game Played", "You played your first game!");
            this.saveAchievements();
        }
    }
    
    getGameTitle(gameType) {
        const titles = {
            'plantMemory': '🧠 Plant Memory Game',
            'plantQuiz': '❓ Plant Quiz Challenge',
            'seasonSort': '🌸 Season Sorting',
            'plantPuzzle': '🧩 Plant Puzzle',
            'gardenDesigner': '🎨 Garden Designer',
            'plantBingo': '🎯 Plant Bingo'
        };
        return titles[gameType] || '🎮 Game';
    }
    
    startGameTimer() {
        this.gameTimer = setInterval(() => {
            this.gameTime++;
            this.updateGameDisplay();
        }, 1000);
    }
    
    stopGameTimer() {
        if (this.gameTimer) {
            clearInterval(this.gameTimer);
            this.gameTimer = null;
        }
    }
    
    updateGameDisplay() {
        document.getElementById('gameScore').textContent = `Score: ${this.gameScore}`;
        const minutes = Math.floor(this.gameTime / 60);
        const seconds = this.gameTime % 60;
        document.getElementById('gameTime').textContent = `Time: ${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
    
    exitGame() {
        this.stopGameTimer();
        document.getElementById('gameArea').style.display = 'none';
        this.currentGame = null;
    }
    
    addScore(points) {
        this.gameScore += points;
        this.updateGameDisplay();
        
        // Check for high score
        if (this.gameScore > this.highScores[this.currentGame]) {
            this.highScores[this.currentGame] = this.gameScore;
            this.saveHighScores();
            
            if (!this.achievements.highScore) {
                this.achievements.highScore = true;
                this.showAchievementUnlocked("🏆 High Score Master", "You got a new high score!");
                this.saveAchievements();
            }
        }
    }
    
    startMemoryGame() {
        const gameContent = document.getElementById('gameContent');
        gameContent.innerHTML = `
            <div class="memory-game">
                <h5>🧠 Match the Plant Pairs!</h5>
                <p>Click on the cards to flip them and find matching plant pairs!</p>
                <div class="row" id="memoryCards">
                    ${this.generateMemoryCards()}
                </div>
                <div class="mt-3">
                    <button class="btn btn-primary" onclick="gamesApp.restartMemoryGame()">
                        <i class="fas fa-redo me-2"></i>🔄 New Game
                    </button>
                </div>
            </div>
        `;
    }
    
    generateMemoryCards() {
        const plants = ['🌸', '🌹', '🌻', '🌺', '🌼', '🌷', '🌿', '🌱'];
        const cards = [...plants, ...plants]; // Duplicate for pairs
        cards.sort(() => Math.random() - 0.5);
        
        return cards.map((plant, index) => `
            <div class="col-3 col-md-2 mb-3">
                <div class="memory-card" data-plant="${plant}" data-index="${index}" onclick="gamesApp.flipMemoryCard(this)">
                    <div class="card-front">❓</div>
                    <div class="card-back">${plant}</div>
                </div>
            </div>
        `).join('');
    }
    
    flipMemoryCard(card) {
        if (card.classList.contains('flipped') || card.classList.contains('matched')) return;
        
        card.classList.add('flipped');
        const flippedCards = document.querySelectorAll('.memory-card.flipped:not(.matched)');
        
        if (flippedCards.length === 2) {
            const [card1, card2] = flippedCards;
            if (card1.dataset.plant === card2.dataset.plant) {
                // Match found!
                card1.classList.add('matched');
                card2.classList.add('matched');
                this.addScore(10);
                this.showNotification("🎉 Match found! +10 points!", "success");
                
                // Check if game is complete
                if (document.querySelectorAll('.memory-card.matched').length === 16) {
                    this.stopGameTimer();
                    this.showNotification("🏆 Memory game completed! Well done!", "success");
                }
            } else {
                // No match
                setTimeout(() => {
                    card1.classList.remove('flipped');
                    card2.classList.remove('flipped');
                }, 1000);
            }
        }
    }
    
    restartMemoryGame() {
        this.startMemoryGame();
    }
    
    startQuizGame() {
        this.quizQuestions = [
            {
                question: "What do plants need to make food?",
                options: ["Sunlight", "Water", "Air", "All of the above"],
                correct: 3
            },
            {
                question: "Which flower blooms in spring?",
                options: ["Sunflower", "Tulip", "Rose", "Marigold"],
                correct: 1
            },
            {
                question: "How do bees help plants?",
                options: ["They water plants", "They carry pollen", "They eat bugs", "They make honey"],
                correct: 1
            }
        ];
        this.currentQuizQuestion = 0;
        this.loadQuizQuestion();
    }
    
    loadQuizQuestion() {
        if (this.currentQuizQuestion >= this.quizQuestions.length) {
            this.currentQuizQuestion = 0;
        }
        
        const question = this.quizQuestions[this.currentQuizQuestion];
        const gameContent = document.getElementById('gameContent');
        
        gameContent.innerHTML = `
            <div class="quiz-game">
                <h5>❓ Plant Quiz Challenge!</h5>
                <p>Answer quickly to get more points!</p>
                <div class="question-card">
                    <h6>${question.question}</h6>
                    <div class="d-grid gap-2 mt-3">
                        ${question.options.map((option, index) => `
                            <button class="btn btn-outline-primary" onclick="gamesApp.answerQuiz(${index})">
                                ${option}
                            </button>
                        `).join('')}
                    </div>
                </div>
                <div class="mt-3">
                    <button class="btn btn-secondary" onclick="gamesApp.nextQuizQuestion()">
                        <i class="fas fa-arrow-right me-2"></i>Skip Question
                    </button>
                </div>
            </div>
        `;
    }
    
    answerQuiz(answerIndex) {
        const question = this.quizQuestions[this.currentQuizQuestion];
        const buttons = document.querySelectorAll('.quiz-game .btn');
        
        // Disable all buttons
        buttons.forEach(btn => btn.disabled = true);
        
        if (answerIndex === question.correct) {
            buttons[answerIndex].className = 'btn btn-success';
            this.addScore(20);
            this.showNotification("🎉 Correct! +20 points!", "success");
        } else {
            buttons[answerIndex].className = 'btn btn-danger';
            buttons[question.correct].className = 'btn btn-success';
            this.showNotification("😊 Not quite right, but keep trying!", "info");
        }
        
        setTimeout(() => {
            this.nextQuizQuestion();
        }, 2000);
    }
    
    nextQuizQuestion() {
        this.currentQuizQuestion++;
        this.loadQuizQuestion();
    }
    
    startSeasonSortGame() {
        const gameContent = document.getElementById('gameContent');
        gameContent.innerHTML = `
            <div class="season-sort-game">
                <h5>🌸 Sort Plants by Season!</h5>
                <p>Drag the plants to their correct blooming season!</p>
                <div class="row">
                    <div class="col-md-3">
                        <div class="season-dropzone" data-season="spring" ondrop="gamesApp.drop(event)" ondragover="gamesApp.allowDrop(event)">
                            <h6>🌸 Spring</h6>
                            <div class="dropzone-area"></div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="season-dropzone" data-season="summer" ondrop="gamesApp.drop(event)" ondragover="gamesApp.allowDrop(event)">
                            <h6>🌻 Summer</h6>
                            <div class="dropzone-area"></div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="season-dropzone" data-season="autumn" ondrop="gamesApp.drop(event)" ondragover="gamesApp.allowDrop(event)">
                            <h6>🍂 Autumn</h6>
                            <div class="dropzone-area"></div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="season-dropzone" data-season="winter" ondrop="gamesApp.drop(event)" ondragover="gamesApp.allowDrop(event)">
                            <h6>❄️ Winter</h6>
                            <div class="dropzone-area"></div>
                        </div>
                    </div>
                </div>
                <div class="mt-4">
                    <h6>Drag these plants to the right season:</h6>
                    <div class="plant-items">
                        <div class="plant-item" draggable="true" ondragstart="gamesApp.drag(event)" data-plant="tulip" data-season="spring">🌷 Tulip</div>
                        <div class="plant-item" draggable="true" ondragstart="gamesApp.drag(event)" data-plant="sunflower" data-season="summer">🌻 Sunflower</div>
                        <div class="plant-item" draggable="true" ondragstart="gamesApp.drag(event)" data-plant="chrysanthemum" data-season="autumn">🌼 Chrysanthemum</div>
                        <div class="plant-item" draggable="true" ondragstart="gamesApp.drag(event)" data-plant="poinsettia" data-season="winter">🌺 Poinsettia</div>
                    </div>
                </div>
            </div>
        `;
    }
    
    allowDrop(ev) {
        ev.preventDefault();
    }
    
    drag(ev) {
        ev.dataTransfer.setData("text", ev.target.id);
    }
    
    drop(ev) {
        ev.preventDefault();
        const data = ev.dataTransfer.getData("text");
        const draggedElement = document.getElementById(data);
        const season = ev.currentTarget.dataset.season;
        
        if (draggedElement.dataset.season === season) {
            ev.currentTarget.querySelector('.dropzone-area').appendChild(draggedElement);
            this.addScore(15);
            this.showNotification("🎉 Correct season! +15 points!", "success");
        } else {
            this.showNotification("😊 Try again! That's not the right season.", "warning");
        }
    }
    
    startPuzzleGame() {
        const gameContent = document.getElementById('gameContent');
        gameContent.innerHTML = `
            <div class="puzzle-game">
                <h5>🧩 Plant Puzzle!</h5>
                <p>Click and drag the pieces to complete the plant puzzle!</p>
                <div class="puzzle-container">
                    <div class="puzzle-pieces" id="puzzlePieces">
                        <!-- Puzzle pieces will be generated here -->
                    </div>
                    <div class="puzzle-board" id="puzzleBoard">
                        <!-- Puzzle board will be generated here -->
                    </div>
                </div>
            </div>
        `;
        
        this.generatePuzzle();
    }
    
    generatePuzzle() {
        // Simple puzzle implementation
        const puzzlePieces = document.getElementById('puzzlePieces');
        const puzzleBoard = document.getElementById('puzzleBoard');
        
        puzzlePieces.innerHTML = `
            <div class="puzzle-piece" draggable="true" ondragstart="gamesApp.dragPuzzlePiece(event)">🌱</div>
            <div class="puzzle-piece" draggable="true" ondragstart="gamesApp.dragPuzzlePiece(event)">🌸</div>
            <div class="puzzle-piece" draggable="true" ondragstart="gamesApp.dragPuzzlePiece(event)">🌿</div>
            <div class="puzzle-piece" draggable="true" ondragstart="gamesApp.dragPuzzlePiece(event)">🌳</div>
        `;
        
        puzzleBoard.innerHTML = `
            <div class="puzzle-slot" ondrop="gamesApp.dropPuzzlePiece(event)" ondragover="gamesApp.allowDrop(event)"></div>
            <div class="puzzle-slot" ondrop="gamesApp.dropPuzzlePiece(event)" ondragover="gamesApp.allowDrop(event)"></div>
            <div class="puzzle-slot" ondrop="gamesApp.dropPuzzlePiece(event)" ondragover="gamesApp.allowDrop(event)"></div>
            <div class="puzzle-slot" ondrop="gamesApp.dropPuzzlePiece(event)" ondragover="gamesApp.allowDrop(event)"></div>
        `;
    }
    
    dragPuzzlePiece(ev) {
        ev.dataTransfer.setData("text", ev.target.outerHTML);
    }
    
    dropPuzzlePiece(ev) {
        ev.preventDefault();
        const data = ev.dataTransfer.getData("text");
        ev.currentTarget.innerHTML = data;
        this.addScore(5);
        this.showNotification("🎉 Puzzle piece placed! +5 points!", "success");
    }
    
    startGardenDesigner() {
        const gameContent = document.getElementById('gameContent');
        gameContent.innerHTML = `
            <div class="garden-designer">
                <h5>🎨 Design Your Garden!</h5>
                <p>Click to plant flowers and create your beautiful garden!</p>
                <div class="garden-tools">
                    <button class="btn btn-success btn-sm me-2" onclick="gamesApp.selectPlant('🌷')">🌷 Tulip</button>
                    <button class="btn btn-primary btn-sm me-2" onclick="gamesApp.selectPlant('🌹')">🌹 Rose</button>
                    <button class="btn btn-warning btn-sm me-2" onclick="gamesApp.selectPlant('🌻')">🌻 Sunflower</button>
                    <button class="btn btn-info btn-sm me-2" onclick="gamesApp.selectPlant('🌺')">🌺 Lily</button>
                    <button class="btn btn-danger btn-sm" onclick="gamesApp.selectPlant('')">🧹 Remove</button>
                </div>
                <div class="garden-grid" id="gardenGrid">
                    ${this.generateGardenGrid()}
                </div>
                <div class="mt-3">
                    <button class="btn btn-primary" onclick="gamesApp.saveGarden()">
                        <i class="fas fa-save me-2"></i>💾 Save Garden
                    </button>
                </div>
            </div>
        `;
        
        this.selectedPlant = '🌷';
    }
    
    generateGardenGrid() {
        let grid = '';
        for (let i = 0; i < 64; i++) {
            grid += `<div class="garden-cell" onclick="gamesApp.plantInCell(this, ${i})"></div>`;
        }
        return grid;
    }
    
    selectPlant(plant) {
        this.selectedPlant = plant;
        this.showNotification(`🌱 Selected: ${plant || 'Remove tool'}`, "info");
    }
    
    plantInCell(cell, index) {
        if (this.selectedPlant) {
            cell.textContent = this.selectedPlant;
            this.addScore(2);
        } else {
            cell.textContent = '';
        }
    }
    
    saveGarden() {
        this.addScore(20);
        this.showNotification("🏆 Garden saved! +20 points!", "success");
    }
    
    startPlantBingo() {
        const gameContent = document.getElementById('gameContent');
        gameContent.innerHTML = `
            <div class="plant-bingo">
                <h5>🎯 Plant Bingo!</h5>
                <p>Find these plants in your neighborhood and mark them off!</p>
                <div class="bingo-card">
                    ${this.generateBingoCard()}
                </div>
                <div class="mt-3">
                    <button class="btn btn-success" onclick="gamesApp.markBingoItem()">
                        <i class="fas fa-check me-2"></i>✅ Found a Plant!
                    </button>
                </div>
            </div>
        `;
    }
    
    generateBingoCard() {
        const plants = ['🌹 Rose', '🌷 Tulip', '🌻 Sunflower', '🌺 Lily', '🌼 Daisy', '🌿 Fern', '🌱 Grass', '🌳 Tree', '🌲 Pine'];
        let card = '';
        for (let i = 0; i < 9; i++) {
            card += `
                <div class="bingo-item" onclick="gamesApp.toggleBingoItem(this)">
                    ${plants[i]}
                </div>
            `;
        }
        return card;
    }
    
    toggleBingoItem(item) {
        if (item.classList.contains('marked')) {
            item.classList.remove('marked');
            this.gameScore -= 5;
        } else {
            item.classList.add('marked');
            this.addScore(10);
            this.showNotification("🎉 Plant found! +10 points!", "success");
        }
        this.updateGameDisplay();
    }
    
    markBingoItem() {
        const unmarkedItems = document.querySelectorAll('.bingo-item:not(.marked)');
        if (unmarkedItems.length > 0) {
            const randomItem = unmarkedItems[Math.floor(Math.random() * unmarkedItems.length)];
            randomItem.classList.add('marked');
            this.addScore(5);
            this.showNotification("🎯 Random plant marked! +5 points!", "success");
        }
    }
    
    // Coloring canvas functionality
    initializeColoringCanvas() {
        this.selectedColor = '#FF6B6B';
        this.canvas = document.getElementById('plantColoringCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        
        if (this.canvas) {
            this.setupCanvasEvents();
            this.drawPlantOutline();
        }
    }
    
    setupCanvasEvents() {
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseout', () => this.stopDrawing());
    }
    
    drawPlantOutline() {
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 2;
        
        // Draw a simple plant outline
        this.ctx.beginPath();
        this.ctx.arc(150, 100, 40, 0, Math.PI * 2); // Flower head
        this.ctx.moveTo(150, 140);
        this.ctx.lineTo(150, 250); // Stem
        this.ctx.moveTo(140, 200);
        this.ctx.lineTo(160, 200); // Leaf
        this.ctx.stroke();
    }
    
    startDrawing(e) {
        this.isDrawing = true;
        this.draw(e);
    }
    
    draw(e) {
        if (!this.isDrawing) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        this.ctx.fillStyle = this.selectedColor;
        this.ctx.beginPath();
        this.ctx.arc(x, y, 5, 0, Math.PI * 2);
        this.ctx.fill();
    }
    
    stopDrawing() {
        this.isDrawing = false;
    }
    
    selectColor(color) {
        this.selectedColor = color;
    }
    
    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawPlantOutline();
    }
    
    // Story generator
    loadStoryGenerator() {
        this.storyTemplates = [
            "Once upon a time, there was a magical {plant} that lived in a {location}. Every morning, it would {action} and make the whole garden {emotion}!",
            "In a secret garden, a brave {plant} decided to {action}. Along the way, it met a friendly {animal} who helped it {help}. Together, they made the world more {emotion}!",
            "Deep in the enchanted forest, a special {plant} had the power to {action}. When children visited, the plant would {emotion} them with its {color} colors!"
        ];
        
        this.storyWords = {
            plant: ['🌹 rose', '🌻 sunflower', '🌷 tulip', '🌸 cherry blossom', '🌺 lily', '🌼 daisy'],
            location: ['beautiful garden', 'mysterious forest', 'colorful meadow', 'enchanted park', 'secret greenhouse'],
            action: ['dance in the breeze', 'sing with the birds', 'glow in the moonlight', 'bloom brighter than ever', 'make friends with butterflies'],
            emotion: ['happy', 'magical', 'wonderful', 'amazing', 'spectacular', 'beautiful'],
            animal: ['🐝 bee', '🦋 butterfly', '🐛 caterpillar', '🐞 ladybug', '🐦 hummingbird'],
            help: ['find the perfect spot to grow', 'spread seeds around', 'make the garden more colorful'],
            color: ['bright', 'vibrant', 'beautiful', 'magical', 'brilliant']
        };
    }
    
    generateStory() {
        const template = this.storyTemplates[Math.floor(Math.random() * this.storyTemplates.length)];
        let story = template;
        
        Object.keys(this.storyWords).forEach(key => {
            const words = this.storyWords[key];
            const randomWord = words[Math.floor(Math.random() * words.length)];
            story = story.replace(new RegExp(`{${key}}`, 'g'), randomWord);
        });
        
        document.getElementById('storyText').innerHTML = `
            <div class="story-content">
                <h6>🌟 Your Plant Adventure:</h6>
                <p class="story-paragraph">${story}</p>
                <div class="story-moral">
                    <strong>💡 The moral of the story:</strong> Plants make our world beautiful and magical! 🌱✨
                </div>
            </div>
        `;
        
        this.addScore(5);
        this.showNotification("📖 New story generated! +5 points!", "success");
    }
    
    // Utility functions
    loadHighScores() {
        const saved = localStorage.getItem('bloomWatchHighScores');
        if (saved) {
            this.highScores = JSON.parse(saved);
        }
        this.updateLeaderboard();
    }
    
    saveHighScores() {
        localStorage.setItem('bloomWatchHighScores', JSON.stringify(this.highScores));
        this.updateLeaderboard();
    }
    
    loadAchievements() {
        const saved = localStorage.getItem('bloomWatchAchievements');
        if (saved) {
            this.achievements = JSON.parse(saved);
        }
        this.updateAchievementDisplay();
    }
    
    saveAchievements() {
        localStorage.setItem('bloomWatchAchievements', JSON.stringify(this.achievements));
        this.updateAchievementDisplay();
    }
    
    updateLeaderboard() {
        // Update the leaderboard display
        const highScores = document.querySelectorAll('.list-group-item .badge');
        if (highScores.length >= 3) {
            highScores[0].textContent = `${this.highScores.plantMemory} points`;
            highScores[1].textContent = `${this.highScores.plantQuiz} points`;
            highScores[2].textContent = `${this.highScores.seasonSort} points`;
        }
    }
    
    updateAchievementDisplay() {
        const achievements = document.querySelectorAll('.list-group-item:nth-child(2) .badge, .list-group-item:nth-child(3) .badge');
        if (achievements.length >= 2) {
            achievements[0].textContent = this.achievements.firstGame ? 'Unlocked!' : 'Locked';
            achievements[0].className = this.achievements.firstGame ? 'badge bg-success' : 'badge bg-secondary';
            
            achievements[1].textContent = this.achievements.highScore ? 'Unlocked!' : 'Locked';
            achievements[1].className = this.achievements.highScore ? 'badge bg-warning text-dark' : 'badge bg-secondary';
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
function startGame(gameType) {
    gamesApp.startGame(gameType);
}

function exitGame() {
    gamesApp.exitGame();
}

function selectColor(color) {
    gamesApp.selectColor(color);
}

function clearCanvas() {
    gamesApp.clearCanvas();
}

function generateStory() {
    gamesApp.generateStory();
}

// Initialize the app when the page loads
let gamesApp;
document.addEventListener('DOMContentLoaded', () => {
    gamesApp = new GamesApp();
});

// Add CSS styles
const style = document.createElement('style');
style.textContent = `
    .game-card {
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .game-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15) !important;
        border-color: #28a745;
    }
    
    .memory-card {
        aspect-ratio: 1;
        position: relative;
        cursor: pointer;
        transform-style: preserve-3d;
        transition: transform 0.6s;
    }
    
    .memory-card.flipped {
        transform: rotateY(180deg);
    }
    
    .card-front, .card-back {
        position: absolute;
        width: 100%;
        height: 100%;
        backface-visibility: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        border-radius: 8px;
        border: 2px solid #ddd;
    }
    
    .card-front {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .card-back {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        transform: rotateY(180deg);
    }
    
    .memory-card.matched {
        opacity: 0.7;
        transform: scale(0.95);
    }
    
    .season-dropzone {
        border: 2px dashed #ccc;
        border-radius: 8px;
        padding: 15px;
        min-height: 150px;
        text-align: center;
    }
    
    .plant-item {
        display: inline-block;
        padding: 10px 15px;
        margin: 5px;
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        cursor: grab;
        transition: all 0.3s ease;
    }
    
    .plant-item:hover {
        background: #e9ecef;
        transform: translateY(-2px);
    }
    
    .plant-item:active {
        cursor: grabbing;
    }
    
    .garden-grid {
        display: grid;
        grid-template-columns: repeat(8, 1fr);
        gap: 2px;
        max-width: 400px;
        margin: 0 auto;
    }
    
    .garden-cell {
        aspect-ratio: 1;
        border: 1px solid #ddd;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        transition: background-color 0.3s ease;
    }
    
    .garden-cell:hover {
        background-color: #f8f9fa;
    }
    
    .bingo-card {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        max-width: 400px;
        margin: 0 auto;
    }
    
    .bingo-item {
        padding: 15px;
        text-align: center;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .bingo-item:hover {
        background-color: #f8f9fa;
    }
    
    .bingo-item.marked {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .puzzle-container {
        display: flex;
        gap: 20px;
        justify-content: center;
    }
    
    .puzzle-pieces {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    
    .puzzle-piece {
        padding: 15px;
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        cursor: grab;
        font-size: 2rem;
    }
    
    .puzzle-board {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
    }
    
    .puzzle-slot {
        width: 80px;
        height: 80px;
        border: 2px dashed #ccc;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .story-content {
        font-family: 'Comic Neue', cursive;
        line-height: 1.8;
    }
    
    .story-paragraph {
        font-size: 1.1rem;
        margin-bottom: 15px;
    }
    
    .story-moral {
        background: #fff3cd;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
`;
document.head.appendChild(style);
