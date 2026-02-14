/* ============================================
   DASHBOARD PAGE
   ============================================ */

function renderDashboard(container) {
    const hour = new Date().getHours();
    const greeting = hour < 12 ? 'Good Morning' : hour < 17 ? 'Good Afternoon' : 'Good Evening';

    container.innerHTML = `
        <!-- Hero Banner -->
        <div class="dashboard-hero animate-gradient" style="background-size:200% 200%">
            <h2 class="heading-display">${greeting}, Farmer! 🌾</h2>
            <p>Your AI-powered farming assistant is ready. Get crop recommendations, detect plant diseases, check weather, and track market prices — all in one place.</p>
            <div class="hero-stats">
                <div class="hero-stat">
                    <div class="value" id="stat-crops">22</div>
                    <div class="label">Disease Classes</div>
                </div>
                <div class="hero-stat">
                    <div class="value" id="stat-models">5</div>
                    <div class="label">AI Models</div>
                </div>
                <div class="hero-stat">
                    <div class="value" id="stat-accuracy">80%</div>
                    <div class="label">Train Accuracy</div>
                </div>
            </div>
        </div>

        <!-- Quick Actions -->
        <div class="quick-actions stagger-children">
            <div class="quick-action qa-crop ripple" onclick="window.location.hash='crops'">
                <i class="fas fa-wheat-awn"></i>
                <h4>Crop Advisor</h4>
                <p>Get soil-based recommendations</p>
            </div>
            <div class="quick-action qa-disease ripple" onclick="window.location.hash='diseases'">
                <i class="fas fa-microscope"></i>
                <h4>Disease Scanner</h4>
                <p>Upload leaf images for diagnosis</p>
            </div>
            <div class="quick-action qa-weather ripple" onclick="window.location.hash='weather'">
                <i class="fas fa-cloud-sun"></i>
                <h4>Weather</h4>
                <p>Forecasts & crop advice</p>
            </div>
            <div class="quick-action qa-price ripple" onclick="window.location.hash='prices'">
                <i class="fas fa-chart-line"></i>
                <h4>Market Prices</h4>
                <p>Track commodity prices</p>
            </div>
        </div>

        <!-- Stats Row -->
        <div class="grid-4 stagger-children" style="margin-bottom: 28px;">
            <div class="stat-card hover-lift">
                <div class="stat-icon green"><i class="fas fa-seedling"></i></div>
                <div class="stat-content">
                    <div class="stat-value text-gradient" id="dash-crop-count">15</div>
                    <div class="stat-label">Crops Supported</div>
                </div>
            </div>
            <div class="stat-card hover-lift">
                <div class="stat-icon blue"><i class="fas fa-brain"></i></div>
                <div class="stat-content">
                    <div class="stat-value" style="color: var(--accent-600)">AI</div>
                    <div class="stat-label">Powered Assistant</div>
                </div>
            </div>
            <div class="stat-card hover-lift">
                <div class="stat-icon orange"><i class="fas fa-image"></i></div>
                <div class="stat-content">
                    <div class="stat-value" style="color: var(--secondary-500)" id="dash-disease-count">22</div>
                    <div class="stat-label">Disease Types</div>
                </div>
            </div>
            <div class="stat-card hover-lift">
                <div class="stat-icon red"><i class="fas fa-heartbeat"></i></div>
                <div class="stat-content">
                    <div class="stat-value" id="dash-api-status" style="color: var(--success)">●</div>
                    <div class="stat-label">System Status</div>
                </div>
            </div>
        </div>

        <!-- Two Column -->
        <div class="grid-2">
            <!-- Features List -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title"><i class="fas fa-star"></i> Features</div>
                </div>
                <div class="card-body stagger-children">
                    <div class="result-item">
                        <div class="result-rank"><i class="fas fa-comments" style="font-size:14px"></i></div>
                        <div class="result-content">
                            <div class="result-title">Natural Language Chat</div>
                            <div class="result-desc">Ask farming questions in plain English and get instant answers</div>
                        </div>
                    </div>
                    <div class="result-item">
                        <div class="result-rank"><i class="fas fa-camera" style="font-size:14px"></i></div>
                        <div class="result-content">
                            <div class="result-title">Image-Based Disease Detection</div>
                            <div class="result-desc">Upload a photo of your crop leaf for AI diagnosis</div>
                        </div>
                    </div>
                    <div class="result-item">
                        <div class="result-rank"><i class="fas fa-flask" style="font-size:14px"></i></div>
                        <div class="result-content">
                            <div class="result-title">Soil Analysis & Crop Match</div>
                            <div class="result-desc">Enter soil nutrients to get the best crop recommendation</div>
                        </div>
                    </div>
                    <div class="result-item">
                        <div class="result-rank"><i class="fas fa-chart-area" style="font-size:14px"></i></div>
                        <div class="result-content">
                            <div class="result-title">Market Price Intelligence</div>
                            <div class="result-desc">Track real-time commodity prices and trends</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Supported Crops -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title"><i class="fas fa-leaf"></i> Disease Detection — Supported Crops</div>
                </div>
                <div class="card-body">
                    <div class="grid-2" style="gap:10px" id="supported-crops">
                        ${['Cashew', 'Cassava', 'Maize', 'Tomato'].map(crop => `
                            <div style="display:flex;align-items:center;gap:10px;padding:12px;background:var(--bg-input);border-radius:var(--radius-md);">
                                <span style="font-size:24px">${getCropEmoji(crop)}</span>
                                <div>
                                    <div style="font-weight:600;font-size:14px">${crop}</div>
                                    <div style="font-size:12px;color:var(--text-muted)">Multiple diseases</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                    <div class="divider"></div>
                    <h4 style="font-size:14px;font-weight:600;margin-bottom:10px"><i class="fas fa-wheat-awn" style="color:var(--primary-500);margin-right:6px"></i>Crop Recommendation — Supported</h4>
                    <div style="display:flex;flex-wrap:wrap;gap:6px;" id="crop-badges">
                        ${['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Tomato', 'Potato', 'Onion', 'Mango', 'Banana', 'Pomegranate', 'Groundnut', 'Soybean', 'Chickpea', 'Mustard'].map(c => 
                            `<span class="badge badge-green">${getCropEmoji(c)} ${c}</span>`
                        ).join('')}
                    </div>
                </div>
            </div>
        </div>
    `;

    // Animate counters
    setTimeout(() => {
        const cropEl = document.getElementById('dash-crop-count');
        const diseaseEl = document.getElementById('dash-disease-count');
        if (cropEl) animateCounter(cropEl, 15, 800);
        if (diseaseEl) animateCounter(diseaseEl, 22, 800);
    }, 300);

    // Check API status
    api.checkHealth().then(() => {
        const el = document.getElementById('dash-api-status');
        if (el) el.innerHTML = '<span style="color:var(--success)">Online</span>';
    }).catch(() => {
        const el = document.getElementById('dash-api-status');
        if (el) el.innerHTML = '<span style="color:var(--danger)">Offline</span>';
    });
}
