/* ============================================
   CROP RECOMMENDATION PAGE
   ============================================ */

function renderCrops(container) {
    container.innerHTML = `
        <div class="page-header">
            <h2><i class="fas fa-wheat-awn"></i> Crop Advisor</h2>
            <p>Enter your soil parameters to get AI-powered crop recommendations</p>
        </div>

        <div class="grid-2" style="grid-template-columns: 1.2fr 0.8fr;">
            <!-- Input Form -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title"><i class="fas fa-flask"></i> Soil Parameters</div>
                    <button class="btn btn-ghost btn-sm" onclick="fillSampleSoilData()">
                        <i class="fas fa-wand-magic-sparkles"></i> Sample Data
                    </button>
                </div>
                <div class="card-body">
                    <div class="soil-form-grid">
                        <div class="form-group">
                            <label class="form-label">Nitrogen (N) *</label>
                            <input type="number" class="form-input" id="soil-N" placeholder="e.g. 80" min="0" max="200">
                            <span class="form-hint">kg/ha</span>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Phosphorus (P) *</label>
                            <input type="number" class="form-input" id="soil-P" placeholder="e.g. 40" min="0" max="200">
                            <span class="form-hint">kg/ha</span>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Potassium (K) *</label>
                            <input type="number" class="form-input" id="soil-K" placeholder="e.g. 40" min="0" max="200">
                            <span class="form-hint">kg/ha</span>
                        </div>
                        <div class="form-group">
                            <label class="form-label">pH Level *</label>
                            <input type="number" class="form-input" id="soil-ph" placeholder="e.g. 6.5" min="0" max="14" step="0.1">
                            <span class="form-hint">0 - 14</span>
                        </div>
                        <div class="form-group">
                            <label class="form-label">EC</label>
                            <input type="number" class="form-input" id="soil-EC" placeholder="e.g. 0.5" min="0" step="0.1" value="0">
                            <span class="form-hint">dS/m</span>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Sulfur (S)</label>
                            <input type="number" class="form-input" id="soil-S" placeholder="e.g. 10" min="0" value="0">
                            <span class="form-hint">ppm</span>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Copper (Cu)</label>
                            <input type="number" class="form-input" id="soil-Cu" placeholder="e.g. 2" min="0" value="0">
                            <span class="form-hint">ppm</span>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Iron (Fe)</label>
                            <input type="number" class="form-input" id="soil-Fe" placeholder="e.g. 50" min="0" value="0">
                            <span class="form-hint">ppm</span>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Manganese (Mn)</label>
                            <input type="number" class="form-input" id="soil-Mn" placeholder="e.g. 20" min="0" value="0">
                            <span class="form-hint">ppm</span>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Zinc (Zn)</label>
                            <input type="number" class="form-input" id="soil-Zn" placeholder="e.g. 1" min="0" value="0">
                            <span class="form-hint">ppm</span>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Boron (B)</label>
                            <input type="number" class="form-input" id="soil-B" placeholder="e.g. 1" min="0" value="0">
                            <span class="form-hint">ppm</span>
                        </div>
                    </div>
                    <div style="margin-top: 4px;">
                        <button class="btn btn-primary btn-lg btn-block" id="crop-submit" onclick="submitCropForm()">
                            <i class="fas fa-magnifying-glass-chart"></i> Get Recommendations
                        </button>
                    </div>
                </div>
            </div>

            <!-- Results Panel -->
            <div id="crop-results-panel">
                <div class="card">
                    <div class="empty-state">
                        <i class="fas fa-seedling animate-float"></i>
                        <h3>Enter Soil Data</h3>
                        <p>Fill in your soil parameters and click "Get Recommendations" to see AI-powered crop suggestions</p>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function fillSampleSoilData() {
    const samples = [
        { N: 90, P: 45, K: 45, ph: 6.5, EC: 0.5, S: 12, Cu: 2, Fe: 50, Mn: 20, Zn: 1, B: 1 },
        { N: 40, P: 60, K: 30, ph: 7.2, EC: 0.3, S: 8, Cu: 1, Fe: 30, Mn: 15, Zn: 2, B: 1 },
        { N: 120, P: 30, K: 60, ph: 5.8, EC: 0.8, S: 15, Cu: 3, Fe: 70, Mn: 25, Zn: 1, B: 0 },
    ];
    const sample = samples[Math.floor(Math.random() * samples.length)];
    Object.keys(sample).forEach(key => {
        const el = document.getElementById(`soil-${key}`);
        if (el) el.value = sample[key];
    });
    Toast.info('Sample soil data filled in! Click "Get Recommendations" to analyze.');
}

async function submitCropForm() {
    const btn = document.getElementById('crop-submit');
    const panel = document.getElementById('crop-results-panel');

    // Gather values
    const fields = ['N', 'P', 'K', 'ph', 'EC', 'S', 'Cu', 'Fe', 'Mn', 'Zn', 'B'];
    const params = {};
    let valid = true;

    ['N', 'P', 'K', 'ph'].forEach(f => {
        const el = document.getElementById(`soil-${f}`);
        if (!el.value) {
            el.style.borderColor = 'var(--danger)';
            valid = false;
        } else {
            el.style.borderColor = '';
        }
    });

    if (!valid) {
        Toast.warning('Please fill in the required fields (N, P, K, pH)');
        return;
    }

    fields.forEach(f => {
        const el = document.getElementById(`soil-${f}`);
        const val = parseFloat(el.value) || 0;
        // B must be int
        params[f] = (f === 'Cu' || f === 'Fe' || f === 'Mn' || f === 'Zn' || f === 'B' || f === 'N' || f === 'P' || f === 'K') ? Math.round(val) : val;
    });

    setLoading(btn, true);
    panel.innerHTML = `<div class="card"><div class="card-body">${createSkeleton(5)}</div></div>`;

    try {
        const result = await api.getCropRecommendation(params);
        
        const recommendations = result.top_recommendations || [];
        const topCrop = recommendations[0];

        panel.innerHTML = `
            <div class="card scale-in" style="margin-bottom: 16px;">
                <div style="text-align:center; padding: 16px 0;">
                    <span style="font-size:48px; display:block; margin-bottom:8px;">${getCropEmoji(topCrop?.crop || 'crop')}</span>
                    <h3 style="font-size:22px; font-weight:700;">${topCrop?.crop || 'Unknown'}</h3>
                    <span class="badge badge-green" style="margin-top:8px;">Top Recommendation</span>
                    <div style="margin-top:16px;">
                        ${createProgressRing(topCrop?.confidence || 0, 90)}
                    </div>
                    <p style="font-size:13px; color:var(--text-muted); margin-top:8px;">Confidence Score</p>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <div class="card-title"><i class="fas fa-ranking-star"></i> All Recommendations</div>
                </div>
                <div class="card-body stagger-children">
                    ${recommendations.map((rec, i) => `
                        <div class="crop-result-card">
                            <div class="crop-icon">${getCropEmoji(rec.crop)}</div>
                            <div class="crop-info">
                                <div class="crop-name">#${i + 1} ${rec.crop}</div>
                                <div style="font-size:12px;color:var(--text-muted)">${(rec.confidence * 100).toFixed(1)}% match</div>
                            </div>
                            <div class="crop-confidence">
                                ${createConfidenceBar(rec.confidence)}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>

            ${result.recommendation_text ? `
                <div class="card" style="margin-top:16px;">
                    <div class="card-header">
                        <div class="card-title"><i class="fas fa-lightbulb"></i> Analysis</div>
                    </div>
                    <div class="card-body">
                        <p style="font-size:14px; line-height:1.7; color:var(--text-secondary);">${result.recommendation_text}</p>
                    </div>
                </div>
            ` : ''}
        `;

        Toast.success(`Top recommendation: ${topCrop?.crop} (${(topCrop?.confidence * 100).toFixed(1)}%)`);
    } catch (error) {
        panel.innerHTML = `
            <div class="card">
                <div class="empty-state">
                    <i class="fas fa-exclamation-triangle" style="color:var(--danger)"></i>
                    <h3>Analysis Failed</h3>
                    <p>${error.message}</p>
                </div>
            </div>
        `;
        Toast.error('Crop analysis failed: ' + error.message);
    }

    setLoading(btn, false);
}
