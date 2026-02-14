/* ============================================
   DISEASE DETECTION PAGE
   ============================================ */

let selectedDiseaseFile = null;

function renderDiseases(container) {
    container.innerHTML = `
        <div class="page-header">
            <h2><i class="fas fa-microscope"></i> Disease Scanner</h2>
            <p>Upload a photo of your crop leaf to identify diseases using AI image analysis</p>
        </div>

        <div class="grid-2" style="grid-template-columns: 1fr 1fr;">
            <!-- Upload Area -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title"><i class="fas fa-camera"></i> Upload Image</div>
                    <span class="badge badge-blue"><i class="fas fa-brain"></i> MobileNetV2</span>
                </div>
                <div class="card-body">
                    <div class="drop-zone" id="disease-dropzone" 
                        onclick="document.getElementById('disease-file-input').click()"
                        ondragover="event.preventDefault(); this.classList.add('dragover')"
                        ondragleave="this.classList.remove('dragover')"
                        ondrop="handleDiseaseFileDrop(event)">
                        <i class="fas fa-cloud-arrow-up"></i>
                        <h3>Drop your leaf image here</h3>
                        <p>or click to browse — JPG, JPEG, PNG supported</p>
                    </div>
                    <input type="file" id="disease-file-input" accept=".jpg,.jpeg,.png" 
                        style="display:none" onchange="handleDiseaseFileSelect(event)">

                    <div id="disease-preview" style="margin-top:16px; display:none;">
                        <div class="image-preview" id="disease-image-container">
                            <img id="disease-preview-img" src="" alt="Selected image">
                            <button class="remove-btn" onclick="removeDiseaseImage()">
                                <i class="fas fa-xmark"></i>
                            </button>
                        </div>
                        <div style="margin-top:12px; display:flex; gap:10px; align-items:center;">
                            <span id="disease-filename" style="font-size:13px; color:var(--text-muted); flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;"></span>
                            <span id="disease-filesize" class="badge badge-gray"></span>
                        </div>
                    </div>

                    <div style="margin-top:20px;">
                        <button class="btn btn-primary btn-lg btn-block" id="disease-submit" 
                            onclick="submitDiseaseImage()" disabled>
                            <i class="fas fa-search"></i> Analyze Image
                        </button>
                    </div>

                    <div style="margin-top:16px; padding:12px; background:var(--bg-input); border-radius:var(--radius-md);">
                        <p style="font-size:12px; color:var(--text-muted); display:flex; align-items:flex-start; gap:8px;">
                            <i class="fas fa-info-circle" style="margin-top:2px; color:var(--accent-500);"></i>
                            Supported crops: <strong>Cashew, Cassava, Maize, Tomato</strong>. 
                            For best results, upload a clear, well-lit photo of the affected leaf.
                        </p>
                    </div>
                </div>
            </div>

            <!-- Results Panel -->
            <div id="disease-results-panel">
                <div class="card" style="height:100%;">
                    <div class="empty-state" style="padding-top:100px;">
                        <i class="fas fa-leaf animate-float" style="color:var(--primary-400)"></i>
                        <h3>Upload a Leaf Image</h3>
                        <p>Our AI will analyze the image and identify potential diseases, along with symptoms and treatment recommendations</p>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function handleDiseaseFileSelect(event) {
    const file = event.target.files[0];
    if (file) loadDiseaseImage(file);
}

function handleDiseaseFileDrop(event) {
    event.preventDefault();
    event.target.classList.remove('dragover');
    const file = event.dataTransfer.files[0];
    if (file) loadDiseaseImage(file);
}

function loadDiseaseImage(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        Toast.error('Please upload a JPG, JPEG, or PNG image');
        return;
    }

    // Validate size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        Toast.error('Image too large. Max size is 10MB.');
        return;
    }

    selectedDiseaseFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('disease-preview-img').src = e.target.result;
        document.getElementById('disease-preview').style.display = 'block';
        document.getElementById('disease-dropzone').style.display = 'none';
        document.getElementById('disease-filename').textContent = file.name;
        document.getElementById('disease-filesize').textContent = formatFileSize(file.size);
        document.getElementById('disease-submit').disabled = false;
    };
    reader.readAsDataURL(file);
}

function removeDiseaseImage() {
    selectedDiseaseFile = null;
    document.getElementById('disease-preview').style.display = 'none';
    document.getElementById('disease-dropzone').style.display = '';
    document.getElementById('disease-submit').disabled = true;
    document.getElementById('disease-file-input').value = '';
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function submitDiseaseImage() {
    if (!selectedDiseaseFile) return;

    const btn = document.getElementById('disease-submit');
    const panel = document.getElementById('disease-results-panel');
    
    setLoading(btn, true);
    panel.innerHTML = `
        <div class="card">
            <div class="card-body" style="padding:40px; text-align:center;">
                <div class="leaf-spinner" style="margin:0 auto 20px;">
                    <div class="leaf leaf-1"><i class="fas fa-leaf"></i></div>
                    <div class="leaf leaf-2"><i class="fas fa-leaf"></i></div>
                    <div class="leaf leaf-3"><i class="fas fa-leaf"></i></div>
                </div>
                <h3 style="font-size:16px;">Analyzing Image...</h3>
                <p style="font-size:13px; color:var(--text-muted); margin-top:4px;">Our AI is examining the leaf for disease patterns</p>
            </div>
        </div>
    `;

    try {
        const result = await api.identifyDiseaseImage(selectedDiseaseFile);
        
        if (!result.found || !result.results || result.results.length === 0) {
            panel.innerHTML = `
                <div class="card">
                    <div class="empty-state">
                        <i class="fas fa-check-circle" style="color:var(--success)"></i>
                        <h3>No Disease Detected</h3>
                        <p>The leaf appears to be healthy! If you believe this is incorrect, try uploading a clearer image.</p>
                    </div>
                </div>
            `;
            Toast.success('No disease detected — leaf appears healthy!');
            setLoading(btn, false);
            return;
        }

        const topResult = result.results[0];

        panel.innerHTML = `
            <!-- Top Result -->
            <div class="card scale-in" style="margin-bottom:16px; border-left: 4px solid ${getSeverityColor(topResult.severity)};">
                <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px;">
                    <div style="width:56px;height:56px;border-radius:var(--radius-md);background:${getSeverityBg(topResult.severity)};display:flex;align-items:center;justify-content:center;font-size:24px;">
                        ${getDiseaseIcon(topResult.type)}
                    </div>
                    <div style="flex:1;">
                        <h3 style="font-size:18px; font-weight:700;">${topResult.name}</h3>
                        <div style="display:flex; gap:8px; margin-top:4px; flex-wrap:wrap;">
                            <span class="badge badge-blue">${topResult.crop}</span>
                            <span class="badge badge-gray">${topResult.type}</span>
                            ${severityBadge(topResult.severity)}
                        </div>
                    </div>
                    <div style="text-align:center;">
                        ${createProgressRing(topResult.confidence, 70)}
                        <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">Confidence</div>
                    </div>
                </div>

                <div class="disease-detail-grid">
                    <div class="disease-detail-section">
                        <h4><i class="fas fa-stethoscope"></i> Symptoms</h4>
                        <ul>
                            ${(topResult.symptoms || []).map(s => `<li>${s}</li>`).join('')}
                        </ul>
                    </div>
                    <div class="disease-detail-section">
                        <h4><i class="fas fa-prescription"></i> Treatment</h4>
                        <ul>
                            ${(topResult.treatment || []).map(t => `<li>${t}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Other Possible Diseases -->
            ${result.results.length > 1 ? `
                <div class="card">
                    <div class="card-header">
                        <div class="card-title"><i class="fas fa-list-ol"></i> Other Possibilities</div>
                    </div>
                    <div class="card-body stagger-children">
                        ${result.results.slice(1).map((d, i) => `
                            <div class="disease-card" id="disease-card-${i}">
                                <div class="disease-card-header" onclick="toggleDiseaseCard(${i})">
                                    <div class="disease-name">
                                        <span class="severity-dot ${d.severity}"></span>
                                        ${d.name}
                                        <span class="badge badge-gray" style="font-size:11px;">${(d.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <i class="fas fa-chevron-down" style="font-size:12px; color:var(--text-muted); transition:transform 0.2s;"></i>
                                </div>
                                <div class="disease-card-body">
                                    <div class="disease-detail-grid">
                                        <div class="disease-detail-section">
                                            <h4><i class="fas fa-stethoscope"></i> Symptoms</h4>
                                            <ul>${(d.symptoms || []).map(s => `<li>${s}</li>`).join('')}</ul>
                                        </div>
                                        <div class="disease-detail-section">
                                            <h4><i class="fas fa-prescription"></i> Treatment</h4>
                                            <ul>${(d.treatment || []).map(t => `<li>${t}</li>`).join('')}</ul>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        `;

        Toast.success(`Detected: ${topResult.name} (${(topResult.confidence * 100).toFixed(1)}%)`);
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
        Toast.error('Disease analysis failed: ' + error.message);
    }

    setLoading(btn, false);
}

function toggleDiseaseCard(index) {
    const card = document.getElementById(`disease-card-${index}`);
    if (card) {
        card.classList.toggle('expanded');
        const chevron = card.querySelector('.fa-chevron-down');
        if (chevron) {
            chevron.style.transform = card.classList.contains('expanded') ? 'rotate(180deg)' : '';
        }
    }
}

function getSeverityColor(severity) {
    const colors = { high: '#ef4444', medium: '#f59e0b', low: '#22c55e' };
    return colors[severity] || '#64748b';
}

function getSeverityBg(severity) {
    const bgs = { high: '#fee2e2', medium: '#fef3c7', low: '#dcfce7' };
    return bgs[severity] || '#f1f5f9';
}

function getDiseaseIcon(type) {
    const icons = {
        'Fungal': '🍄', 'Bacterial': '🦠', 'Viral': '🧬', 'Pest': '🐛'
    };
    return icons[type] || '🔬';
}
