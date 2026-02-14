/* ============================================
   SHARED COMPONENTS & UTILITIES
   ============================================ */

// ── Toast Notification System ──
const Toast = {
    show(message, type = 'info', duration = 4000) {
        const container = document.getElementById('toast-container');
        const icons = {
            success: 'fa-circle-check',
            error: 'fa-circle-xmark',
            warning: 'fa-triangle-exclamation',
            info: 'fa-circle-info'
        };

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-icon"><i class="fas ${icons[type]}"></i></div>
            <div class="toast-message">${message}</div>
            <button class="toast-close" onclick="this.closest('.toast').remove()">
                <i class="fas fa-xmark"></i>
            </button>
        `;

        container.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('toast-exit');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    },

    success(msg) { this.show(msg, 'success'); },
    error(msg) { this.show(msg, 'error', 6000); },
    warning(msg) { this.show(msg, 'warning'); },
    info(msg) { this.show(msg, 'info'); }
};

// ── Loading State Helpers ──
function setLoading(button, loading) {
    if (loading) {
        button.dataset.originalText = button.innerHTML;
        button.disabled = true;
        button.innerHTML = '<span class="spinner"></span> Processing...';
    } else {
        button.disabled = false;
        button.innerHTML = button.dataset.originalText || button.innerHTML;
    }
}

// ── Skeleton Loader ──
function createSkeleton(lines = 3) {
    let html = '<div class="skeleton-loader">';
    for (let i = 0; i < lines; i++) {
        const w = 60 + Math.random() * 30;
        html += `<div class="skeleton skeleton-text" style="width:${w}%"></div>`;
    }
    html += '</div>';
    return html;
}

// ── Confidence Bar ──
function createConfidenceBar(value, maxWidth = '100%') {
    const pct = Math.round(value * 100);
    const level = pct >= 70 ? 'high' : pct >= 40 ? 'medium' : 'low';
    return `
        <div class="confidence-bar" style="max-width:${maxWidth}">
            <div class="confidence-fill ${level}" style="width:${pct}%"></div>
        </div>
    `;
}

// ── Progress Ring ──
function createProgressRing(value, size = 80) {
    const pct = Math.round(value * 100);
    const r = (size - 12) / 2;
    const c = Math.PI * 2 * r;
    const offset = c - (pct / 100) * c;
    const center = size / 2;
    
    return `
        <div class="progress-ring" style="width:${size}px;height:${size}px">
            <svg width="${size}" height="${size}">
                <circle class="track" cx="${center}" cy="${center}" r="${r}" />
                <circle class="fill" cx="${center}" cy="${center}" r="${r}"
                    stroke-dasharray="${c}" stroke-dashoffset="${offset}" />
            </svg>
            <div class="value">${pct}%</div>
        </div>
    `;
}

// ── Crop Emoji Mapping ──
function getCropEmoji(cropName) {
    const map = {
        'rice': '🌾', 'wheat': '🌾', 'maize': '🌽', 'corn': '🌽',
        'sugarcane': '🎋', 'cotton': '🏵️', 'tomato': '🍅',
        'potato': '🥔', 'onion': '🧅', 'mango': '🥭',
        'banana': '🍌', 'pomegranate': '🍎', 'groundnut': '🥜',
        'soybean': '🫘', 'coffee': '☕', 'tea': '🍵',
        'coconut': '🥥', 'cashew': '🥜', 'cassava': '🌿',
        'chickpea': '🫛', 'lentil': '🫘', 'mustard': '🟡',
        'chilli': '🌶️', 'turmeric': '🟡', 'ginger': '🫚',
    };
    const lower = cropName.toLowerCase();
    for (const [key, emoji] of Object.entries(map)) {
        if (lower.includes(key)) return emoji;
    }
    return '🌱';
}

// ── Weather Icon Mapping ──
function getWeatherIcon(condition) {
    if (!condition) return '🌤️';
    const c = condition.toLowerCase();
    if (c.includes('clear') || c.includes('sunny')) return '☀️';
    if (c.includes('cloud') && c.includes('part')) return '⛅';
    if (c.includes('cloud') || c.includes('overcast')) return '☁️';
    if (c.includes('rain') && c.includes('heavy')) return '🌧️';
    if (c.includes('rain') || c.includes('drizzle')) return '🌦️';
    if (c.includes('thunder') || c.includes('storm')) return '⛈️';
    if (c.includes('snow')) return '🌨️';
    if (c.includes('fog') || c.includes('mist')) return '🌫️';
    if (c.includes('wind')) return '💨';
    return '🌤️';
}

// ── Time Formatting ──
function formatTimestamp(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    return d.toLocaleString('en-US', {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit'
    });
}

function timeAgo(ts) {
    if (!ts) return '';
    const seconds = Math.floor((new Date() - new Date(ts)) / 1000);
    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
}

// ── Severity Badge ──
function severityBadge(severity) {
    const colors = { high: 'badge-red', medium: 'badge-orange', low: 'badge-green' };
    return `<span class="badge ${colors[severity] || 'badge-gray'}">${severity}</span>`;
}

// ── Animate Counter ──
function animateCounter(element, target, duration = 1000) {
    let start = 0;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = Math.round(eased * target);
        
        element.textContent = current.toLocaleString();
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    requestAnimationFrame(update);
}

// ── De-bounce ──
function debounce(fn, delay = 300) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), delay);
    };
}
