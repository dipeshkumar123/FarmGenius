/* ============================================
   API SERVICE LAYER
   Handles all communication with the FarmGenius API
   ============================================ */

const API_BASE = window.location.origin + '/api';

class FarmAPI {
    constructor() {
        this.baseUrl = API_BASE;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const headers = { 'Content-Type': 'application/json', ...options.headers };

        // Attach JWT token if available
        const token = typeof authManager !== 'undefined' ? authManager.getToken() : null;
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        // Don't set Content-Type for FormData (browser sets boundary automatically)
        if (options.body instanceof FormData) {
            delete headers['Content-Type'];
        }

        const config = { ...options, headers };

        try {
            const response = await fetch(url, config);
            const data = await response.json();

            if (!response.ok) {
                // If 401 and we have a token, it's expired → logout
                if (response.status === 401 && token && typeof authManager !== 'undefined') {
                    authManager.logout();
                }
                const errMsg = Array.isArray(data.detail) 
                    ? data.detail.map(e => e.msg || e.message || JSON.stringify(e)).join('; ')
                    : (data.detail || data.message || `HTTP ${response.status}`);
                throw new Error(errMsg);
            }
            return data;
        } catch (error) {
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('Cannot connect to FarmGenius API. Make sure the server is running.');
            }
            throw error;
        }
    }

    // ── Auth ──
    async register(email, password, displayName = null) {
        return this.request('/auth/register', {
            method: 'POST',
            body: JSON.stringify({ email, password, display_name: displayName })
        });
    }

    async login(email, password) {
        return this.request('/auth/login', {
            method: 'POST',
            body: JSON.stringify({ email, password })
        });
    }

    async getMe() {
        return this.request('/auth/me');
    }

    async updateProfile(data) {
        return this.request('/auth/profile', {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    // ── Health ──
    async checkHealth() {
        return this.request('/health');
    }

    // ── Chat / Query ──
    async sendQuery(query, userId = null, targetLang = null) {
        const uid = userId || (typeof authManager !== 'undefined' ? authManager.getUserId() : 'guest');
        const body = { query, user_id: uid };
        if (targetLang) body.target_lang = targetLang;
        return this.request('/query', {
            method: 'POST',
            body: JSON.stringify(body)
        });
    }

    // ── Crop Recommendation ──
    async getCropRecommendation(soilParams) {
        return this.request('/crops/recommend', {
            method: 'POST',
            body: JSON.stringify(soilParams)
        });
    }

    // ── Disease Detection ──
    async identifyDiseaseImage(file, crop = null, userId = 'guest') {
        const formData = new FormData();
        formData.append('file', file);
        if (crop) formData.append('crop', crop);
        formData.append('user_id', userId);

        const url = `${this.baseUrl}/diseases/identify-image`;
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || data.message || 'Upload failed');
        return data;
    }

    // ── Weather ──
    async getCurrentWeather(location, userId = 'guest') {
        return this.request('/weather/current', {
            method: 'POST',
            body: JSON.stringify({ location, user_id: userId })
        });
    }

    async getWeatherForecast(location, days = 5, userId = 'guest') {
        return this.request('/weather/forecast', {
            method: 'POST',
            body: JSON.stringify({ location, days, user_id: userId })
        });
    }

    async getCropWeatherAdvice(location, crop, userId = 'guest') {
        return this.request('/weather/crop-advice', {
            method: 'POST',
            body: JSON.stringify({ location, crop, user_id: userId })
        });
    }

    // ── Prices ──
    async getPrice(commodity, date = null, includeTrends = false) {
        const body = { commodity, include_trends: includeTrends };
        if (date) body.date = date;
        return this.request('/prices/get', {
            method: 'POST',
            body: JSON.stringify(body)
        });
    }

    async getCommodities() {
        return this.request('/prices/commodities');
    }

    async getPriceTrends(commodity) {
        return this.request(`/prices/trends?commodity=${encodeURIComponent(commodity)}`, {
            method: 'POST'
        });
    }

    // ── History ──
    async getHistory(userId = null, maxEntries = 50) {
        // Use authenticated endpoint if logged in
        if (typeof authManager !== 'undefined' && authManager.isLoggedIn()) {
            return this.request('/my/history?limit=' + maxEntries);
        }
        const uid = userId || 'guest';
        return this.request('/history', {
            method: 'POST',
            body: JSON.stringify({ user_id: uid, max_entries: maxEntries })
        });
    }

    async clearHistory() {
        return this.request('/my/history', { method: 'DELETE' });
    }

    async getMyStats() {
        return this.request('/my/stats');
    }

    // ── Languages ──
    async getLanguages() {
        return this.request('/languages');
    }

    // ── Translation ──
    async translate(text, targetLang, sourceLang = null) {
        const body = { text, target_lang: targetLang };
        if (sourceLang) body.source_lang = sourceLang;
        return this.request('/translate', {
            method: 'POST',
            body: JSON.stringify(body)
        });
    }
}

// Singleton
const api = new FarmAPI();
