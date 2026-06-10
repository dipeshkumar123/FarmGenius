/* ============================================
   WEATHER PAGE
   ============================================ */

let weatherChart = null;

/**
 * Extract forecast days array from the nested API response.
 * API returns: { forecast_data: { forecast_days: [...] } }
 */
function extractForecastDays(forecastResponse) {
    if (!forecastResponse) return [];
    // Direct array
    if (Array.isArray(forecastResponse)) return forecastResponse;
    // { forecast_data: { forecast_days: [...] } }
    const fd = forecastResponse.forecast_data;
    if (fd) {
        if (Array.isArray(fd.forecast_days)) return fd.forecast_days;
        if (Array.isArray(fd)) return fd;
    }
    // { forecast: [...] }
    if (Array.isArray(forecastResponse.forecast)) return forecastResponse.forecast;
    // { forecast_days: [...] }
    if (Array.isArray(forecastResponse.forecast_days)) return forecastResponse.forecast_days;
    return [];
}

function renderWeather(container) {
    container.innerHTML = `
        <div class="page-header">
            <h2><i class="fas fa-cloud-sun"></i> Weather Intelligence</h2>
            <p>Get real-time weather data and agricultural advisory for any location</p>
        </div>

        <!-- Search Bar -->
        <div class="card" style="margin-bottom:24px;">
            <div class="card-body" style="padding:16px 20px;">
                <div style="display:flex; gap:12px; align-items:center;">
                    <div style="flex:1; position:relative;">
                        <i class="fas fa-map-marker-alt" style="position:absolute; left:14px; top:50%; transform:translateY(-50%); color:var(--text-muted);"></i>
                        <input type="text" id="weather-location" class="form-input" 
                            placeholder="Enter city or location (e.g., Mumbai, Delhi, Bangalore)" 
                            style="padding-left:40px;"
                            onkeydown="if(event.key==='Enter') fetchWeather()">
                    </div>
                    <button class="btn btn-primary" id="weather-search-btn" onclick="fetchWeather()">
                        <i class="fas fa-search"></i> Get Weather
                    </button>
                </div>
                <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
                    <span style="font-size:12px; color:var(--text-muted); line-height:26px;">Quick:</span>
                    <button class="btn btn-ghost btn-sm" onclick="setWeatherLocation('Mumbai')">Mumbai</button>
                    <button class="btn btn-ghost btn-sm" onclick="setWeatherLocation('Delhi')">Delhi</button>
                    <button class="btn btn-ghost btn-sm" onclick="setWeatherLocation('Bangalore')">Bangalore</button>
                    <button class="btn btn-ghost btn-sm" onclick="setWeatherLocation('Hyderabad')">Hyderabad</button>
                    <button class="btn btn-ghost btn-sm" onclick="setWeatherLocation('Pune')">Pune</button>
                </div>
            </div>
        </div>

        <div id="weather-content">
            <div class="card" style="height: 300px;">
                <div class="empty-state">
                    <i class="fas fa-cloud-sun animate-float" style="color:var(--accent-500)"></i>
                    <h3>Search for a Location</h3>
                    <p>Enter a city name above to get current weather conditions, forecast, and crop-specific advisory</p>
                </div>
            </div>
        </div>
    `;
}

function setWeatherLocation(city) {
    document.getElementById('weather-location').value = city;
    fetchWeather();
}

async function fetchWeather() {
    const location = document.getElementById('weather-location').value.trim();
    if (!location) {
        Toast.warning('Please enter a location');
        return;
    }

    const btn = document.getElementById('weather-search-btn');
    const content = document.getElementById('weather-content');
    setLoading(btn, true);

    content.innerHTML = `
        <div class="grid-2" style="grid-template-columns: 1fr 1fr; gap:16px;">
            <div class="card"><div class="card-body">${createSkeleton(4, 'medium')}</div></div>
            <div class="card"><div class="card-body">${createSkeleton(4, 'medium')}</div></div>
        </div>
    `;

    try {
        // Fetch current weather, forecast, and crop advice in parallel
        const [current, forecast, advice] = await Promise.allSettled([
            api.getCurrentWeather(location),
            api.getWeatherForecast(location),
            api.getCropWeatherAdvice(location, 'rice')
        ]);

        const currentData = current.status === 'fulfilled' ? current.value : null;
        const forecastData = forecast.status === 'fulfilled' ? forecast.value : null;
        const adviceData = advice.status === 'fulfilled' ? advice.value : null;

        if (!currentData && !forecastData) {
            throw new Error('Could not fetch weather data for this location');
        }

        content.innerHTML = buildWeatherDisplay(currentData, forecastData, adviceData, location);
        
        // Build forecast chart if data available
        const forecastDays = extractForecastDays(forecastData);
        if (forecastDays && forecastDays.length > 0) {
            buildForecastChart(forecastDays);
        }

        Toast.success(`Weather loaded for ${location}`);
    } catch (error) {
        content.innerHTML = `
            <div class="card">
                <div class="empty-state">
                    <i class="fas fa-exclamation-triangle" style="color:var(--danger)"></i>
                    <h3>Weather Unavailable</h3>
                    <p>${error.message}</p>
                </div>
            </div>
        `;
        Toast.error('Failed to load weather: ' + error.message);
    }

    setLoading(btn, false);
}

function buildWeatherDisplay(current, forecast, advice, location) {
    // Extract from nested API response: { weather_data: { temperature_c, humidity, ... } }
    const wd = (current && current.weather_data) ? current.weather_data : (current || {});
    const temp = wd.temperature_c || wd.temperature || wd.temp || '--';
    const humidity = wd.humidity || '--';
    const windSpeed = wd.wind_speed_kmh || wd.wind_speed || wd.windSpeed || '--';
    const condition = wd.condition || wd.description || 'Unknown';
    const feelsLike = wd.feels_like || wd.feelsLike || temp;
    const pressure = wd.pressure || '--';
    const windDir = wd.wind_direction || '';
    const precip = wd.precipitation_mm || 0;
    const icon = getWeatherIcon(condition);

    return `
        <!-- Current Weather -->
        <div class="card scale-in" style="margin-bottom:20px; overflow:hidden;">
            <div class="weather-current">
                <div class="weather-main">
                    <div class="weather-icon-big">${icon}</div>
                    <div>
                        <div class="weather-temp">${temp}°C</div>
                        <div class="weather-condition">${condition}</div>
                        <div class="weather-location"><i class="fas fa-map-marker-alt"></i> ${location}</div>
                    </div>
                </div>
                <div class="weather-details-grid">
                    <div class="weather-detail">
                        <i class="fas fa-tint" style="color:#3b82f6"></i>
                        <div>
                            <span class="weather-detail-value">${humidity}%</span>
                            <span class="weather-detail-label">Humidity</span>
                        </div>
                    </div>
                    <div class="weather-detail">
                        <i class="fas fa-wind" style="color:#64748b"></i>
                        <div>
                            <span class="weather-detail-value">${windSpeed} km/h</span>
                            <span class="weather-detail-label">Wind Speed</span>
                        </div>
                    </div>
                    <div class="weather-detail">
                        <i class="fas fa-temperature-half" style="color:#f59e0b"></i>
                        <div>
                            <span class="weather-detail-value">${feelsLike}°C</span>
                            <span class="weather-detail-label">Feels Like</span>
                        </div>
                    </div>
                    <div class="weather-detail">
                        <i class="fas fa-gauge-high" style="color:#8b5cf6"></i>
                        <div>
                            <span class="weather-detail-value">${pressure} hPa</span>
                            <span class="weather-detail-label">Pressure</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="grid-2" style="grid-template-columns: 1fr 1fr; gap:16px; margin-bottom:20px;">
            <!-- Forecast Chart -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title"><i class="fas fa-chart-line"></i> Forecast</div>
                </div>
                <div class="card-body">
                    ${extractForecastDays(forecast).length > 0 ? `
                        <canvas id="forecast-chart" height="200"></canvas>
                    ` : `
                        <div class="empty-state" style="padding:30px 0;">
                            <i class="fas fa-calendar-xmark" style="color:var(--text-muted)"></i>
                            <p>Forecast data unavailable</p>
                        </div>
                    `}
                </div>
            </div>

            <!-- Forecast Cards -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title"><i class="fas fa-calendar-days"></i> Daily Details</div>
                </div>
                <div class="card-body stagger-children" style="max-height:300px; overflow-y:auto;">
                    ${(function() { const days = extractForecastDays(forecast); return days.length > 0 ? days.map(day => `
                        <div class="forecast-day-card">
                            <div class="forecast-day-date">
                                <span class="forecast-icon">${getWeatherIcon(day.condition || '')}</span>
                                <span>${day.date || day.day || '--'}</span>
                            </div>
                            <div class="forecast-day-temps">
                                <span class="temp-high"><i class="fas fa-arrow-up"></i> ${day.temp_c_max || day.max_temp || day.high || '--'}°</span>
                                <span class="temp-low"><i class="fas fa-arrow-down"></i> ${day.temp_c_min || day.min_temp || day.low || '--'}°</span>
                            </div>
                            <div class="forecast-condition">${day.condition || '--'}</div>
                        </div>
                    `).join('') : '<p style="color:var(--text-muted); text-align:center;">No forecast data</p>'; })()}
                </div>
            </div>
        </div>

        <!-- Crop Advisory -->
        ${advice ? `
            <div class="card scale-in">
                <div class="card-header">
                    <div class="card-title"><i class="fas fa-seedling"></i> Agricultural Advisory</div>
                    <span class="badge badge-green">AI-Powered</span>
                </div>
                <div class="card-body">
                    <div class="advisory-content">
                        ${(function() {
                            // advice from API: { advice: [...], crop_preferences: {...}, weather_data: {...} }
                            const adviceList = advice.advice || advice.recommendations || [];
                            const cropPrefs = advice.crop_preferences || {};
                            if (Array.isArray(adviceList) && adviceList.length > 0) {
                                return `
                                    <h4><i class="fas fa-lightbulb" style="color:#f59e0b;"></i> Advisory Notes</h4>
                                    <ul style="margin-top:8px;">
                                        ${adviceList.map(r => '<li>' + r + '</li>').join('')}
                                    </ul>
                                    ${Object.keys(cropPrefs).length > 0 ? '<h4 style="margin-top:16px;"><i class="fas fa-wheat-awn" style="color:var(--primary-500);"></i> Crop: ' + (advice.crop || '') + '</h4>' : ''}
                                `;
                            } else if (advice.advisory) {
                                return '<p style="font-size:15px; line-height:1.7;">' + advice.advisory + '</p>';
                            } else {
                                return '<p>No specific advisory available for this location.</p>';
                            }
                        })()}
                    </div>
                </div>
            </div>
        ` : ''}
    `;
}

function buildForecastChart(forecastDays) {
    const canvas = document.getElementById('forecast-chart');
    if (!canvas) return;

    if (weatherChart) weatherChart.destroy();

    const labels = forecastDays.map(d => d.date || d.day || '');
    const highs = forecastDays.map(d => d.temp_c_max || d.max_temp || d.high || 0);
    const lows = forecastDays.map(d => d.temp_c_min || d.min_temp || d.low || 0);

    const ctx = canvas.getContext('2d');
    
    const primary = getComputedStyle(document.documentElement).getPropertyValue('--primary-500').trim() || '#22c55e';
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent-500').trim() || '#0ea5e9';
    const textColor = getComputedStyle(document.documentElement).getPropertyValue('--text-muted').trim() || '#64748b';

    weatherChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'High °C',
                    data: highs,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239,68,68,0.1)',
                    tension: 0.4,
                    fill: false,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    borderWidth: 2
                },
                {
                    label: 'Low °C',
                    data: lows,
                    borderColor: accent,
                    backgroundColor: 'rgba(14,165,233,0.1)',
                    tension: 0.4,
                    fill: false,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { 
                    position: 'top',
                    labels: { color: textColor, usePointStyle: true, padding: 12 }
                },
                tooltip: {
                    backgroundColor: 'rgba(15,23,42,0.9)',
                    cornerRadius: 8,
                    padding: 10
                }
            },
            scales: {
                y: { 
                    ticks: { color: textColor, callback: v => v + '°' },
                    grid: { color: 'rgba(148,163,184,0.1)' }
                },
                x: { 
                    ticks: { color: textColor, maxRotation: 45 },
                    grid: { display: false }
                }
            }
        }
    });
}
