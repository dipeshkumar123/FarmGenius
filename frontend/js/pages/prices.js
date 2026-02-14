/* ============================================
   MARKET PRICES PAGE
   ============================================ */

let priceChart = null;

function renderPrices(container) {
    container.innerHTML = `
        <div class="page-header">
            <h2><i class="fas fa-indian-rupee-sign"></i> Market Prices</h2>
            <p>Check current commodity prices and market trends</p>
        </div>

        <!-- Search & Filter -->
        <div class="card" style="margin-bottom:24px;">
            <div class="card-body" style="padding:16px 20px;">
                <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
                    <div style="flex:1; min-width:200px;">
                        <select id="price-commodity" class="form-select" onchange="fetchPrice()">
                            <option value="">— Select Commodity —</option>
                        </select>
                    </div>
                    <button class="btn btn-primary" id="price-search-btn" onclick="fetchPrice()">
                        <i class="fas fa-search"></i> Get Price
                    </button>
                    <button class="btn btn-outline" id="price-trends-btn" onclick="fetchTrends()">
                        <i class="fas fa-chart-line"></i> Trends
                    </button>
                </div>
                <div style="margin-top:12px;" id="price-quick-commodities">
                    <span style="font-size:12px; color:var(--text-muted); line-height:26px;">Popular:</span>
                </div>
            </div>
        </div>

        <div id="price-content">
            <div class="card" style="height: 280px;">
                <div class="empty-state">
                    <i class="fas fa-store animate-float" style="color:var(--secondary-500)"></i>
                    <h3>Select a Commodity</h3>
                    <p>Choose a commodity above to view current market prices, historical trends, and market analysis</p>
                </div>
            </div>
        </div>
    `;

    // Load available commodities
    loadCommodities();
}

async function loadCommodities() {
    const select = document.getElementById('price-commodity');
    const quickBar = document.getElementById('price-quick-commodities');
    
    try {
        const result = await api.getCommodities();
        const commodities = result.commodities || result || [];

        if (Array.isArray(commodities) && commodities.length > 0) {
            commodities.forEach(c => {
                const name = typeof c === 'string' ? c : c.name || c.commodity;
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = `${getCropEmoji(name)} ${name}`;
                select.appendChild(opt);
            });

            // Quick buttons for first 6
            const quickItems = commodities.slice(0, 6);
            quickItems.forEach(c => {
                const name = typeof c === 'string' ? c : c.name || c.commodity;
                const btn = document.createElement('button');
                btn.className = 'btn btn-ghost btn-sm';
                btn.textContent = `${getCropEmoji(name)} ${name}`;
                btn.onclick = () => {
                    select.value = name;
                    fetchPrice();
                };
                quickBar.appendChild(btn);
            });
        }
    } catch (err) {
        // Fallback commodity list
        const fallback = ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize', 'Soybean'];
        fallback.forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = `${getCropEmoji(name)} ${name}`;
            select.appendChild(opt);
            
            const btn = document.createElement('button');
            btn.className = 'btn btn-ghost btn-sm';
            btn.textContent = `${getCropEmoji(name)} ${name}`;
            btn.onclick = () => { select.value = name; fetchPrice(); };
            quickBar.appendChild(btn);
        });
    }
}

async function fetchPrice() {
    const commodity = document.getElementById('price-commodity').value;
    if (!commodity) {
        Toast.warning('Please select a commodity');
        return;
    }

    const btn = document.getElementById('price-search-btn');
    const content = document.getElementById('price-content');
    setLoading(btn, true);

    content.innerHTML = `
        <div class="grid-2" style="grid-template-columns: 1fr 1fr; gap:16px;">
            <div class="card"><div class="card-body">${createSkeleton(3, 'medium')}</div></div>
            <div class="card"><div class="card-body">${createSkeleton(3, 'medium')}</div></div>
        </div>
    `;

    try {
        const result = await api.getPrice(commodity);
        
        const price = result.price || result.current_price || result.avg_price || '--';
        const unit = result.unit || 'per quintal';
        const currency = result.currency || '₹';
        const market = result.market || result.mandi || 'Market';
        const change = result.change || result.price_change || 0;
        const changePercent = result.change_percent || result.price_change_percent || 0;
        const minPrice = result.min_price || '--';
        const maxPrice = result.max_price || '--';
        const date = result.date || result.last_updated || new Date().toLocaleDateString();

        const isPositive = changePercent >= 0;
        const changeColor = isPositive ? 'var(--success)' : 'var(--danger)';
        const changeIcon = isPositive ? 'fa-arrow-up' : 'fa-arrow-down';

        content.innerHTML = `
            <!-- Main Price Card -->
            <div class="card scale-in" style="margin-bottom:20px;">
                <div class="price-display">
                    <div class="price-header-row">
                        <div>
                            <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                                <span style="font-size:28px;">${getCropEmoji(commodity)}</span>
                                <h3 style="font-size:22px; font-weight:700;">${commodity}</h3>
                            </div>
                            <div style="display:flex; gap:8px; flex-wrap:wrap;">
                                <span class="badge badge-blue"><i class="fas fa-store"></i> ${market}</span>
                                <span class="badge badge-gray"><i class="fas fa-clock"></i> ${date}</span>
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div class="price-value" style="font-size:36px; font-weight:800; color:var(--primary-600);">
                                ${currency}${typeof price === 'number' ? price.toLocaleString() : price}
                            </div>
                            <div style="font-size:13px; color:var(--text-muted);">${unit}</div>
                            ${changePercent !== 0 ? `
                                <div style="margin-top:6px; font-size:14px; font-weight:600; color:${changeColor};">
                                    <i class="fas ${changeIcon}"></i> ${Math.abs(changePercent).toFixed(2)}%
                                    <span style="color:var(--text-muted); font-weight:400;">(${currency}${Math.abs(change)})</span>
                                </div>
                            ` : ''}
                        </div>
                    </div>

                    <!-- Price Range -->
                    <div style="margin-top:24px; padding:16px; background:var(--bg-input); border-radius:var(--radius-md);">
                        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                            <span style="font-size:13px; color:var(--text-muted);">Price Range</span>
                            <span style="font-size:13px; font-weight:500;">${currency}${minPrice} — ${currency}${maxPrice}</span>
                        </div>
                        <div style="height:6px; background:var(--bg-card); border-radius:3px; overflow:hidden; position:relative;">
                            ${typeof price === 'number' && typeof minPrice === 'number' && typeof maxPrice === 'number' ? `
                                <div style="height:100%; width:${((price - minPrice) / (maxPrice - minPrice) * 100).toFixed(0)}%; background:linear-gradient(90deg, var(--primary-400), var(--primary-600)); border-radius:3px; transition:width 1s ease;"></div>
                            ` : `
                                <div style="height:100%; width:50%; background:var(--primary-500); border-radius:3px;"></div>
                            `}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Stats & Chart row -->
            <div class="grid-2" style="grid-template-columns: 1fr 1fr; gap:16px;">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title"><i class="fas fa-chart-bar"></i> Price Statistics</div>
                    </div>
                    <div class="card-body stagger-children">
                        <div class="stat-row">
                            <span>Current Price</span>
                            <strong>${currency}${typeof price === 'number' ? price.toLocaleString() : price}</strong>
                        </div>
                        <div class="stat-row">
                            <span>Minimum Price</span>
                            <strong>${currency}${minPrice}</strong>
                        </div>
                        <div class="stat-row">
                            <span>Maximum Price</span>
                            <strong>${currency}${maxPrice}</strong>
                        </div>
                        <div class="stat-row">
                            <span>Market</span>
                            <strong>${market}</strong>
                        </div>
                        <div class="stat-row">
                            <span>Last Updated</span>
                            <strong>${date}</strong>
                        </div>
                    </div>
                </div>
                <div class="card" id="price-chart-card">
                    <div class="card-header">
                        <div class="card-title"><i class="fas fa-chart-line"></i> Price Trend</div>
                    </div>
                    <div class="card-body">
                        <canvas id="price-chart" height="220"></canvas>
                    </div>
                </div>
            </div>
        `;

        // Try to build trends chart automatically
        fetchTrendsQuiet(commodity);
        
        Toast.success(`Price loaded for ${commodity}`);
    } catch (error) {
        content.innerHTML = `
            <div class="card">
                <div class="empty-state">
                    <i class="fas fa-exclamation-triangle" style="color:var(--danger)"></i>
                    <h3>Price Unavailable</h3>
                    <p>${error.message}</p>
                </div>
            </div>
        `;
        Toast.error('Failed to get price: ' + error.message);
    }

    setLoading(btn, false);
}

async function fetchTrends() {
    const commodity = document.getElementById('price-commodity').value;
    if (!commodity) {
        Toast.warning('Please select a commodity first');
        return;
    }
    const btn = document.getElementById('price-trends-btn');
    setLoading(btn, true);
    await fetchTrendsQuiet(commodity);
    setLoading(btn, false);
}

async function fetchTrendsQuiet(commodity) {
    try {
        const result = await api.getPriceTrends(commodity);
        // API returns: { trend_data: { prices: [...] } }
        const td = result.trend_data || result;
        const trends = td.prices || td.trends || td.data || [];

        if (Array.isArray(trends) && trends.length > 0) {
            buildPriceChart(trends, commodity);
        }
    } catch (err) {
        // Silently handle — trends are optional
        const chartCanvas = document.getElementById('price-chart');
        if (chartCanvas) {
            chartCanvas.parentElement.innerHTML = `
                <div class="empty-state" style="padding:20px 0;">
                    <i class="fas fa-chart-line" style="color:var(--text-muted); font-size:20px;"></i>
                    <p style="font-size:13px;">Trend data unavailable</p>
                </div>
            `;
        }
    }
}

function buildPriceChart(trends, commodity) {
    const canvas = document.getElementById('price-chart');
    if (!canvas) return;

    if (priceChart) priceChart.destroy();

    const labels = trends.map(t => t.date || t.month || t.period || '');
    const prices = trends.map(t => t.price || t.avg_price || t.value || 0);

    const ctx = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 250);
    gradient.addColorStop(0, 'rgba(34, 197, 94, 0.25)');
    gradient.addColorStop(1, 'rgba(34, 197, 94, 0.01)');

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: `${commodity} Price (₹)`,
                data: prices,
                borderColor: '#22c55e',
                backgroundColor: gradient,
                tension: 0.4,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 7,
                pointBackgroundColor: '#22c55e',
                borderWidth: 2.5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15,23,42,0.9)',
                    cornerRadius: 8,
                    padding: 10,
                    callbacks: {
                        label: (ctx) => `₹${ctx.parsed.y.toLocaleString()}`
                    }
                }
            },
            scales: {
                y: {
                    ticks: { 
                        color: '#94a3b8',
                        callback: v => '₹' + v.toLocaleString()
                    },
                    grid: { color: 'rgba(148,163,184,0.1)' }
                },
                x: {
                    ticks: { color: '#94a3b8', maxRotation: 45 },
                    grid: { display: false }
                }
            }
        }
    });
}
