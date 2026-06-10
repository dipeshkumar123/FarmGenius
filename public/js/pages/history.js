/* ============================================
   CONVERSATION HISTORY PAGE
   ============================================ */

function renderHistory(container) {
    const name = authManager.getDisplayName();
    const loggedIn = authManager.isLoggedIn();

    container.innerHTML = `
        <div class="page-header">
            <h2><i class="fas fa-clock-rotate-left"></i> Conversation History</h2>
            <p>View your past interactions with the FarmGenius AI assistant</p>
        </div>

        <div class="card" style="margin-bottom:16px;">
            <div class="card-body" style="padding:12px 20px;">
                <div style="display:flex; gap:12px; align-items:center; justify-content:space-between; flex-wrap:wrap;">
                    <div style="display:flex; gap:10px; align-items:center;">
                        <i class="fas fa-user-circle" style="font-size:20px; color:var(--primary-500);"></i>
                        <span style="font-weight:500;">${loggedIn ? name : 'Guest'}</span>
                        ${!loggedIn ? '<span class="badge badge-warning" style="font-size:11px;">Sign in to save history</span>' : ''}
                    </div>
                    <div style="display:flex; gap:8px;">
                        ${loggedIn ? '<button class="btn btn-danger btn-sm" onclick="clearMyHistory()"><i class="fas fa-trash"></i> Clear</button>' : ''}
                        <button class="btn btn-outline btn-sm" onclick="loadHistory()">
                            <i class="fas fa-refresh"></i> Refresh
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <div id="history-content">
            <div class="card">
                <div class="card-body" style="padding:40px;">
                    ${createSkeleton(5, 'large')}
                </div>
            </div>
        </div>
    `;

    if (!loggedIn) {
        document.getElementById('history-content').innerHTML = `
            <div class="card">
                <div class="empty-state">
                    <i class="fas fa-lock" style="color:var(--text-muted)"></i>
                    <h3>Sign In to View History</h3>
                    <p>Create an account to keep your conversation history across sessions</p>
                    <button class="btn btn-primary" onclick="router.navigateTo('auth')" style="margin-top:12px;">
                        <i class="fas fa-sign-in-alt"></i> Sign In
                    </button>
                </div>
            </div>
        `;
        return;
    }

    loadHistory();
}

async function loadHistory() {
    const content = document.getElementById('history-content');

    try {
        const result = await api.getHistory('guest');
        const history = result.entries || result.history || result.conversations || result || [];

        if (!Array.isArray(history) || history.length === 0) {
            content.innerHTML = `
                <div class="card">
                    <div class="empty-state">
                        <i class="fas fa-inbox" style="color:var(--text-muted)"></i>
                        <h3>No History Yet</h3>
                        <p>Start chatting with the AI Assistant and your conversations will appear here</p>
                        <button class="btn btn-primary" onclick="router.navigateTo('chat')" style="margin-top:12px;">
                            <i class="fas fa-comments"></i> Start Chat
                        </button>
                    </div>
                </div>
            `;
            return;
        }

        // Reverse to show newest first
        const sorted = [...history].reverse();

        content.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <div class="card-title"><i class="fas fa-list"></i> ${sorted.length} Conversation${sorted.length !== 1 ? 's' : ''}</div>
                </div>
                <div class="card-body stagger-children" style="max-height:600px; overflow-y:auto;">
                    ${sorted.map((item, i) => renderHistoryItem(item, i)).join('')}
                </div>
            </div>
        `;
    } catch (error) {
        content.innerHTML = `
            <div class="card">
                <div class="empty-state">
                    <i class="fas fa-exclamation-triangle" style="color:var(--danger)"></i>
                    <h3>Failed to Load History</h3>
                    <p>${error.message}</p>
                    <button class="btn btn-outline" onclick="loadHistory()" style="margin-top:12px;">
                        <i class="fas fa-refresh"></i> Retry
                    </button>
                </div>
            </div>
        `;
    }
}

function renderHistoryItem(item, index) {
    const query = item.query || item.message || item.input || 'Unknown query';
    const response = item.response || item.answer || item.output || '';
    const timestamp = item.timestamp || item.created_at || item.date || '';
    const intent = item.intent || item.category || item.type || '';
    const confidence = item.confidence || 0;

    const intentIcon = getIntentIcon(intent);
    const intentColor = getIntentColor(intent);
    const timeStr = timestamp ? timeAgo(timestamp) : '';

    return `
        <div class="history-item" onclick="toggleHistoryItem(${index})">
            <div class="history-item-header">
                <div class="history-item-left">
                    <div class="history-intent-icon" style="background:${intentColor}15; color:${intentColor};">
                        <i class="fas ${intentIcon}"></i>
                    </div>
                    <div>
                        <div class="history-query">${escapeHtml(query)}</div>
                        <div class="history-meta">
                            ${intent ? `<span class="badge badge-gray" style="font-size:10px;">${intent}</span>` : ''}
                            ${confidence ? `<span style="font-size:11px; color:var(--text-muted);">${(confidence * 100).toFixed(0)}% confidence</span>` : ''}
                            ${timeStr ? `<span style="font-size:11px; color:var(--text-muted);"><i class="fas fa-clock"></i> ${timeStr}</span>` : ''}
                        </div>
                    </div>
                </div>
                <i class="fas fa-chevron-down history-chevron" style="font-size:12px; color:var(--text-muted); transition:transform 0.2s;"></i>
            </div>
            <div class="history-item-body" id="history-body-${index}">
                <div style="padding:12px 16px; background:var(--bg-input); border-radius:var(--radius-md); margin-top:12px;">
                    <div style="font-size:11px; color:var(--text-muted); margin-bottom:6px; font-weight:500;">
                        <i class="fas fa-robot"></i> AI Response
                    </div>
                    <div style="font-size:14px; line-height:1.6;">${escapeHtml(typeof response === 'string' ? response : JSON.stringify(response))}</div>
                </div>
            </div>
        </div>
    `;
}

function toggleHistoryItem(index) {
    const item = document.querySelector(`.history-item:nth-child(${index + 1})`);
    if (item) {
        item.classList.toggle('expanded');
        const chevron = item.querySelector('.history-chevron');
        if (chevron) {
            chevron.style.transform = item.classList.contains('expanded') ? 'rotate(180deg)' : '';
        }
    }
}

function getIntentIcon(intent) {
    const icons = {
        'faq': 'fa-question-circle',
        'crop': 'fa-seedling',
        'crop_recommendation': 'fa-seedling',
        'disease': 'fa-virus',
        'disease_detection': 'fa-virus',
        'weather': 'fa-cloud-sun',
        'price': 'fa-indian-rupee-sign',
        'market_price': 'fa-indian-rupee-sign',
        'greeting': 'fa-hand-wave',
        'general': 'fa-comment'
    };
    return icons[intent?.toLowerCase()] || 'fa-comment-dots';
}

function getIntentColor(intent) {
    const colors = {
        'faq': '#8b5cf6',
        'crop': '#22c55e',
        'crop_recommendation': '#22c55e',
        'disease': '#ef4444',
        'disease_detection': '#ef4444',
        'weather': '#0ea5e9',
        'price': '#f59e0b',
        'market_price': '#f59e0b',
        'greeting': '#ec4899',
        'general': '#64748b'
    };
    return colors[intent?.toLowerCase()] || '#64748b';
}

async function clearMyHistory() {
    if (!confirm('Clear all your conversation history? This cannot be undone.')) return;
    try {
        const result = await api.clearHistory();
        showToast(`Cleared ${result.cleared} conversation(s)`, 'success');
        loadHistory();
    } catch (err) {
        showToast('Failed to clear history: ' + err.message, 'error');
    }
}
