/* ============================================
   CHAT / AI ASSISTANT PAGE
   ============================================ */

function renderChat(container) {
    container.innerHTML = `
        <div class="page-header">
            <h2><i class="fas fa-comments"></i> AI Farming Assistant</h2>
            <p>Ask anything about crops, diseases, weather, prices, or farming techniques</p>
        </div>

        <div class="chat-container">
            <div class="chat-messages" id="chat-messages">
                <!-- Welcome message -->
                <div class="chat-message bot fade-in-up">
                    <div class="chat-avatar"><i class="fas fa-seedling"></i></div>
                    <div class="chat-bubble">
                        <strong>Welcome to FarmGenius! 🌾</strong><br><br>
                        I can help you with:<br>
                        • <strong>Crop recommendations</strong> — "What crop should I grow in clay soil?"<br>
                        • <strong>Disease info</strong> — "What are symptoms of tomato leaf blight?"<br>
                        • <strong>Weather updates</strong> — "What's the weather in Mumbai?"<br>
                        • <strong>Market prices</strong> — "What's the price of wheat?"<br>
                        • <strong>Farming tips</strong> — "How do I improve soil fertility?"<br><br>
                        Just type your question below!
                    </div>
                </div>
            </div>
            
            <div class="chat-input-area">
                <input type="text" id="chat-input" placeholder="Type your farming question..." 
                    autocomplete="off" onkeydown="if(event.key==='Enter') sendChatMessage()">
                <button class="chat-send-btn" id="chat-send-btn" onclick="sendChatMessage()" title="Send">
                    <i class="fas fa-paper-plane"></i>
                </button>
            </div>
        </div>
    `;

    // Focus input
    setTimeout(() => {
        const input = document.getElementById('chat-input');
        if (input) input.focus();
    }, 400);
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');
    const messages = document.getElementById('chat-messages');
    
    if (!input || !messages) return;

    const query = input.value.trim();
    if (!query) return;

    const selectedLang = localStorage.getItem('fg-language') || 'en';

    // Add user message (show original text)
    messages.innerHTML += `
        <div class="chat-message user">
            <div class="chat-avatar"><i class="fas fa-user"></i></div>
            <div class="chat-bubble">${escapeHtml(query)}</div>
        </div>
    `;

    input.value = '';
    input.disabled = true;
    sendBtn.disabled = true;

    // Add typing indicator
    const typingId = 'typing-' + Date.now();
    messages.innerHTML += `
        <div class="chat-message bot" id="${typingId}">
            <div class="chat-avatar"><i class="fas fa-seedling"></i></div>
            <div class="chat-bubble">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
    `;
    messages.scrollTop = messages.scrollHeight;

    try {
        // Step 1: If non-English, translate the query to English before sending
        let queryToSend = query;
        if (selectedLang !== 'en') {
            try {
                const translateResult = await api.translate(query, 'en', selectedLang);
                if (translateResult.success && translateResult.translated_text) {
                    queryToSend = translateResult.translated_text;
                }
            } catch (translateErr) {
                console.warn('Translation to English failed, sending original:', translateErr);
            }
        }

        // Step 2: Send the English query to the backend
        const result = await api.sendQuery(queryToSend);
        
        // Step 3: If non-English, translate the English response back to user's language
        let responseText = result.response_text || 'No response received.';
        if (selectedLang !== 'en' && responseText) {
            try {
                const translateBack = await api.translate(responseText, selectedLang, 'en');
                if (translateBack.success && translateBack.translated_text) {
                    responseText = translateBack.translated_text;
                }
            } catch (translateErr) {
                console.warn('Translation back to user language failed:', translateErr);
            }
        }

        // Remove typing
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();

        // Add bot response
        let responseHtml = escapeHtml(responseText);
        
        // Format line breaks
        responseHtml = responseHtml.replace(/\n/g, '<br>');

        // Add source badge
        let sourceBadge = '';
        if (result.source) {
            const sourceLabels = {
                'faq_model': '📚 FAQ',
                'crop_model': '🌾 Crop AI',
                'weather_model': '🌤️ Weather',
                'price_model': '📊 Prices',
                'disease_model': '🔬 Disease',
                'deepseek': '🤖 DeepSeek AI'
            };
            const label = sourceLabels[result.source] || result.source;
            sourceBadge = `<span class="source-badge">${label} • ${Math.round((result.confidence || 0) * 100)}% confidence</span>`;
        }

        // Add language badge if translated
        let langBadge = '';
        if (selectedLang !== 'en') {
            const langNames = { hi: 'Hindi', bn: 'Bengali', es: 'Spanish', fr: 'French', de: 'German', pt: 'Portuguese', ar: 'Arabic', ru: 'Russian', it: 'Italian' };
            langBadge = `<span class="lang-badge">🌐 Translated to ${langNames[selectedLang] || selectedLang}</span>`;
        }

        messages.innerHTML += `
            <div class="chat-message bot">
                <div class="chat-avatar"><i class="fas fa-seedling"></i></div>
                <div class="chat-bubble">
                    ${responseHtml}
                    ${sourceBadge}
                    ${langBadge}
                </div>
            </div>
        `;
    } catch (error) {
        // Remove typing
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();

        messages.innerHTML += `
            <div class="chat-message bot">
                <div class="chat-avatar"><i class="fas fa-seedling"></i></div>
                <div class="chat-bubble" style="border-left: 3px solid var(--danger);">
                    ⚠️ Sorry, I couldn't process your request. ${escapeHtml(error.message)}
                </div>
            </div>
        `;
        Toast.error('Failed to get response: ' + error.message);
    }

    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
    messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
