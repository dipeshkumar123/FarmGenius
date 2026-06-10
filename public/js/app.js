/* ============================================
   FARMGENIUS — Main App Initialization
   ============================================ */

(function () {
    'use strict';

    // ── Loading Screen ──
    function dismissLoadingScreen() {
        const screen = document.getElementById('loading-screen');
        const app = document.getElementById('app');
        if (screen) {
            screen.style.opacity = '0';
            screen.style.pointerEvents = 'none';
            setTimeout(() => screen.remove(), 500);
        }
        if (app) {
            app.style.opacity = '1';
            app.style.transition = 'opacity 0.4s ease';
        }
    }

    // ── Register Routes ──
    function setupRoutes() {
        router.register('auth', renderAuth);
        router.register('dashboard', renderDashboard);
        router.register('chat', renderChat);
        router.register('crops', renderCrops);
        router.register('diseases', renderDiseases);
        router.register('weather', renderWeather);
        router.register('prices', renderPrices);
        router.register('history', renderHistory);
    }

    // ── Sidebar Toggle (Desktop) ──
    function setupSidebar() {
        const sidebar = document.getElementById('sidebar');
        const toggleBtn = document.getElementById('sidebar-toggle');
        
        // Restore saved state
        const collapsed = localStorage.getItem('fg-sidebar-collapsed') === 'true';
        if (collapsed) sidebar.classList.add('collapsed');

        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                sidebar.classList.toggle('collapsed');
                localStorage.setItem('fg-sidebar-collapsed', sidebar.classList.contains('collapsed'));
            });
        }
    }

    // ── Mobile Menu ──
    function setupMobileMenu() {
        const sidebar = document.getElementById('sidebar');
        const menuBtn = document.getElementById('mobile-menu-btn');
        const overlay = document.getElementById('mobile-overlay');

        function openMenu() {
            sidebar.classList.add('mobile-open');
            overlay.classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function closeMenu() {
            sidebar.classList.remove('mobile-open');
            overlay.classList.remove('active');
            document.body.style.overflow = '';
        }

        if (menuBtn) menuBtn.addEventListener('click', openMenu);
        if (overlay) overlay.addEventListener('click', closeMenu);

        // Close menu on nav link click (mobile)
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                if (window.innerWidth <= 768) closeMenu();
            });
        });
    }

    // ── Language Selector ──
    function setupLanguage() {
        const select = document.getElementById('lang-select');
        if (!select) return;

        // Restore saved language
        const saved = localStorage.getItem('fg-language') || 'en';
        select.value = saved;

        select.addEventListener('change', () => {
            localStorage.setItem('fg-language', select.value);
        });
    }

    // ── Theme Toggle ──
    function setupTheme() {
        const toggle = document.getElementById('theme-toggle');
        const icon = toggle?.querySelector('i');
        
        // Restore saved theme
        const saved = localStorage.getItem('fg-theme') || 'light';
        document.documentElement.setAttribute('data-theme', saved);
        if (icon) icon.className = saved === 'dark' ? 'fas fa-sun' : 'fas fa-moon';

        if (toggle) {
            toggle.addEventListener('click', () => {
                const current = document.documentElement.getAttribute('data-theme');
                const next = current === 'dark' ? 'light' : 'dark';
                document.documentElement.setAttribute('data-theme', next);
                localStorage.setItem('fg-theme', next);
                if (icon) icon.className = next === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
            });
        }
    }

    // ── Global Search ──
    function setupGlobalSearch() {
        const searchInput = document.getElementById('global-search');
        if (!searchInput) return;

        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const query = searchInput.value.trim();
                if (query) {
                    router.navigateTo('chat');
                    // Wait for chat page to render, then fill the input
                    setTimeout(() => {
                        const chatInput = document.getElementById('chat-input');
                        if (chatInput) {
                            chatInput.value = query;
                            chatInput.focus();
                            // Auto-send the query
                            if (typeof sendChatMessage === 'function') {
                                sendChatMessage();
                            }
                        }
                    }, 200);
                    searchInput.value = '';
                }
            }
        });
    }

    // ── API Status Check ──
    async function checkApiStatus() {
        const statusEl = document.getElementById('api-status');
        if (!statusEl) return;

        const dot = statusEl.querySelector('.status-dot');
        const text = statusEl.querySelector('.status-text');

        try {
            await api.checkHealth();
            dot.style.background = 'var(--success)';
            dot.style.boxShadow = '0 0 6px var(--success)';
            text.textContent = 'API Connected';
            statusEl.title = 'Connected to FarmGenius API at localhost:8000';
        } catch (err) {
            dot.style.background = 'var(--danger)';
            dot.style.boxShadow = '0 0 6px var(--danger)';
            text.textContent = 'API Offline';
            statusEl.title = 'Cannot reach the API server. Is it running?';
        }
    }

    // ── Keyboard Shortcuts ──
    function setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+K → Focus search
            if (e.ctrlKey && e.key === 'k') {
                e.preventDefault();
                document.getElementById('global-search')?.focus();
            }
            // Escape → Close mobile menu
            if (e.key === 'Escape') {
                const sidebar = document.getElementById('sidebar');
                const overlay = document.getElementById('mobile-overlay');
                if (sidebar?.classList.contains('mobile-open')) {
                    sidebar.classList.remove('mobile-open');
                    overlay?.classList.remove('active');
                    document.body.style.overflow = '';
                }
            }
        });
    }

    // ── Initialize App ──
    function init() {
        setupRoutes();
        setupSidebar();
        setupMobileMenu();
        setupLanguage();
        setupTheme();
        setupGlobalSearch();
        setupKeyboardShortcuts();

        // Initialize auth UI state
        authManager.updateUI();

        // Initialize router (reads current hash and renders page)
        router.init('page-container');

        // Check API status
        checkApiStatus();
        // Re-check every 30 seconds
        setInterval(checkApiStatus, 30000);

        // Refresh user profile if logged in
        if (authManager.isLoggedIn()) {
            authManager.refreshProfile();
        }

        // Dismiss loading screen after a short delay
        setTimeout(dismissLoadingScreen, 1800);
    }

    // ── Start when DOM is ready ──
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
