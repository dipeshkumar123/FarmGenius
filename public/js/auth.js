/* ============================================
   AUTH MANAGER — Token storage, user state,
   and UI updates for authentication
   ============================================ */

const authManager = (() => {
    const TOKEN_KEY = 'fg-auth-token';
    const USER_KEY  = 'fg-auth-user';

    // ── State ──
    let _token = localStorage.getItem(TOKEN_KEY);
    let _user  = JSON.parse(localStorage.getItem(USER_KEY) || 'null');

    function isLoggedIn() {
        return !!_token && !!_user;
    }

    function getToken() {
        return _token;
    }

    function getUser() {
        return _user;
    }

    function getUserId() {
        return _user ? _user.user_id : 'guest';
    }

    function getDisplayName() {
        if (!_user) return 'Guest';
        return _user.display_name || _user.email?.split('@')[0] || _user.user_id;
    }

    // ── Login callback ──
    function onLogin(token, user) {
        _token = token;
        _user = user;
        localStorage.setItem(TOKEN_KEY, token);
        localStorage.setItem(USER_KEY, JSON.stringify(user));
        updateUI();
        // Navigate to dashboard
        router.navigateTo('dashboard');
        showToast('Welcome, ' + getDisplayName() + '!', 'success');
    }

    // ── Guest mode ──
    function onGuestMode() {
        _token = null;
        _user = null;
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
        updateUI();
        router.navigateTo('dashboard');
    }

    // ── Logout ──
    function logout() {
        _token = null;
        _user = null;
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
        updateUI();
        router.navigateTo('auth');
        showToast('Signed out', 'info');
    }

    // ── Refresh user profile from server ──
    async function refreshProfile() {
        if (!_token) return;
        try {
            const user = await api.getMe();
            _user = user;
            localStorage.setItem(USER_KEY, JSON.stringify(user));
            updateUI();
        } catch (err) {
            // Token expired / invalid → force logout
            if (err.message?.includes('401') || err.message?.includes('expired') || err.message?.includes('Invalid')) {
                logout();
            }
        }
    }

    // ── Update DOM to reflect auth state ──
    function updateUI() {
        // User avatar / name in topbar
        const avatar = document.querySelector('.user-avatar');
        if (avatar) {
            if (isLoggedIn()) {
                const initials = getDisplayName().charAt(0).toUpperCase();
                avatar.innerHTML = `<span class="avatar-initials">${initials}</span>`;
                avatar.title = getDisplayName();
                avatar.style.cursor = 'pointer';
                avatar.onclick = showUserMenu;
            } else {
                avatar.innerHTML = '<i class="fas fa-user"></i>';
                avatar.title = 'Guest — click to sign in';
                avatar.style.cursor = 'pointer';
                avatar.onclick = () => router.navigateTo('auth');
            }
        }

        // Update sidebar auth link
        const authLink = document.getElementById('nav-auth-link');
        if (authLink) {
            if (isLoggedIn()) {
                authLink.href = '#';
                authLink.dataset.page = '';
                authLink.innerHTML = '<i class="fas fa-sign-out-alt"></i><span>Sign Out</span>';
                authLink.onclick = (e) => { e.preventDefault(); logout(); };
            } else {
                authLink.href = '#auth';
                authLink.dataset.page = 'auth';
                authLink.innerHTML = '<i class="fas fa-sign-in-alt"></i><span>Sign In</span>';
                authLink.onclick = null;
            }
        }
    }

    function showUserMenu() {
        // Remove existing menu
        const existing = document.getElementById('user-dropdown');
        if (existing) { existing.remove(); return; }

        const menu = document.createElement('div');
        menu.id = 'user-dropdown';
        menu.className = 'user-dropdown';
        menu.innerHTML = `
            <div class="dropdown-header">
                <strong>${getDisplayName()}</strong>
                <span class="dropdown-email">${_user?.email || ''}</span>
            </div>
            <div class="dropdown-divider"></div>
            <button class="dropdown-item" onclick="router.navigateTo('history'); document.getElementById('user-dropdown')?.remove();">
                <i class="fas fa-clock-rotate-left"></i> My History
            </button>
            <button class="dropdown-item" onclick="router.navigateTo('profile'); document.getElementById('user-dropdown')?.remove();">
                <i class="fas fa-cog"></i> Settings
            </button>
            <div class="dropdown-divider"></div>
            <button class="dropdown-item dropdown-danger" onclick="authManager.logout()">
                <i class="fas fa-sign-out-alt"></i> Sign Out
            </button>
        `;
        document.body.appendChild(menu);

        // Position near avatar
        const avatar = document.querySelector('.user-avatar');
        if (avatar) {
            const rect = avatar.getBoundingClientRect();
            menu.style.top = (rect.bottom + 8) + 'px';
            menu.style.right = (window.innerWidth - rect.right) + 'px';
        }

        // Close on outside click
        setTimeout(() => {
            document.addEventListener('click', function handler(e) {
                if (!menu.contains(e.target) && !e.target.closest('.user-avatar')) {
                    menu.remove();
                    document.removeEventListener('click', handler);
                }
            });
        }, 50);
    }

    return {
        isLoggedIn,
        getToken,
        getUser,
        getUserId,
        getDisplayName,
        onLogin,
        onGuestMode,
        logout,
        refreshProfile,
        updateUI,
    };
})();
