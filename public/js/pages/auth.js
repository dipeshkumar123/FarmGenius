/* ============================================
   AUTH PAGE — Login / Register
   ============================================ */

function renderAuth(container) {
    container.innerHTML = `
        <div class="auth-page">
            <div class="auth-card card">
                <div class="auth-header">
                    <div class="auth-logo">
                        <i class="fas fa-seedling"></i>
                    </div>
                    <h2>Welcome to FarmGenius</h2>
                    <p class="auth-subtitle">Sign in to save your data across sessions</p>
                </div>

                <!-- Tab switcher -->
                <div class="auth-tabs">
                    <button class="auth-tab active" data-tab="login" onclick="switchAuthTab('login')">Sign In</button>
                    <button class="auth-tab" data-tab="register" onclick="switchAuthTab('register')">Create Account</button>
                </div>

                <!-- Login Form -->
                <form id="login-form" class="auth-form" onsubmit="handleLogin(event)">
                    <div class="form-group">
                        <label for="login-email"><i class="fas fa-envelope"></i> Email</label>
                        <input type="email" id="login-email" placeholder="farmer@example.com" required autocomplete="email">
                    </div>
                    <div class="form-group">
                        <label for="login-password"><i class="fas fa-lock"></i> Password</label>
                        <div class="password-wrapper">
                            <input type="password" id="login-password" placeholder="Your password" required autocomplete="current-password">
                            <button type="button" class="password-toggle" onclick="togglePassword('login-password', this)">
                                <i class="fas fa-eye"></i>
                            </button>
                        </div>
                    </div>
                    <div id="login-error" class="auth-error" style="display:none;"></div>
                    <button type="submit" class="btn btn-primary btn-full" id="login-btn">
                        <i class="fas fa-sign-in-alt"></i> Sign In
                    </button>
                </form>

                <!-- Register Form -->
                <form id="register-form" class="auth-form" style="display:none;" onsubmit="handleRegister(event)">
                    <div class="form-group">
                        <label for="reg-name"><i class="fas fa-user"></i> Display Name</label>
                        <input type="text" id="reg-name" placeholder="Your name" autocomplete="name">
                    </div>
                    <div class="form-group">
                        <label for="reg-email"><i class="fas fa-envelope"></i> Email</label>
                        <input type="email" id="reg-email" placeholder="farmer@example.com" required autocomplete="email">
                    </div>
                    <div class="form-group">
                        <label for="reg-password"><i class="fas fa-lock"></i> Password</label>
                        <div class="password-wrapper">
                            <input type="password" id="reg-password" placeholder="Min 6 characters" required minlength="6" autocomplete="new-password">
                            <button type="button" class="password-toggle" onclick="togglePassword('reg-password', this)">
                                <i class="fas fa-eye"></i>
                            </button>
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="reg-confirm"><i class="fas fa-lock"></i> Confirm Password</label>
                        <div class="password-wrapper">
                            <input type="password" id="reg-confirm" placeholder="Repeat password" required minlength="6" autocomplete="new-password">
                            <button type="button" class="password-toggle" onclick="togglePassword('reg-confirm', this)">
                                <i class="fas fa-eye"></i>
                            </button>
                        </div>
                    </div>
                    <div id="register-error" class="auth-error" style="display:none;"></div>
                    <button type="submit" class="btn btn-primary btn-full" id="register-btn">
                        <i class="fas fa-user-plus"></i> Create Account
                    </button>
                </form>

                <div class="auth-footer">
                    <button class="btn btn-outline btn-full" onclick="continueAsGuest()">
                        <i class="fas fa-user-secret"></i> Continue as Guest
                    </button>
                    <p class="auth-hint">Guest data is not saved between sessions</p>
                </div>
            </div>
        </div>
    `;
}

function switchAuthTab(tab) {
    document.querySelectorAll('.auth-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
    document.getElementById('login-form').style.display = tab === 'login' ? '' : 'none';
    document.getElementById('register-form').style.display = tab === 'register' ? '' : 'none';
    // Clear errors
    document.getElementById('login-error').style.display = 'none';
    document.getElementById('register-error').style.display = 'none';
}

function togglePassword(inputId, btn) {
    const inp = document.getElementById(inputId);
    const icon = btn.querySelector('i');
    if (inp.type === 'password') {
        inp.type = 'text';
        icon.className = 'fas fa-eye-slash';
    } else {
        inp.type = 'password';
        icon.className = 'fas fa-eye';
    }
}

function showAuthError(formPrefix, msg) {
    const el = document.getElementById(formPrefix + '-error');
    el.textContent = msg;
    el.style.display = 'block';
}

async function handleLogin(e) {
    e.preventDefault();
    const btn = document.getElementById('login-btn');
    const email = document.getElementById('login-email').value.trim();
    const password = document.getElementById('login-password').value;

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Signing in...';

    try {
        const data = await api.login(email, password);
        authManager.onLogin(data.token, data.user);
    } catch (err) {
        showAuthError('login', err.message || 'Login failed');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Sign In';
    }
}

async function handleRegister(e) {
    e.preventDefault();
    const btn = document.getElementById('register-btn');
    const name = document.getElementById('reg-name').value.trim();
    const email = document.getElementById('reg-email').value.trim();
    const password = document.getElementById('reg-password').value;
    const confirm = document.getElementById('reg-confirm').value;

    if (password !== confirm) {
        showAuthError('register', 'Passwords do not match');
        return;
    }
    if (password.length < 6) {
        showAuthError('register', 'Password must be at least 6 characters');
        return;
    }

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating account...';

    try {
        const data = await api.register(email, password, name);
        authManager.onLogin(data.token, data.user);
    } catch (err) {
        showAuthError('register', err.message || 'Registration failed');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-user-plus"></i> Create Account';
    }
}

function continueAsGuest() {
    authManager.onGuestMode();
}
