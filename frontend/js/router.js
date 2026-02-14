/* ============================================
   ROUTER - Simple hash-based SPA routing
   ============================================ */

class Router {
    constructor() {
        this.routes = {};
        this.currentPage = null;
        this.container = null;
    }

    init(containerId) {
        this.container = document.getElementById(containerId);
        window.addEventListener('hashchange', () => this.handleRoute());
        
        // Handle nav link clicks
        document.querySelectorAll('.nav-link[data-page]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = link.dataset.page;
                window.location.hash = page;
                // Close mobile sidebar
                document.getElementById('sidebar').classList.remove('mobile-open');
                document.getElementById('mobile-overlay').classList.remove('active');
            });
        });

        // Render the initial page based on current hash
        this.handleRoute();
    }

    register(name, renderFn) {
        this.routes[name] = renderFn;
    }

    handleRoute() {
        const hash = window.location.hash.slice(1) || 'dashboard';
        this.navigateTo(hash);
    }

    navigateTo(page) {
        if (!this.routes[page]) {
            page = 'dashboard';
        }

        // Update active nav
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.toggle('active', link.dataset.page === page);
        });

        // Render page
        this.currentPage = page;
        this.container.innerHTML = '';
        
        const pageWrapper = document.createElement('div');
        pageWrapper.className = 'page-enter';
        this.container.appendChild(pageWrapper);

        this.routes[page](pageWrapper);
    }
}

const router = new Router();
