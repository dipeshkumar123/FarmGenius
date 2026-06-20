// src/api/client.ts
import axios from 'axios';

// Use /api in production to leverage Vercel's rewrite and avoid CORS
const BASE_URL = import.meta.env.PROD
  ? '/api'
  : import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor — attach auth token from localStorage
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('fg_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor — handle errors globally
// CRITICAL FIX: Do NOT use window.location.href (causes hard reload).
// Instead, dispatch a custom event that App.tsx listens to and handles
// via React Router's navigate() — preserving SPA state.
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Only trigger logout/redirect for auth-critical routes
      const url = error.config?.url || '';
      const isAuthCritical = url.includes('/chat') || url.includes('/disease') || url.includes('/auth');
      if (isAuthCritical) {
        localStorage.removeItem('fg_token');
        // Dispatch custom event so AuthGuard can respond without hard reload
        window.dispatchEvent(new CustomEvent('fg:unauthorized'));
      }
    }
    return Promise.reject(error);
  }
);

export default apiClient;
