// src/main.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import posthog from 'posthog-js';
import { PostHogProvider } from 'posthog-js/react';
import App from './App';
import './i18n';
import './index.css';

// Initialize PostHog (using dummy fallback if env var is missing)
posthog.init(
  import.meta.env.VITE_POSTHOG_KEY || 'phc_dummy_key_for_telemetry',
  { api_host: import.meta.env.VITE_POSTHOG_HOST || 'https://app.posthog.com' }
);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <PostHogProvider client={posthog}>
      <App />
    </PostHogProvider>
  </React.StrictMode>
);
