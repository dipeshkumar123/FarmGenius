// src/App.tsx
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { AnimatePresence } from 'framer-motion';
import { useEffect } from 'react';
import { useAppStore } from './store/appStore';

import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import ChatPage from './pages/ChatPage';
import ScanPage from './pages/ScanPage';
import MarketPage from './pages/MarketPage';
import WeatherPage from './pages/WeatherPage';
import CropsPage from './pages/CropsPage';
import SchemesPage from './pages/SchemesPage';
import ProfilePage from './pages/ProfilePage';
import AppShell from './components/layout/AppShell';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,  // 5 minutes
      retry: 1,
    },
  },
});

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAppStore((s) => s.isAuthenticated);
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

// Inner component that has access to React Router hooks
function AppRoutes() {
  const navigate = useNavigate();
  const setOffline = useAppStore((s) => s.setOffline);
  const logout = useAppStore((s) => s.logout);

  useEffect(() => {
    const handleOnline = () => setOffline(false);
    const handleOffline = () => setOffline(true);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [setOffline]);

  // CRITICAL FIX: Listen for 401 unauthorized events from apiClient
  // and do a SOFT React Router navigation instead of hard page reload.
  useEffect(() => {
    const handleUnauthorized = () => {
      logout();
      navigate('/login', { replace: true });
    };
    window.addEventListener('fg:unauthorized', handleUnauthorized);
    return () => window.removeEventListener('fg:unauthorized', handleUnauthorized);
  }, [navigate, logout]);

  return (
    <>
      <AnimatePresence mode="wait">
        <Routes>
            {/* Public */}
            <Route path="/" element={<LandingPage />} />
            <Route path="/login" element={<LoginPage />} />

            {/* Protected — inside app shell */}
            <Route
              element={
                <ProtectedRoute>
                  <AppShell />
                </ProtectedRoute>
              }
            >
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/scan" element={<ScanPage />} />
              <Route path="/market" element={<MarketPage />} />
              <Route path="/weather" element={<WeatherPage />} />
              <Route path="/crops" element={<CropsPage />} />
              <Route path="/schemes" element={<SchemesPage />} />
              <Route path="/profile" element={<ProfilePage />} />
            </Route>

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AnimatePresence>
        <Toaster
          position="top-center"
          toastOptions={{
            style: {
              fontFamily: 'Poppins, sans-serif',
              fontSize: '14px',
              borderRadius: '100px',
              padding: '12px 20px',
            },
            success: { iconTheme: { primary: '#2E7D32', secondary: '#fff' } },
          }}
        />
    </>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
