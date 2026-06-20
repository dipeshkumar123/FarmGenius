// src/components/layout/AppShell.tsx
import { useState, useRef, useEffect } from 'react';
import { Outlet, NavLink, useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  House,
  ChatCircleText,
  Scan,
  ChartLineUp,
  UserCircle,
  Leaf,
  WifiSlash,
  Bell,
  CaretDown,
  SignOut,
  Gear,
  Globe,
  Sun,
  CloudRain,
  Flower,
  FileText,
  Check,
} from 'phosphor-react';
import { useAppStore } from '../../store/appStore';

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────
interface NavItem {
  to: string;
  icon: React.ElementType;
  label: string;
  isFab?: boolean;
}

interface DesktopNavItem {
  to: string;
  icon: React.ElementType;
  label: string;
}

interface LanguageOption {
  code: string;
  label: string;
  nativeLabel: string;
}

// ─────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────
const mobileNavItems: NavItem[] = [
  { to: '/dashboard', icon: House, label: 'Home' },
  { to: '/market', icon: ChartLineUp, label: 'Market' },
  { to: '/scan', icon: Scan, label: 'Scan', isFab: true },
  { to: '/chat', icon: ChatCircleText, label: 'AI Chat' },
  { to: '/profile', icon: UserCircle, label: 'Profile' },
];

const desktopNavItems: DesktopNavItem[] = [
  { to: '/dashboard', icon: House, label: 'Dashboard' },
  { to: '/chat', icon: ChatCircleText, label: 'AI Chat' },
  { to: '/market', icon: ChartLineUp, label: 'Market' },
  { to: '/weather', icon: Sun, label: 'Weather' },
  { to: '/crops', icon: Flower, label: 'Crops' },
  { to: '/schemes', icon: FileText, label: 'Schemes' },
];

const languageOptions: LanguageOption[] = [
  { code: 'en', label: 'English', nativeLabel: 'EN' },
  { code: 'hi', label: 'Hindi', nativeLabel: 'हि' },
  { code: 'kn', label: 'Kannada', nativeLabel: 'ಕ' },
  { code: 'te', label: 'Telugu', nativeLabel: 'తె' },
  { code: 'ta', label: 'Tamil', nativeLabel: 'த' },
  { code: 'mr', label: 'Marathi', nativeLabel: 'म' },
];

// ─────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────

/** Offline warning banner that slides down from the top */
function OfflineBanner() {
  return (
    <motion.div
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: -50, opacity: 0 }}
      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
      className="bg-amber-50 border-b border-amber-200 px-4 py-2 flex items-center gap-2 text-sm text-amber-800 z-50 relative"
    >
      <WifiSlash size={16} weight="bold" className="shrink-0 text-amber-600" />
      <span className="font-noto">
        Offline mode — showing cached data
      </span>
      <span className="ml-auto text-xs text-amber-600 font-poppins font-semibold">
        No Internet
      </span>
    </motion.div>
  );
}

/** Language selector dropdown */
function LanguageSelector() {
  const { language, setLanguage } = useAppStore();
  const [isOpen, setIsOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const current = languageOptions.find((l) => l.code === language) ?? languageOptions[0];

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-surface-variant border border-farm-divider
                   text-sm font-poppins font-semibold text-primary hover:bg-primary hover:text-white
                   transition-all duration-200 active:scale-95 min-h-[36px]"
        aria-label="Select language"
        aria-expanded={isOpen}
      >
        <Globe size={14} weight="bold" />
        <span>{current.nativeLabel}</span>
        <motion.span animate={{ rotate: isOpen ? 180 : 0 }} transition={{ duration: 0.2 }}>
          <CaretDown size={12} weight="bold" />
        </motion.span>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 top-full mt-2 w-40 bg-white rounded-md shadow-card-hover
                       border border-farm-divider overflow-hidden z-50"
          >
            {languageOptions.map((opt) => (
              <button
                key={opt.code}
                onClick={() => { setLanguage(opt.code); setIsOpen(false); }}
                className="w-full flex items-center justify-between px-4 py-2.5 text-sm font-noto
                           text-text-primary hover:bg-surface-variant transition-colors duration-150
                           active:bg-primary/10"
              >
                <span className="flex items-center gap-2">
                  <span className="w-6 text-center font-semibold text-primary text-xs">{opt.nativeLabel}</span>
                  <span>{opt.label}</span>
                </span>
                {language === opt.code && (
                  <Check size={14} weight="bold" className="text-primary" />
                )}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/** Notification bell with badge */
function NotificationBell() {
  const [hasNew] = useState(true);
  return (
    <button
      className="relative flex items-center justify-center w-9 h-9 rounded-full
                 hover:bg-surface-variant text-text-secondary hover:text-primary
                 transition-all duration-200 active:scale-95"
      aria-label="Notifications"
    >
      <Bell size={20} weight={hasNew ? 'fill' : 'regular'} />
      {hasNew && (
        <motion.span
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          className="absolute top-1.5 right-1.5 w-2 h-2 bg-accent rounded-full ring-2 ring-white"
        />
      )}
    </button>
  );
}

/** Farmer avatar + profile dropdown */
function ProfileDropdown() {
  const { farmer, logout } = useAppStore();
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();
  const ref = useRef<HTMLDivElement>(null);

  const initials = farmer?.name
    ? farmer.name.split(' ').map((w) => w[0]).slice(0, 2).join('').toUpperCase()
    : 'F';

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        className="flex items-center gap-2 px-2 py-1 rounded-full hover:bg-surface-variant
                   transition-all duration-200 active:scale-95 min-h-[36px]"
        aria-label="Profile menu"
        aria-expanded={isOpen}
      >
        {/* Avatar circle */}
        <div className="w-8 h-8 rounded-full bg-gradient-primary flex items-center justify-center
                        text-white text-xs font-poppins font-bold shrink-0 shadow-sm">
          {initials}
        </div>
        <div className="hidden xl:block text-left">
          <p className="text-xs font-poppins font-semibold text-text-primary leading-tight">
            {farmer?.name ?? 'Farmer'}
          </p>
          <p className="text-xs text-text-secondary leading-tight">
            {farmer?.district ?? 'India'}
          </p>
        </div>
        <motion.span
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="hidden xl:block"
        >
          <CaretDown size={12} weight="bold" className="text-text-secondary" />
        </motion.span>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 top-full mt-2 w-52 bg-white rounded-md shadow-card-hover
                       border border-farm-divider overflow-hidden z-50"
          >
            {/* Profile header */}
            <div className="px-4 py-3 bg-gradient-to-br from-surface-variant to-white border-b border-farm-divider">
              <p className="font-poppins font-semibold text-text-primary text-sm">
                {farmer?.name ?? 'Farmer'}
              </p>
              <p className="text-xs text-text-secondary font-noto mt-0.5">
                {farmer?.phone ?? '+91 XXXXX XXXXX'}
              </p>
            </div>

            {/* Menu items */}
            <button
              onClick={() => { navigate('/profile'); setIsOpen(false); }}
              className="w-full flex items-center gap-3 px-4 py-3 text-sm font-noto text-text-primary
                         hover:bg-surface-variant transition-colors duration-150"
            >
              <UserCircle size={16} className="text-primary" />
              My Profile
            </button>
            <button
              onClick={() => { navigate('/profile'); setIsOpen(false); }}
              className="w-full flex items-center gap-3 px-4 py-3 text-sm font-noto text-text-primary
                         hover:bg-surface-variant transition-colors duration-150"
            >
              <Gear size={16} className="text-text-secondary" />
              Settings
            </button>
            <div className="border-t border-farm-divider" />
            <button
              onClick={() => { logout(); navigate('/login'); setIsOpen(false); }}
              className="w-full flex items-center gap-3 px-4 py-3 text-sm font-noto text-farm-error
                         hover:bg-red-50 transition-colors duration-150"
            >
              <SignOut size={16} />
              Sign Out
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/** Desktop top navigation bar — hidden on mobile */
function TopNavBar() {
  const location = useLocation();

  return (
    <header className="hidden lg:flex items-center justify-between px-6 py-3 bg-white/90
                       border-b border-farm-divider sticky top-0 z-40
                       backdrop-blur-md shadow-[0_1px_8px_rgba(46,125,50,0.06)]">
      {/* ── Logo ── */}
      <NavLink
        to="/dashboard"
        className="flex items-center gap-2 group shrink-0"
        aria-label="FarmGenius home"
      >
        <div className="w-9 h-9 rounded-md bg-gradient-primary flex items-center justify-center shadow-sm
                        group-hover:shadow-md transition-shadow duration-200">
          <Leaf size={20} weight="fill" className="text-white" />
        </div>
        <span className="font-poppins font-bold text-xl text-gradient-primary tracking-tight">
          FarmGenius
        </span>
      </NavLink>

      {/* ── Center nav links ── */}
      <nav className="flex items-center gap-1" aria-label="Main navigation">
        {desktopNavItems.map((item) => {
          const isActive = location.pathname.startsWith(item.to);
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className="relative flex items-center gap-1.5 px-3 py-2 rounded-md text-sm
                         font-poppins font-medium transition-all duration-200 group
                         hover:text-primary hover:bg-surface-variant
                         focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
              style={{ color: isActive ? '#2E7D32' : '#546E7A' }}
            >
              <Icon
                size={16}
                weight={isActive ? 'fill' : 'regular'}
                className="transition-transform duration-200 group-hover:scale-110"
              />
              {item.label}
              {/* Animated underline indicator */}
              {isActive && (
                <motion.span
                  layoutId="desktop-nav-indicator"
                  className="absolute bottom-0 left-2 right-2 h-0.5 bg-primary rounded-full"
                  transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                />
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* ── Right actions ── */}
      <div className="flex items-center gap-2 shrink-0">
        <LanguageSelector />
        <NotificationBell />
        <div className="w-px h-6 bg-farm-divider mx-1" />
        <ProfileDropdown />
      </div>
    </header>
  );
}

/** Mobile bottom navigation bar — hidden on desktop */
function BottomNavBar() {
  const location = useLocation();

  return (
    <nav
      className="lg:hidden fixed bottom-0 left-0 right-0 z-40 bg-white/95 backdrop-blur-md
                 border-t border-farm-divider shadow-[0_-2px_16px_rgba(46,125,50,0.08)]"
      aria-label="Mobile navigation"
      style={{ paddingBottom: 'env(safe-area-inset-bottom, 0px)' }}
    >
      <div className="flex items-end justify-around px-2 pt-1 pb-2">
        {mobileNavItems.map((item) => {
          const isActive = location.pathname.startsWith(item.to);
          const Icon = item.icon;

          // FAB-style Scan button
          if (item.isFab) {
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className="flex flex-col items-center gap-1 -mt-5 focus:outline-none"
                aria-label={item.label}
              >
                <motion.div
                  whileHover={{ scale: 1.08 }}
                  whileTap={{ scale: 0.93 }}
                  className={`w-14 h-14 rounded-full flex items-center justify-center shadow-lg
                             transition-all duration-300
                             ${isActive
                               ? 'bg-gradient-primary shadow-[0_4px_16px_rgba(46,125,50,0.4)]'
                               : 'bg-gradient-primary shadow-[0_4px_12px_rgba(46,125,50,0.3)]'}`}
                >
                  <Icon
                    size={26}
                    weight="bold"
                    className="text-white"
                  />
                </motion.div>
                <span
                  className={`text-[10px] font-poppins font-semibold transition-colors duration-200 ${
                    isActive ? 'text-primary' : 'text-text-secondary'
                  }`}
                >
                  {item.label}
                </span>
              </NavLink>
            );
          }

          // Regular nav tab
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className="relative flex flex-col items-center gap-1 py-1 px-3 min-w-[52px]
                         focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 rounded-md"
              aria-label={item.label}
              aria-current={isActive ? 'page' : undefined}
            >
              <motion.div
                className="relative flex items-center justify-center"
                whileTap={{ scale: 0.85 }}
              >
                {/* Pill background for active state */}
                {isActive && (
                  <motion.div
                    layoutId="mobile-nav-pill"
                    className="absolute inset-[-4px_-8px] bg-surface-variant rounded-full"
                    transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                  />
                )}
                <Icon
                  size={22}
                  weight={isActive ? 'fill' : 'regular'}
                  className={`relative z-10 transition-colors duration-200 ${
                    isActive ? 'text-primary' : 'text-text-secondary'
                  }`}
                />
              </motion.div>

              <span
                className={`text-[10px] font-poppins font-semibold transition-colors duration-200 ${
                  isActive ? 'text-primary' : 'text-text-secondary'
                }`}
              >
                {item.label}
              </span>

              {/* Active dot */}
              <AnimatePresence>
                {isActive && (
                  <motion.span
                    key="dot"
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="absolute -bottom-0.5 w-1 h-1 rounded-full bg-primary"
                  />
                )}
              </AnimatePresence>
            </NavLink>
          );
        })}
      </div>
    </nav>
  );
}

// ─────────────────────────────────────────────────────────────
// Main AppShell
// ─────────────────────────────────────────────────────────────
export default function AppShell() {
  const isOffline = useAppStore((s) => s.isOffline);

  return (
    <div className="flex flex-col min-h-screen bg-bg">
      {/* Offline banner */}
      <AnimatePresence>
        {isOffline && <OfflineBanner key="offline-banner" />}
      </AnimatePresence>

      {/* Desktop top bar */}
      <TopNavBar />

      {/* Page content */}
      <main
        className="flex-1 overflow-y-auto pb-20 lg:pb-0"
        id="main-content"
        tabIndex={-1}
      >
        {/*
          AnimatePresence here wraps page transitions.
          Each page's PageWrapper handles its own entry/exit animation.
        */}
        <Outlet />
      </main>

      {/* Mobile bottom bar */}
      <BottomNavBar />
    </div>
  );
}
