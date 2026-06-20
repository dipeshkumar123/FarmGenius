// src/pages/DashboardPage.tsx
import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BellSimple,
  Leaf as Plant,
  Scan,
  ChatCircleDots,
  ChartLineUp,
  CloudSun,
  Bank,
  Drop,
  Wind,
  Warning,
  Robot,
  MagnifyingGlass as Microscope,
  TrendUp,
  CaretRight,
  ArrowUp,
  ArrowDown,
  Minus,
} from 'phosphor-react';
import { useAppStore } from '../store/appStore';
import PageWrapper from '../components/ui/PageWrapper';

// ─── Types ────────────────────────────────────────────────────────────────────

interface WeatherData {
  temp: number;
  condition: string;
  humidity: number;
  windKph: number;
  advisory: string;
  emoji: string;
}

interface PriceCard {
  id: string;
  crop: string;
  emoji: string;
  price: number;
  unit: string;
  change: number; // percent, negative = down, 0 = neutral
}

interface ActivityItem {
  id: string;
  icon: 'ai' | 'scan' | 'price';
  text: string;
  timeAgo: string;
}

// ─── Mock Data ────────────────────────────────────────────────────────────────

const MOCK_WEATHER: WeatherData = {
  temp: 32,
  condition: 'Partly Cloudy',
  humidity: 65,
  windKph: 12,
  advisory: '🌧️ Rain expected Thursday — avoid spraying pesticides',
  emoji: '⛅',
};

const MOCK_PRICES: PriceCard[] = [
  { id: '1', crop: 'Wheat',   emoji: '🌾', price: 2180, unit: '₹/qtl', change:  3 },
  { id: '2', crop: 'Maize',   emoji: '🌽', price: 1920, unit: '₹/qtl', change: -1 },
  { id: '3', crop: 'Soybean', emoji: '🫘', price: 4420, unit: '₹/qtl', change:  0 },
  { id: '4', crop: 'Rice',    emoji: '🍚', price: 2060, unit: '₹/qtl', change:  2 },
  { id: '5', crop: 'Cotton',  emoji: '🌿', price: 6150, unit: '₹/qtl', change: -2 },
];

const MOCK_ACTIVITY: ActivityItem[] = [
  { id: '1', icon: 'ai',    text: 'AI advised wheat irrigation schedule',         timeAgo: '2h ago' },
  { id: '2', icon: 'scan',  text: 'Tomato Late Blight detected — 94% confidence', timeAgo: '1d ago' },
  { id: '3', icon: 'price', text: 'Wheat prices checked for Dharwad APMC',        timeAgo: '2d ago' },
];

interface QuickAction {
  id: string;
  emoji: string;
  label: string;
  desc: string;
  route: string;
  colorFrom: string;
  colorTo: string;
}

const QUICK_ACTIONS: QuickAction[] = [
  { id: 'crops',   emoji: '🌱', label: 'Crop Advice',    desc: 'Sowing & care',      route: '/crops',   colorFrom: '#2E7D32', colorTo: '#60AD5E' },
  { id: 'scan',    emoji: '🔬', label: 'Scan Disease',   desc: 'Photo diagnosis',    route: '/scan',    colorFrom: '#1565C0', colorTo: '#42A5F5' },
  { id: 'chat',    emoji: '💬', label: 'Ask AI',         desc: 'Any question',       route: '/chat',    colorFrom: '#6A1B9A', colorTo: '#AB47BC' },
  { id: 'market',  emoji: '📈', label: 'Market Prices',  desc: 'Live mandi rates',   route: '/market',  colorFrom: '#E65100', colorTo: '#FF8A65' },
  { id: 'weather', emoji: '☁️', label: 'Weather',        desc: '7-day forecast',     route: '/weather', colorFrom: '#006064', colorTo: '#26C6DA' },
  { id: 'schemes', emoji: '🏛️', label: 'Schemes',        desc: 'Govt benefits',      route: '/schemes', colorFrom: '#558B2F', colorTo: '#8BC34A' },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────

function getGreeting(): string {
  const h = new Date().getHours();
  if (h < 12) return 'Good morning';
  if (h < 17) return 'Good afternoon';
  return 'Good evening';
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function ShimmerBlock({ className }: { className: string }) {
  return <div className={`shimmer rounded-md ${className}`} />;
}

function WeatherCardSkeleton() {
  return (
    <div className="rounded-md p-5 bg-gradient-sky min-h-[148px] flex flex-col gap-3">
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <ShimmerBlock className="h-12 w-24 opacity-40" />
          <ShimmerBlock className="h-4 w-32 opacity-40" />
          <ShimmerBlock className="h-4 w-20 opacity-40" />
        </div>
        <ShimmerBlock className="h-16 w-16 rounded-full opacity-40" />
      </div>
      <ShimmerBlock className="h-4 w-3/4 mt-2 opacity-40" />
    </div>
  );
}

interface PriceChangeBadgeProps {
  change: number;
}
function PriceChangeBadge({ change }: PriceChangeBadgeProps) {
  if (change > 0)
    return (
      <span className="inline-flex items-center gap-0.5 text-xs font-poppins font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full">
        <ArrowUp size={10} weight="bold" />
        {change}%
      </span>
    );
  if (change < 0)
    return (
      <span className="inline-flex items-center gap-0.5 text-xs font-poppins font-semibold text-red-700 bg-red-50 px-2 py-0.5 rounded-full">
        <ArrowDown size={10} weight="bold" />
        {Math.abs(change)}%
      </span>
    );
  return (
    <span className="inline-flex items-center gap-0.5 text-xs font-poppins font-semibold text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">
      <Minus size={10} weight="bold" />
      0%
    </span>
  );
}

interface ActivityIconProps {
  type: ActivityItem['icon'];
}
function ActivityIcon({ type }: ActivityIconProps) {
  const base = 'w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0';
  if (type === 'ai')
    return (
      <div className={`${base} bg-purple-100`}>
        <Robot size={18} weight="fill" className="text-purple-600" />
      </div>
    );
  if (type === 'scan')
    return (
      <div className={`${base} bg-blue-100`}>
        <Microscope size={18} weight="fill" className="text-blue-600" />
      </div>
    );
  return (
    <div className={`${base} bg-amber-100`}>
      <TrendUp size={18} weight="fill" className="text-amber-600" />
    </div>
  );
}

// ─── Animation Variants ───────────────────────────────────────────────────────

const sectionVariant = {
  hidden: { opacity: 0, y: 18 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.42, },
  }),
};

const cardStagger = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.055 } },
};

const cardChild = {
  hidden: { opacity: 0, y: 16, scale: 0.96 },
  visible: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.35, } },
};

const slideInRight = {
  hidden: { opacity: 0, x: 30 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.45, },
  },
};

// ─── Main Component ───────────────────────────────────────────────────────────

export default function DashboardPage() {
  const navigate = useNavigate();
  const farmer = useAppStore((s) => s.farmer);
  const isOffline = useAppStore((s) => s.isOffline);

  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [weatherLoading, setWeatherLoading] = useState(true);
  const [prices] = useState<PriceCard[]>(MOCK_PRICES);
  const [notifCount] = useState(2);

  const farmerName = farmer?.name ?? 'Ramesh';
  const district    = farmer?.district ?? 'Dharwad';
  const state       = farmer?.state ?? 'Karnataka';

  // Simulate weather fetch
  const fetchWeather = useCallback(async () => {
    setWeatherLoading(true);
    try {
      // Simulate network latency; swap with real API call when backend is live
      await new Promise((r) => setTimeout(r, 1200));
      setWeather(MOCK_WEATHER);
    } finally {
      setWeatherLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWeather();
  }, [fetchWeather]);

  return (
    <PageWrapper>
      <div className="px-4 py-6 max-w-4xl mx-auto space-y-6">

        {/* ── Offline Banner ────────────────────────────────────────── */}
        <AnimatePresence>
          {isOffline && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="bg-amber-100 border border-amber-300 text-amber-800 rounded-md px-4 py-2.5 flex items-center gap-2 text-sm font-noto"
            >
              <Warning size={16} weight="fill" />
              You're offline. Showing cached data.
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Section 1: Greeting Header ───────────────────────────── */}
        <motion.div
          custom={0}
          variants={sectionVariant}
          initial="hidden"
          animate="visible"
          className="flex items-start justify-between"
        >
          <div>
            <h1 className="font-poppins font-bold text-2xl text-text-primary leading-tight">
              {getGreeting()}, {farmerName}! 🌅
            </h1>
            <p className="text-text-secondary text-sm mt-1 font-noto flex items-center gap-1">
              <span>📍</span>
              <span>{district}, {state}</span>
            </p>
          </div>

          <motion.button
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.92 }}
            onClick={() => {/* open notifications panel */}}
            className="relative w-11 h-11 rounded-full bg-white shadow-card flex items-center justify-center mt-0.5 flex-shrink-0"
            aria-label="Notifications"
          >
            <BellSimple size={22} weight="duotone" className="text-text-primary" />
            {notifCount > 0 && (
              <span className="absolute -top-0.5 -right-0.5 w-5 h-5 rounded-full bg-farm-error text-white text-[10px] font-poppins font-bold flex items-center justify-center">
                {notifCount}
              </span>
            )}
          </motion.button>
        </motion.div>

        {/* ── Section 2: Weather Card ───────────────────────────────── */}
        <motion.div
          custom={1}
          variants={sectionVariant}
          initial="hidden"
          animate="visible"
        >
          <AnimatePresence mode="wait">
            {weatherLoading ? (
              <motion.div key="skeleton" initial={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <WeatherCardSkeleton />
              </motion.div>
            ) : weather ? (
              <motion.div
                key="weather"
                initial={{ opacity: 0, scale: 0.97 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4 }}
                onClick={() => navigate('/weather')}
                className="rounded-md p-5 bg-gradient-sky cursor-pointer select-none
                           shadow-blue hover:shadow-[0_12px_32px_rgba(21,101,192,0.35)]
                           transition-shadow duration-300 overflow-hidden relative"
              >
                {/* Decorative blobs */}
                <div className="absolute top-0 right-0 w-40 h-40 rounded-full bg-white/5 -translate-y-1/2 translate-x-1/4 pointer-events-none" />
                <div className="absolute bottom-0 left-10 w-24 h-24 rounded-full bg-white/5 translate-y-1/2 pointer-events-none" />

                <div className="flex items-start justify-between relative">
                  {/* Left info */}
                  <div className="space-y-1">
                    <div className="flex items-end gap-1">
                      <span className="text-5xl font-poppins font-bold text-white leading-none">
                        {weather.temp}°
                      </span>
                      <span className="text-lg font-poppins text-white/80 mb-1">C</span>
                    </div>
                    <p className="text-white font-poppins font-semibold text-base">
                      {weather.condition}
                    </p>
                    <div className="flex items-center gap-4 mt-2">
                      <span className="flex items-center gap-1 text-white/80 text-sm font-noto">
                        <Drop size={14} weight="fill" />
                        {weather.humidity}%
                      </span>
                      <span className="flex items-center gap-1 text-white/80 text-sm font-noto">
                        <Wind size={14} weight="fill" />
                        {weather.windKph} km/h
                      </span>
                    </div>
                  </div>

                  {/* Animated weather emoji */}
                  <motion.div
                    animate={{
                      y: [0, -8, 0],
                      rotate: [0, 4, -4, 0],
                    }}
                    transition={{
                      duration: 3.5,
                      repeat: Infinity,
                      
                    }}
                    className="text-7xl select-none leading-none"
                    role="img"
                    aria-label={weather.condition}
                  >
                    {weather.emoji}
                  </motion.div>
                </div>

                {/* Advisory strip */}
                <div className="mt-4 pt-3 border-t border-white/20">
                  <p className="text-white/90 text-sm font-noto">
                    {weather.advisory}
                  </p>
                </div>

                {/* Tap cue */}
                <div className="absolute bottom-4 right-5 flex items-center gap-1 text-white/60 text-xs font-noto">
                  See forecast <CaretRight size={12} />
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="error"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="rounded-md p-5 bg-gradient-sky flex items-center justify-center min-h-[120px]"
              >
                <button
                  onClick={fetchWeather}
                  className="text-white font-noto text-sm underline"
                >
                  Failed to load weather — tap to retry
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* ── Section 3: Quick Actions Grid ─────────────────────────── */}
        <motion.div
          custom={2}
          variants={sectionVariant}
          initial="hidden"
          animate="visible"
        >
          <h2 className="section-title mb-3">Quick Actions</h2>
          <motion.div
            variants={cardStagger}
            initial="hidden"
            animate="visible"
            className="grid grid-cols-2 sm:grid-cols-3 gap-3"
          >
            {QUICK_ACTIONS.map((action) => (
              <motion.button
                key={action.id}
                variants={cardChild}
                whileHover={{ y: -4, boxShadow: '0 8px 24px rgba(46,125,50,0.16)' }}
                whileTap={{ scale: 0.94 }}
                onClick={() => navigate(action.route)}
                className="quick-action text-left"
                aria-label={action.label}
              >
                {/* Icon circle */}
                <div
                  className="w-12 h-12 rounded-full flex items-center justify-center text-2xl shadow-sm mb-1"
                  style={{
                    background: `linear-gradient(135deg, ${action.colorFrom}, ${action.colorTo})`,
                  }}
                >
                  {action.emoji}
                </div>
                <span className="font-poppins font-semibold text-sm text-text-primary leading-tight">
                  {action.label}
                </span>
                <span className="font-noto text-xs text-text-secondary">
                  {action.desc}
                </span>
              </motion.button>
            ))}
          </motion.div>
        </motion.div>

        {/* ── Section 4: Market Prices ──────────────────────────────── */}
        <motion.div
          custom={3}
          variants={sectionVariant}
          initial="hidden"
          animate="visible"
        >
          <div className="flex items-center justify-between mb-3">
            <h2 className="section-title">Today's Mandi Prices</h2>
            <button
              onClick={() => navigate('/market')}
              className="flex items-center gap-0.5 text-primary text-sm font-poppins font-semibold hover:underline"
            >
              View all <CaretRight size={14} weight="bold" />
            </button>
          </div>

          {/* Horizontal scrollable row */}
          <div className="flex gap-3 overflow-x-auto scrollbar-hide pb-1 -mx-1 px-1">
            {prices.map((p, i) => (
              <motion.div
                key={p.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.06 * i, duration: 0.38 }}
                whileHover={{ y: -3, boxShadow: '0 8px 20px rgba(46,125,50,0.14)' }}
                onClick={() => navigate('/market')}
                className="flex-shrink-0 bg-white rounded-md shadow-card px-4 py-3
                           min-w-[120px] cursor-pointer border border-transparent
                           hover:border-primary/15 transition-all duration-200"
              >
                <div className="text-2xl mb-1">{p.emoji}</div>
                <p className="font-poppins font-semibold text-text-primary text-sm leading-tight">
                  {p.crop}
                </p>
                <p className="font-poppins font-bold text-base text-text-primary mt-0.5">
                  ₹{p.price.toLocaleString('en-IN')}
                </p>
                <p className="font-noto text-[10px] text-text-secondary mb-1.5">{p.unit}</p>
                <PriceChangeBadge change={p.change} />
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* ── Section 5: Pest Alert ─────────────────────────────────── */}
        <motion.div
          variants={slideInRight}
          initial="hidden"
          animate="visible"
          custom={4}
          className="rounded-md p-4 bg-amber-50 border border-amber-200 overflow-hidden relative"
        >
          {/* Decorative circle */}
          <div className="absolute -right-8 -top-8 w-32 h-32 rounded-full bg-amber-200/40 pointer-events-none" />

          <div className="flex items-start gap-3 relative">
            <motion.div
              animate={{ rotate: [0, -8, 8, -8, 0] }}
              transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
              className="w-10 h-10 rounded-full bg-amber-100 border border-amber-300
                         flex items-center justify-center flex-shrink-0 text-xl"
            >
              ⚠️
            </motion.div>

            <div className="flex-1 min-w-0">
              <h3 className="font-poppins font-bold text-amber-900 text-sm">
                🚨 Pest Advisory — Your Region
              </h3>
              <p className="font-noto text-amber-800 text-sm mt-1 leading-relaxed">
                High aphid risk detected in {district} district this week.
                Check your wheat and tomato crops.
              </p>
              <motion.button
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.96 }}
                onClick={() => navigate('/chat')}
                className="mt-3 inline-flex items-center gap-1.5 bg-accent text-text-primary
                           font-poppins font-semibold text-sm px-4 py-2 rounded-full
                           shadow-accent hover:bg-accent-dark transition-colors duration-200"
              >
                Learn More
                <CaretRight size={14} weight="bold" />
              </motion.button>
            </div>
          </div>
        </motion.div>

        {/* ── Section 6: Recent Activity ────────────────────────────── */}
        <motion.div
          custom={5}
          variants={sectionVariant}
          initial="hidden"
          animate="visible"
          className="pb-4"
        >
          <h2 className="section-title mb-3">Recent Activity</h2>

          <div className="bg-white rounded-md shadow-card overflow-hidden divide-y divide-farm-divider">
            {MOCK_ACTIVITY.map((item, i) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.07 * i + 0.2, duration: 0.35 }}
                className="flex items-center gap-3 px-4 py-3.5
                           hover:bg-surface-variant transition-colors duration-150 cursor-pointer"
              >
                <ActivityIcon type={item.icon} />
                <div className="flex-1 min-w-0">
                  <p className="font-noto text-sm text-text-primary leading-snug truncate">
                    {item.text}
                  </p>
                </div>
                <span className="font-noto text-xs text-text-secondary flex-shrink-0 ml-2">
                  {item.timeAgo}
                </span>
              </motion.div>
            ))}
          </div>
        </motion.div>

      </div>
    </PageWrapper>
  );
}
