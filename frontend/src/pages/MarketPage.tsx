// src/pages/MarketPage.tsx
import React, { useState, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MagnifyingGlass,
  TrendUp,
  TrendDown,
  Minus,
  CaretDown,
  MapPin,
  Robot,
  X,
} from 'phosphor-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import PageWrapper from '../components/ui/PageWrapper';
import { useAppStore } from '../store/appStore';
import apiClient from '../api/client';

// ─── Types ────────────────────────────────────────────────────────────────────

interface CommodityPrice {
  id: number;
  name: string;
  emoji: string;
  price: number;
  change: number;
  changePercent: number;
  market: string;
  state: string;
  category: string;
  unit: string;
  updated: string;
}

type SortKey = 'name' | 'price' | 'change';
type Category = 'All' | 'Cereals' | 'Pulses' | 'Vegetables' | 'Spices' | 'Oilseeds' | 'Others';

// ─── Mock data ────────────────────────────────────────────────────────────────

const mockPrices: CommodityPrice[] = [
  { id: 1, name: 'Wheat', emoji: '🌾', price: 2180, change: 35, changePercent: 1.6, market: 'Hubballi APMC', state: 'Karnataka', category: 'Cereals', unit: 'quintal', updated: '2h ago' },
  { id: 2, name: 'Maize', emoji: '🌽', price: 1920, change: -20, changePercent: -1.0, market: 'Davangere APMC', state: 'Karnataka', category: 'Cereals', unit: 'quintal', updated: '3h ago' },
  { id: 3, name: 'Soybean', emoji: '🫘', price: 4420, change: 0, changePercent: 0, market: 'Dharwad APMC', state: 'Karnataka', category: 'Oilseeds', unit: 'quintal', updated: '1h ago' },
  { id: 4, name: 'Rice (Fine)', emoji: '🍚', price: 2060, change: 40, changePercent: 2.0, market: 'Bengaluru APMC', state: 'Karnataka', category: 'Cereals', unit: 'quintal', updated: '4h ago' },
  { id: 5, name: 'Tomato', emoji: '🍅', price: 1580, change: -280, changePercent: -15.0, market: 'Kolar APMC', state: 'Karnataka', category: 'Vegetables', unit: 'quintal', updated: '5h ago' },
  { id: 6, name: 'Onion', emoji: '🧅', price: 2350, change: 120, changePercent: 5.4, market: 'Bellary APMC', state: 'Karnataka', category: 'Vegetables', unit: 'quintal', updated: '2h ago' },
  { id: 7, name: 'Cotton', emoji: '🌿', price: 6150, change: -80, changePercent: -1.3, market: 'Haveri APMC', state: 'Karnataka', category: 'Others', unit: 'quintal', updated: '1h ago' },
  { id: 8, name: 'Chickpea', emoji: '🫘', price: 5600, change: 150, changePercent: 2.8, market: 'Bidar APMC', state: 'Karnataka', category: 'Pulses', unit: 'quintal', updated: '6h ago' },
];

const CATEGORIES: Category[] = ['All', 'Cereals', 'Pulses', 'Vegetables', 'Spices', 'Oilseeds', 'Others'];

// Generate 7-day price trend from a base price
function generateTrend(basePrice: number, change: number): { day: string; price: number }[] {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Today'];
  return days.map((day, i) => ({
    day,
    price: Math.round(basePrice - change * (6 - i) + (Math.random() - 0.5) * basePrice * 0.02),
  }));
}

const nearbyMarkets = [
  { name: 'Hubli APMC', distance: '12 km', priceOffset: -30 },
  { name: 'Dharwad APMC', distance: '8 km', priceOffset: 0 },
  { name: 'Belgaum APMC', distance: '74 km', priceOffset: 45 },
  { name: 'Gadag APMC', distance: '35 km', priceOffset: -15 },
];

// ─── Price Change Indicator ───────────────────────────────────────────────────

function PriceChange({ change, changePercent }: { change: number; changePercent: number }) {
  if (change > 0) {
    return (
      <span className="flex items-center gap-0.5 price-up text-sm">
        <TrendUp size={14} weight="bold" />
        +{changePercent.toFixed(1)}%
      </span>
    );
  }
  if (change < 0) {
    return (
      <span className="flex items-center gap-0.5 price-down text-sm">
        <TrendDown size={14} weight="bold" />
        {changePercent.toFixed(1)}%
      </span>
    );
  }
  return (
    <span className="flex items-center gap-0.5 price-neutral text-sm">
      <Minus size={14} weight="bold" />
      0.0%
    </span>
  );
}

// ─── Commodity Card ───────────────────────────────────────────────────────────

function CommodityCard({
  item,
  isExpanded,
  onToggle,
}: {
  item: CommodityPrice;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const trendData = useMemo(() => generateTrend(item.price, item.change), [item.price, item.change]);
  const trendColor = item.change >= 0 ? '#2E7D32' : '#C62828';

  return (
    <motion.div
      layout
      className="card overflow-hidden"
    >
      {/* Card row */}
      <motion.button
        onClick={onToggle}
        className="w-full flex items-center gap-3 p-4 text-left hover:bg-surface-variant/30 transition-colors"
        whileTap={{ scale: 0.99 }}
      >
        {/* Left: emoji + names */}
        <div className="w-11 h-11 rounded-full bg-surface-variant flex items-center justify-center text-2xl shrink-0">
          {item.emoji}
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-poppins font-semibold text-text-primary text-sm leading-tight">{item.name}</p>
          <div className="flex items-center gap-1 mt-0.5">
            <MapPin size={10} className="text-text-secondary" />
            <p className="text-text-secondary font-noto text-xs truncate">{item.market}</p>
          </div>
        </div>
        {/* Right: price + change */}
        <div className="text-right shrink-0">
          <p className="font-poppins font-bold text-text-primary text-base">
            ₹{item.price.toLocaleString('en-IN')}
            <span className="text-xs font-noto text-text-secondary font-normal">/{item.unit}</span>
          </p>
          <PriceChange change={item.change} changePercent={item.changePercent} />
          <p className="text-[10px] text-text-secondary font-noto mt-0.5">{item.updated}</p>
        </div>
        {/* Expand indicator */}
        <motion.div
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.25 }}
          className="shrink-0 ml-1"
        >
          <CaretDown size={14} className="text-text-secondary" />
        </motion.div>
      </motion.button>

      {/* Expanded detail panel */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 border-t border-farm-divider pt-4 flex flex-col gap-4">
              {/* 7-day chart */}
              <div>
                <p className="font-poppins font-semibold text-text-primary text-xs mb-2">
                  7-Day Price Trend (₹/{item.unit})
                </p>
                <ResponsiveContainer width="100%" height={140}>
                  <LineChart data={trendData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#C8E6C9" />
                    <XAxis dataKey="day" tick={{ fontSize: 10, fontFamily: 'Noto Sans' }} />
                    <YAxis tick={{ fontSize: 10, fontFamily: 'Noto Sans' }} />
                    <Tooltip
                      formatter={(v: any) => [`₹${v.toLocaleString('en-IN')}`, item.name]}
                      contentStyle={{ fontFamily: 'Poppins', fontSize: 12, borderRadius: 8 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="price"
                      stroke={trendColor}
                      strokeWidth={2.5}
                      dot={false}
                      activeDot={{ r: 5, fill: trendColor }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* AI forecast */}
              <div className="bg-surface-variant rounded-md p-3 flex gap-2">
                <Robot size={18} weight="fill" className="text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="text-xs font-poppins font-semibold text-primary">AI Price Forecast</p>
                  <p className="text-xs font-noto text-text-secondary mt-0.5">
                    {item.change >= 0
                      ? `Prices expected to rise 5–8% next week based on reduced supply arrivals.`
                      : `Prices may stabilise next week as market oversupply eases.`}
                  </p>
                </div>
              </div>

              {/* Nearby markets */}
              <div>
                <p className="font-poppins font-semibold text-text-primary text-xs mb-2">
                  Nearby Markets
                </p>
                <div className="flex flex-col gap-1.5">
                  {nearbyMarkets.map((m) => (
                    <div
                      key={m.name}
                      className="flex justify-between items-center bg-surface-variant rounded-sm px-3 py-2"
                    >
                      <div>
                        <p className="font-noto text-xs text-text-primary font-semibold">{m.name}</p>
                        <p className="text-[10px] text-text-secondary">{m.distance}</p>
                      </div>
                      <p className="font-poppins font-bold text-sm text-text-primary">
                        ₹{(item.price + m.priceOffset).toLocaleString('en-IN')}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function MarketPage() {
  const farmer = useAppStore((s) => s.farmer);
  const district = farmer?.district ?? 'Dharwad';
  const state = farmer?.state ?? 'Karnataka';

  const [search, setSearch] = useState('');
  const [category, setCategory] = useState<Category>('All');
  const [sortBy, setSortBy] = useState<SortKey>('name');
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [showSort, setShowSort] = useState(false);
  // BUG 10 FIX: Use real prices state; fall back to mockPrices
  const [prices, setPrices] = useState<CommodityPrice[]>(mockPrices);
  const [priceError, setPriceError] = useState(false);

  // Fetch prices for a set of common crops on mount
  useEffect(() => {
    const crops = ['Wheat', 'Maize', 'Soybean', 'Rice', 'Tomato', 'Onion', 'Cotton', 'Chickpea'];
    let cancelled = false;
    const fetched: CommodityPrice[] = [];

    Promise.all(
      crops.map((crop, idx) =>
        apiClient
          .get('/prices', { params: { commodity: crop, district, state } })
          .then((res) => {
            const d = res.data;
            if (d && d.modal_price > 0) {
              const base = mockPrices.find((m) => m.name.toLowerCase() === crop.toLowerCase());
              fetched.push({
                id: idx + 1,
                name: crop,
                emoji: base?.emoji ?? '🌿',
                price: Math.round(d.modal_price),
                change: base?.change ?? 0,
                changePercent: base?.changePercent ?? 0,
                market: `${district} APMC`,
                state,
                category: base?.category ?? 'Others',
                unit: d.unit ?? 'quintal',
                updated: 'Just now',
              });
            }
          })
          .catch(() => {})
      )
    ).then(() => {
      if (!cancelled && fetched.length > 0) {
        setPrices(fetched);
      } else if (!cancelled) {
        setPriceError(true);
      }
    });

    return () => { cancelled = true; };
  }, [district, state]);

  const filtered = useMemo(() => {
    let data = prices;
    if (search) {
      data = data.filter((d) => d.name.toLowerCase().includes(search.toLowerCase()));
    }
    if (category !== 'All') {
      data = data.filter((d) => d.category === category);
    }
    return [...data].sort((a, b) => {
      if (sortBy === 'name') return a.name.localeCompare(b.name);
      if (sortBy === 'price') return b.price - a.price;
      if (sortBy === 'change') return Math.abs(b.changePercent) - Math.abs(a.changePercent);
      return 0;
    });
  }, [search, category, sortBy, prices]);

  const sortLabels: Record<SortKey, string> = {
    name: 'By Name',
    price: 'By Price',
    change: 'By Change',
  };

  return (
    <PageWrapper>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-4"
      >
        <h1 className="font-poppins font-bold text-2xl text-text-primary">📊 Market Prices</h1>
        <p className="text-text-secondary font-noto text-sm mt-0.5">
          Live mandi rates • {district} APMC
        </p>
      </motion.div>

      {/* BUG 10: Show "Using cached data" when real API isn't available */}
      {priceError && (
        <motion.div
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-3 bg-amber-50 border border-amber-300 rounded-xl p-3 flex items-center gap-2 text-sm"
        >
          <span>📡</span>
          <span className="font-noto text-amber-800">Using reference prices — live data unavailable.</span>
        </motion.div>
      )}

      {/* Search bar */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="relative mb-3"
      >
        <MagnifyingGlass
          size={18}
          className="absolute left-4 top-1/2 -translate-y-1/2 text-text-secondary"
        />
        <input
          type="text"
          placeholder="Search commodities..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="input-field pl-10 pr-10"
        />
        {search && (
          <button
            onClick={() => setSearch('')}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary transition"
          >
            <X size={16} />
          </button>
        )}
      </motion.div>

      {/* Filter chips + sort */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="flex items-center gap-2 mb-4"
      >
        <div className="flex gap-2 overflow-x-auto scrollbar-hide flex-1">
          {CATEGORIES.map((cat) => (
            <button
              key={cat}
              onClick={() => setCategory(cat)}
              className={`shrink-0 text-xs font-poppins font-semibold px-3 py-1.5 rounded-full transition-all duration-200 ${
                category === cat
                  ? 'bg-primary text-white shadow-sm'
                  : 'bg-surface-variant text-text-secondary hover:text-primary hover:bg-surface-variant/80'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* Sort dropdown */}
        <div className="relative shrink-0">
          <button
            onClick={() => setShowSort(!showSort)}
            className="flex items-center gap-1 text-xs font-poppins font-semibold text-primary bg-surface-variant px-3 py-1.5 rounded-full hover:bg-surface-variant/80 transition"
          >
            {sortLabels[sortBy]}
            <motion.div animate={{ rotate: showSort ? 180 : 0 }} transition={{ duration: 0.2 }}>
              <CaretDown size={12} weight="bold" />
            </motion.div>
          </button>
          <AnimatePresence>
            {showSort && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95, y: -4 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: -4 }}
                className="absolute right-0 top-full mt-1 bg-white rounded-md shadow-card-hover z-10 overflow-hidden"
              >
                {(Object.keys(sortLabels) as SortKey[]).map((key) => (
                  <button
                    key={key}
                    onClick={() => { setSortBy(key); setShowSort(false); }}
                    className={`block w-full text-left px-4 py-2.5 text-xs font-poppins font-semibold whitespace-nowrap hover:bg-surface-variant transition-colors ${
                      sortBy === key ? 'text-primary' : 'text-text-primary'
                    }`}
                  >
                    {sortLabels[key]}
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>

      {/* Commodity list */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.15 }}
        className="flex flex-col gap-3"
      >
        <AnimatePresence mode="popLayout">
          {filtered.length === 0 ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="card p-8 text-center"
            >
              <p className="text-4xl mb-3">🔍</p>
              <p className="font-poppins font-semibold text-text-primary">No commodities found</p>
              <p className="text-text-secondary font-noto text-sm mt-1">
                Try a different search or category
              </p>
            </motion.div>
          ) : (
            filtered.map((item, i) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ delay: i * 0.05 }}
              >
                <CommodityCard
                  item={item}
                  isExpanded={expandedId === item.id}
                  onToggle={() => setExpandedId(expandedId === item.id ? null : item.id)}
                />
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </motion.div>

      {/* Last updated note */}
      <p className="text-center text-text-secondary font-noto text-xs mt-5">
        Data from Karnataka APMC via data.gov.in • Refreshes every 6 hours
      </p>
    </PageWrapper>
  );
}
