// src/pages/WeatherPage.tsx
// BUG 9 FIX: Replaced all hardcoded mock data with a real API call to /api/weather.
// Shows skeleton loading state while fetching, displays real data for the farmer's district,
// and falls back to sensible static data if the API is unavailable.
import React, { useEffect, useState } from 'react';
import { PageWrapper, AnimatedSection } from '../components/ui/PageWrapper';
import { MapPin, Drop, Wind, Warning } from 'phosphor-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { useAppStore } from '../store/appStore';
import apiClient from '../api/client';
import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';

// ─── Types ────────────────────────────────────────────────────────────────────

interface WeatherDay {
  date: string;
  max_temp: number;
  min_temp: number;
  rainfall_mm: number;
  wind_kmh: number;
  farming_advisory: string;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function getWeatherEmoji(rain: number, maxTemp: number): string {
  if (rain > 20) return '🌧️';
  if (rain > 5) return '🌥️';
  if (maxTemp > 35) return '☀️';
  return '⛅';
}

function getDayLabel(dateStr: string, idx: number, t: any, locale: string): string {
  if (idx === 0) return t('weather.today', 'Today');
  const date = new Date(dateStr);
  return date.toLocaleDateString(locale, { weekday: 'short' });
}

// ─── Skeleton ─────────────────────────────────────────────────────────────────

function WeatherSkeleton() {
  return (
    <div className="animate-pulse space-y-6">
      <div className="bg-blue-200 rounded-2xl h-48 w-full" />
      <div className="flex gap-3 overflow-hidden">
        {[...Array(7)].map((_, i) => (
          <div key={i} className="flex-shrink-0 w-24 bg-gray-200 rounded-xl h-36" />
        ))}
      </div>
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="bg-gray-200 h-16 rounded-xl" />
        ))}
      </div>
    </div>
  );
}

// ─── Static fallback data ─────────────────────────────────────────────────────

const FALLBACK_FORECAST: WeatherDay[] = [
  { date: new Date().toISOString().slice(0, 10), max_temp: 32, min_temp: 22, rainfall_mm: 10, wind_kmh: 14, farming_advisory: 'Conditions look favorable for standard farming activities.' },
  { date: '', max_temp: 34, min_temp: 23, rainfall_mm: 5, wind_kmh: 12, farming_advisory: 'Conditions look favorable for standard farming activities.' },
  { date: '', max_temp: 30, min_temp: 21, rainfall_mm: 20, wind_kmh: 18, farming_advisory: 'Conditions look favorable for standard farming activities.' },
  { date: '', max_temp: 27, min_temp: 20, rainfall_mm: 25, wind_kmh: 22, farming_advisory: 'Heavy rain expected — avoid spraying pesticides and monitor drainage.' },
  { date: '', max_temp: 26, min_temp: 19, rainfall_mm: 30, wind_kmh: 20, farming_advisory: 'Heavy rain expected — avoid spraying pesticides and monitor drainage.' },
  { date: '', max_temp: 31, min_temp: 21, rainfall_mm: 15, wind_kmh: 10, farming_advisory: 'Conditions look favorable for standard farming activities.' },
  { date: '', max_temp: 35, min_temp: 24, rainfall_mm: 5, wind_kmh: 8, farming_advisory: 'High temperatures expected. Ensure adequate crop irrigation.' },
];

// ─── Main Component ───────────────────────────────────────────────────────────

export default function WeatherPage() {
  const { t, i18n } = useTranslation();
  const farmer = useAppStore((s) => s.farmer);
  const district = farmer?.district ?? 'Dharwad';
  const state = farmer?.state ?? 'Karnataka';

  const localeMap: Record<string, string> = {
    en: 'en-IN',
    hi: 'hi-IN',
    kn: 'kn-IN',
    te: 'te-IN',
    ta: 'ta-IN',
    mr: 'mr-IN',
  };
  const currentLocale = localeMap[i18n.language] || 'en-IN';

  const [forecast, setForecast] = useState<WeatherDay[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(false);

    apiClient
      .get<WeatherDay[]>('/weather', {
        params: { district, state },
      })
      .then((res) => {
        if (!cancelled && res.data && res.data.length > 0) {
          setForecast(res.data);
        } else if (!cancelled) {
          setForecast(FALLBACK_FORECAST);
          setError(true);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setForecast(FALLBACK_FORECAST);
          setError(true);
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [district, state]);

  const today = forecast[0];

  return (
    <PageWrapper className="p-4 pb-24 bg-[#F1F8E9]">
      {isLoading ? (
        <WeatherSkeleton />
      ) : (
        <>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-4 bg-amber-50 border border-amber-300 rounded-xl p-3 flex items-center gap-2 text-sm"
            >
              <Warning size={16} weight="fill" className="text-amber-600 shrink-0" />
              <span className="font-noto text-amber-800">
                {t('weather.estimated_warning', 'Using estimated forecast — live weather unavailable.')}
              </span>
            </motion.div>
          )}

          {/* Hero Card */}
          <AnimatedSection custom={0} className="mb-6">
            <div className="bg-gradient-to-br from-blue-500 to-blue-700 rounded-2xl p-5 text-white shadow-lg">
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center space-x-2 bg-white/20 rounded-full px-3 py-1">
                  <MapPin weight="fill" size={16} />
                  <span className="text-sm font-medium">{district}, {state}</span>
                </div>
                <div className="text-xs bg-white/20 rounded-full px-2 py-1">
                  {error ? `📡 ${t('weather.estimated_status', 'Estimated')}` : `🟢 ${t('weather.live_status', 'Live')}`}
                </div>
              </div>

              {today && (
                <>
                  <div className="flex justify-between items-center mb-6">
                    <div>
                      <div className="text-5xl font-bold font-poppins mb-1">{Math.round(today.max_temp)}°C</div>
                      <div className="text-xl font-medium">
                        {today.rainfall_mm > 20 
                          ? t('weather.conditions.heavy_rain', 'Heavy Rain') 
                          : today.rainfall_mm > 5 
                          ? t('weather.conditions.partly_cloudy', 'Partly Cloudy') 
                          : today.max_temp > 35 
                          ? t('weather.conditions.sunny_hot', 'Sunny & Hot') 
                          : t('weather.conditions.partly_cloudy', 'Partly Cloudy')}
                      </div>
                    </div>
                    <div className="text-6xl drop-shadow-md">
                      {getWeatherEmoji(today.rainfall_mm, today.max_temp)}
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-y-3 gap-x-2 text-sm bg-white/10 rounded-xl p-3 backdrop-blur-sm">
                    <div className="flex items-center space-x-1">
                      <span className="opacity-80">{t('weather.high_low', 'High/Low')}</span>
                      <span className="font-semibold">{Math.round(today.max_temp)}°/{Math.round(today.min_temp)}°</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <span className="opacity-80">{t('weather.rain', 'Rain')}</span>
                      <span className="font-semibold">{today.rainfall_mm}mm</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Drop size={16} weight="fill" className="text-blue-200" />
                      <span className="font-semibold">{t('weather.humid', 'Humid')}</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Wind size={16} weight="fill" className="text-blue-200" />
                      <span className="font-semibold">{Math.round(today.wind_kmh)} {t('weather.wind_unit', 'km/h')}</span>
                    </div>
                    <div className="flex items-center space-x-1 col-span-2">
                      <span className="opacity-80 text-xs">
                        {today.farming_advisory === 'Conditions look favorable for standard farming activities.'
                          ? t('weather.advisories.favorable', today.farming_advisory)
                          : today.farming_advisory === 'Heavy rain expected — avoid spraying pesticides and monitor drainage.'
                          ? t('weather.advisories.heavy_rain', today.farming_advisory)
                          : today.farming_advisory === 'High temperatures expected. Ensure adequate crop irrigation.'
                          ? t('weather.advisories.high_temp', today.farming_advisory)
                          : today.farming_advisory}
                      </span>
                    </div>
                  </div>
                </>
              )}
            </div>
          </AnimatedSection>

          {/* 7-day forecast */}
          <AnimatedSection custom={1} className="mb-6">
            <h2 className="text-lg font-poppins font-semibold text-[#1B2B1D] mb-3">{t('weather.forecast_title', '7-Day Forecast')}</h2>
            <div className="flex overflow-x-auto space-x-3 pb-2 -mx-4 px-4 snap-x hide-scrollbar">
              {forecast.map((day, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="flex-shrink-0 w-24 bg-white rounded-xl p-3 shadow-sm flex flex-col items-center snap-center hover:shadow-md transition-shadow"
                >
                  <span className="text-sm font-medium text-[#546E7A] mb-1">{getDayLabel(day.date, idx, t, currentLocale)}</span>
                  <span className="text-2xl mb-2">{getWeatherEmoji(day.rainfall_mm, day.max_temp)}</span>
                  <div className="flex space-x-2 text-sm mb-3">
                    <span className="font-semibold text-[#1B2B1D]">{Math.round(day.max_temp)}°</span>
                    <span className="text-[#546E7A]">{Math.round(day.min_temp)}°</span>
                  </div>
                  <div className="w-full bg-[#E8F5E9] rounded-full h-1.5 mb-1 overflow-hidden">
                    <div
                      className="bg-blue-400 h-full rounded-full"
                      style={{ width: `${Math.min(100, day.rainfall_mm * 2)}%` }}
                    />
                  </div>
                  <span className="text-xs text-blue-500 font-medium">{day.rainfall_mm}{t('weather.rain_unit', 'mm 🌧️')}</span>
                </motion.div>
              ))}
            </div>
          </AnimatedSection>

          {/* Farming Advisories */}
          <AnimatedSection custom={2} className="mb-6">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-lg font-poppins font-semibold text-[#1B2B1D]">{t('weather.advisories_title', 'Farming Advisories')}</h2>
              <Warning size={20} weight="fill" className="text-[#EF6C00]" />
            </div>
            <div className="space-y-3">
              {forecast
                .filter((d) => d.farming_advisory !== 'Conditions look favorable for standard farming activities.')
                .slice(0, 3)
                .map((day, idx) => (
                  <div
                    key={idx}
                    className="bg-white border-l-4 border-[#2E7D32] rounded-r-xl p-4 shadow-sm flex items-start space-x-3 hover:shadow-md transition-shadow"
                  >
                    <div className="text-2xl flex-shrink-0">{getWeatherEmoji(day.rainfall_mm, day.max_temp)}</div>
                    <div>
                      <p className="text-xs font-noto text-text-secondary mb-0.5">
                        {getDayLabel(day.date, forecast.indexOf(day), t, currentLocale)}
                      </p>
                      <p className="text-sm text-[#1B2B1D] leading-relaxed">
                        {day.farming_advisory === 'Conditions look favorable for standard farming activities.'
                          ? t('weather.advisories.favorable', day.farming_advisory)
                          : day.farming_advisory === 'Heavy rain expected — avoid spraying pesticides and monitor drainage.'
                          ? t('weather.advisories.heavy_rain', day.farming_advisory)
                          : day.farming_advisory === 'High temperatures expected. Ensure adequate crop irrigation.'
                          ? t('weather.advisories.high_temp', day.farming_advisory)
                          : day.farming_advisory}
                      </p>
                    </div>
                  </div>
                ))}
              {forecast.every((d) => d.farming_advisory === 'Conditions look favorable for standard farming activities.') && (
                <div className="bg-white border-l-4 border-[#2E7D32] rounded-r-xl p-4 shadow-sm flex items-start space-x-3">
                  <div className="text-2xl">✅</div>
                  <p className="text-sm text-[#1B2B1D] leading-relaxed">
                    {t('weather.advisories.all_clear', 'All 7 days: Good conditions for farming activities. Stay updated for any changes.')}
                  </p>
                </div>
              )}
            </div>
          </AnimatedSection>

          {/* Precipitation Chart */}
          <AnimatedSection custom={3} className="mb-4">
            <h2 className="text-lg font-poppins font-semibold text-[#1B2B1D] mb-3">{t('weather.precipitation_title', 'Precipitation (mm)')}</h2>
            <div className="bg-white p-4 rounded-xl shadow-sm h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={forecast.map((d, i) => ({
                    day: getDayLabel(d.date, i, t, currentLocale),
                    rain: d.rainfall_mm,
                  }))}
                  margin={{ top: 10, right: 0, left: -20, bottom: 0 }}
                >
                  <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#546E7A' }} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#546E7A' }} />
                  <Tooltip
                    cursor={{ fill: '#F1F8E9' }}
                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}
                    formatter={((val: any) => [`${Number(val).toFixed(0)} mm`, t('weather.rainfall', 'Rainfall')]) as any}
                  />
                  <Bar dataKey="rain" radius={[4, 4, 0, 0]}>
                    {forecast.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.rainfall_mm > 20 ? '#3B82F6' : '#93C5FD'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </AnimatedSection>
        </>
      )}
    </PageWrapper>
  );
}
