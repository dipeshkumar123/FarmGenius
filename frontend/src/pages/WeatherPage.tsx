import React from 'react';
import { PageWrapper, AnimatedSection } from '../components/ui/PageWrapper';
import { MapPin, Drop, Wind, Sun, Moon, Warning } from 'phosphor-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const forecast = [
  { day: 'Today', emoji: '⛅', high: 32, low: 22, rain: 10 },
  { day: 'Tue', emoji: '🌤️', high: 34, low: 23, rain: 5 },
  { day: 'Wed', emoji: '🌥️', high: 30, low: 21, rain: 20 },
  { day: 'Thu', emoji: '🌧️', high: 27, low: 20, rain: 75 },
  { day: 'Fri', emoji: '🌧️', high: 26, low: 19, rain: 80 },
  { day: 'Sat', emoji: '🌤️', high: 31, low: 21, rain: 15 },
  { day: 'Sun', emoji: '☀️', high: 35, low: 24, rain: 5 },
];

const advisories = [
  { icon: '🌧️', text: 'Heavy rain Thu-Fri: Avoid spraying pesticides on Wednesday. Harvest any ripe produce before Thursday.' },
  { icon: '☀️', text: 'High temperatures weekend: Ensure irrigation for standing crops on Saturday morning.' },
  { icon: '💨', text: 'Low wind conditions Mon-Wed: Ideal for drone spraying if available.' },
];

export default function WeatherPage() {
  return (
    <PageWrapper className="p-4 pb-24 bg-[#F1F8E9]">
      {/* Hero Card */}
      <AnimatedSection custom={0} className="mb-6">
        <div className="bg-gradient-to-br from-blue-500 to-blue-700 rounded-2xl p-5 text-white shadow-lg">
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center space-x-2 bg-white/20 rounded-full px-3 py-1 cursor-pointer hover:bg-white/30 transition-colors">
              <MapPin weight="fill" size={16} />
              <span className="text-sm font-medium">Dharwad, Karnataka</span>
              <span className="text-xs text-white/80">[Change]</span>
            </div>
          </div>
          
          <div className="flex justify-between items-center mb-6">
            <div>
              <div className="text-5xl font-bold font-poppins mb-1">32°C</div>
              <div className="text-xl font-medium">Partly Cloudy</div>
            </div>
            <div className="text-6xl drop-shadow-md">⛅</div>
          </div>
          
          <div className="grid grid-cols-3 gap-y-3 gap-x-2 text-sm bg-white/10 rounded-xl p-3 backdrop-blur-sm">
            <div className="flex items-center space-x-1">
              <span className="opacity-80">Feels like</span>
              <span className="font-semibold">35°</span>
            </div>
            <div className="flex items-center space-x-1">
              <span className="opacity-80">UV Index</span>
              <span className="font-semibold">7 High</span>
            </div>
            <div className="flex items-center space-x-1">
              <Drop size={16} weight="fill" className="text-blue-200" />
              <span className="font-semibold">65%</span>
            </div>
            <div className="flex items-center space-x-1">
              <Wind size={16} weight="fill" className="text-blue-200" />
              <span className="font-semibold">14 km/h</span>
            </div>
            <div className="flex items-center space-x-1">
              <Sun size={16} weight="fill" className="text-yellow-200" />
              <span className="font-semibold">06:12</span>
            </div>
            <div className="flex items-center space-x-1">
              <Moon size={16} weight="fill" className="text-blue-200" />
              <span className="font-semibold">18:45</span>
            </div>
          </div>
        </div>
      </AnimatedSection>

      {/* 7-day forecast */}
      <AnimatedSection custom={1} className="mb-6">
        <h2 className="text-lg font-poppins font-semibold text-[#1B2B1D] mb-3">7-Day Forecast</h2>
        <div className="flex overflow-x-auto space-x-3 pb-2 -mx-4 px-4 snap-x hide-scrollbar">
          {forecast.map((day, idx) => (
            <div key={idx} className="flex-shrink-0 w-24 bg-white rounded-xl p-3 shadow-sm flex flex-col items-center snap-center hover:shadow-md transition-shadow">
              <span className="text-sm font-medium text-[#546E7A] mb-1">{day.day}</span>
              <span className="text-2xl mb-2">{day.emoji}</span>
              <div className="flex space-x-2 text-sm mb-3">
                <span className="font-semibold text-[#1B2B1D]">{day.high}°</span>
                <span className="text-[#546E7A]">{day.low}°</span>
              </div>
              <div className="w-full bg-[#E8F5E9] rounded-full h-1.5 mb-1 overflow-hidden">
                <div className="bg-blue-400 h-full rounded-full" style={{ width: `${day.rain}%` }}></div>
              </div>
              <span className="text-xs text-blue-500 font-medium">{day.rain}% 🌧️</span>
            </div>
          ))}
        </div>
      </AnimatedSection>

      {/* Farming Advisories */}
      <AnimatedSection custom={2} className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-poppins font-semibold text-[#1B2B1D]">Farming Advisories</h2>
          <Warning size={20} weight="fill" className="text-[#EF6C00]" />
        </div>
        <div className="space-y-3">
          {advisories.map((adv, idx) => (
            <div key={idx} className="bg-white border-l-4 border-[#2E7D32] rounded-r-xl p-4 shadow-sm flex items-start space-x-3 hover:shadow-md transition-shadow">
              <div className="text-2xl flex-shrink-0">{adv.icon}</div>
              <p className="text-sm text-[#1B2B1D] leading-relaxed">{adv.text}</p>
            </div>
          ))}
        </div>
      </AnimatedSection>

      {/* Precipitation Chart */}
      <AnimatedSection custom={3} className="mb-4">
        <h2 className="text-lg font-poppins font-semibold text-[#1B2B1D] mb-3">Precipitation (mm)</h2>
        <div className="bg-white p-4 rounded-xl shadow-sm h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={forecast} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
              <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#546E7A' }} />
              <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#546E7A' }} />
              <Tooltip cursor={{ fill: '#F1F8E9' }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }} />
              <Bar dataKey="rain" radius={[4, 4, 0, 0]}>
                {forecast.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.rain > 50 ? '#3B82F6' : '#93C5FD'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </AnimatedSection>
    </PageWrapper>
  );
}
