import React, { useState } from 'react';
import { PageWrapper, AnimatedSection } from '../components/ui/PageWrapper';
import { User, MapPin, PencilSimple, Translate, Bell, WifiHigh, Moon, Question, PhoneCall, Star, ShareNetwork, SignOut, CaretRight } from 'phosphor-react';

export default function ProfilePage() {
  const [notifications, setNotifications] = useState(true);
  const [offlineMode, setOfflineMode] = useState(true);
  const [darkMode, setDarkMode] = useState(false);

  return (
    <PageWrapper className="p-4 pb-24 bg-[#F1F8E9]">
      {/* Header Profile Section */}
      <AnimatedSection custom={0} className="bg-white rounded-2xl p-5 shadow-sm mb-4 flex items-center space-x-4 border border-[#E8F5E9]">
        <div className="w-16 h-16 rounded-full bg-gradient-to-br from-[#2E7D32] to-[#4CAF50] flex items-center justify-center text-white text-2xl font-bold font-poppins shadow-md">
          RK
        </div>
        <div className="flex-1">
          <h1 className="text-xl font-bold text-[#1B2B1D] font-poppins">Ramesh Kumar</h1>
          <p className="text-[#546E7A] text-sm">+91 98765 43210</p>
          <div className="flex items-center text-[#546E7A] text-sm mt-1">
            <MapPin size={14} className="mr-1 text-[#2E7D32]" /> Dharwad, Karnataka
          </div>
        </div>
        <button className="p-2 bg-[#F1F8E9] text-[#2E7D32] rounded-full hover:bg-[#E8F5E9] transition-colors">
          <PencilSimple size={20} weight="fill" />
        </button>
      </AnimatedSection>

      {/* My Farm Section */}
      <AnimatedSection custom={1} className="bg-white rounded-2xl p-4 shadow-sm mb-4 border border-[#E8F5E9]">
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-base font-semibold text-[#1B2B1D] font-poppins flex items-center">
            <span className="text-xl mr-2">🚜</span> My Farm
          </h2>
          <button className="text-sm text-[#2E7D32] font-medium">Edit</button>
        </div>
        <div className="flex flex-wrap gap-2">
          {['Paddy (🌾)', 'Cotton (🧶)', 'Tomato (🍅)'].map((crop, idx) => (
            <span key={idx} className="bg-[#E8F5E9] text-[#2E7D32] px-3 py-1.5 rounded-full text-sm font-medium border border-[#C8E6C9]">
              {crop}
            </span>
          ))}
          <button className="bg-gray-50 text-[#546E7A] px-3 py-1.5 rounded-full text-sm font-medium border border-dashed border-gray-300 hover:bg-gray-100 flex items-center">
            + Add Crop
          </button>
        </div>
      </AnimatedSection>

      {/* Settings Section */}
      <AnimatedSection custom={2} className="bg-white rounded-2xl p-2 shadow-sm mb-4 border border-[#E8F5E9]">
        <div className="p-3 flex items-center justify-between border-b border-gray-50 cursor-pointer hover:bg-gray-50 rounded-t-xl transition-colors">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-full bg-blue-50 flex items-center justify-center text-blue-500"><Translate size={18} weight="fill" /></div>
            <span className="text-[#1B2B1D] font-medium text-sm">Language</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-[#546E7A] text-sm">English</span>
            <CaretRight size={16} className="text-gray-400" />
          </div>
        </div>

        <div className="p-3 flex items-center justify-between border-b border-gray-50">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-full bg-orange-50 flex items-center justify-center text-orange-500"><Bell size={18} weight="fill" /></div>
            <span className="text-[#1B2B1D] font-medium text-sm">Notifications</span>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input type="checkbox" className="sr-only peer" checked={notifications} onChange={() => setNotifications(!notifications)} />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[#2E7D32]"></div>
          </label>
        </div>

        <div className="p-3 flex items-center justify-between border-b border-gray-50">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-full bg-green-50 flex items-center justify-center text-[#2E7D32]"><WifiHigh size={18} weight="fill" /></div>
            <div>
              <span className="text-[#1B2B1D] font-medium text-sm block">Offline Mode</span>
              <span className="text-xs text-[#546E7A]">Save disease models for offline use</span>
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input type="checkbox" className="sr-only peer" checked={offlineMode} onChange={() => setOfflineMode(!offlineMode)} />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[#2E7D32]"></div>
          </label>
        </div>

        <div className="p-3 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-slate-700"><Moon size={18} weight="fill" /></div>
            <span className="text-[#1B2B1D] font-medium text-sm">Dark Mode</span>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input type="checkbox" className="sr-only peer" checked={darkMode} onChange={() => setDarkMode(!darkMode)} />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[#2E7D32]"></div>
          </label>
        </div>
      </AnimatedSection>

      {/* Support & Actions Section */}
      <AnimatedSection custom={3} className="bg-white rounded-2xl p-2 shadow-sm mb-6 border border-[#E8F5E9]">
        <button className="w-full p-3 flex items-center justify-between border-b border-gray-50 hover:bg-gray-50 rounded-t-xl transition-colors">
          <div className="flex items-center space-x-3">
            <Question size={20} className="text-[#546E7A]" weight="regular" />
            <span className="text-[#1B2B1D] font-medium text-sm">Help & FAQ</span>
          </div>
          <CaretRight size={16} className="text-gray-400" />
        </button>

        <button className="w-full p-3 flex items-center justify-between border-b border-gray-50 hover:bg-gray-50 transition-colors">
          <div className="flex items-center space-x-3">
            <PhoneCall size={20} className="text-[#2E7D32]" weight="fill" />
            <div className="text-left">
              <span className="text-[#1B2B1D] font-medium text-sm block">Kisan Helpline</span>
              <span className="text-xs text-[#546E7A]">1800-180-1551 (Toll Free)</span>
            </div>
          </div>
        </button>

        <button className="w-full p-3 flex items-center justify-between border-b border-gray-50 hover:bg-gray-50 transition-colors">
          <div className="flex items-center space-x-3">
            <Star size={20} className="text-[#F9A825]" weight="fill" />
            <span className="text-[#1B2B1D] font-medium text-sm">Rate App</span>
          </div>
        </button>

        <button className="w-full p-3 flex items-center justify-between hover:bg-gray-50 rounded-b-xl transition-colors">
          <div className="flex items-center space-x-3">
            <ShareNetwork size={20} className="text-blue-500" weight="bold" />
            <span className="text-[#1B2B1D] font-medium text-sm">Share with a Farmer</span>
          </div>
        </button>
      </AnimatedSection>

      {/* Sign Out Button */}
      <AnimatedSection custom={4}>
        <button className="w-full py-3.5 border-2 border-red-500 text-red-500 rounded-xl font-semibold flex items-center justify-center space-x-2 hover:bg-red-50 active:scale-[0.98] transition-all">
          <SignOut weight="bold" size={20} />
          <span>Sign Out</span>
        </button>
        <p className="text-center text-xs text-[#546E7A] mt-4 font-medium">FarmGenius v1.0.0 (Beta)</p>
      </AnimatedSection>
    </PageWrapper>
  );
}
