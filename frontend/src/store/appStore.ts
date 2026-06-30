// src/store/appStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import i18n from '../i18n';

interface FarmerProfile {
  name: string;
  phone: string;
  district: string;
  state: string;
  language: string;
  crops: string[];
}

interface AppState {
  isAuthenticated: boolean;
  farmer: FarmerProfile | null;
  language: string;
  isOffline: boolean;
  setAuthenticated: (v: boolean) => void;
  setFarmer: (f: FarmerProfile) => void;
  setLanguage: (l: string) => void;
  setOffline: (v: boolean) => void;
  logout: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      isAuthenticated: false,
      farmer: null,
      language: 'en',
      isOffline: false,
      setAuthenticated: (v) => set({ isAuthenticated: v }),
      setFarmer: (f) => set({ farmer: f }),
      setLanguage: (l) => {
        i18n.changeLanguage(l);
        set({ language: l });
      },
      setOffline: (v) => set({ isOffline: v }),
      logout: () => {
        localStorage.removeItem('fg_token');
        set({ isAuthenticated: false, farmer: null });
      },
    }),
    { name: 'farmgenius-app' }
  )
);
