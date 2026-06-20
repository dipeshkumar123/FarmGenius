/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#2E7D32',
          light: '#60AD5E',
          dark: '#1B5E20',
        },
        secondary: {
          DEFAULT: '#558B2F',
        },
        accent: {
          DEFAULT: '#F9A825',
          dark: '#F57F17',
        },
        bg: {
          DEFAULT: '#F1F8E9',
        },
        surface: {
          DEFAULT: '#FFFFFF',
          variant: '#E8F5E9',
        },
        text: {
          primary: '#1B2B1D',
          secondary: '#546E7A',
          on_primary: '#FFFFFF',
        },
        farm: {
          error: '#C62828',
          warning: '#EF6C00',
          success: '#388E3C',
          divider: '#C8E6C9',
          info: '#1565C0',
        },
      },
      fontFamily: {
        poppins: ['Poppins', 'sans-serif'],
        noto: ['Noto Sans', 'Noto Sans Devanagari', 'sans-serif'],
      },
      borderRadius: {
        sm: '8px',
        md: '16px',
        lg: '24px',
        full: '100px',
      },
      boxShadow: {
        card: '0 4px 12px 0 rgba(46, 125, 50, 0.08)',
        'card-hover': '0 8px 24px 0 rgba(46, 125, 50, 0.16)',
        accent: '0 6px 16px 0 rgba(249, 168, 37, 0.3)',
        blue: '0 6px 16px 0 rgba(21, 101, 192, 0.3)',
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #388E3C 100%)',
        'gradient-accent': 'linear-gradient(135deg, #F57F17 0%, #F9A825 100%)',
        'gradient-sky': 'linear-gradient(135deg, #1565C0 0%, #42A5F5 100%)',
        'gradient-earth': 'linear-gradient(135deg, #558B2F 0%, #8BC34A 100%)',
        'gradient-card': 'linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(241,248,233,0.9) 100%)',
      },
      animation: {
        'fade-in': 'fadeIn 0.4s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'slide-in-right': 'slideInRight 0.4s ease-out',
        'bounce-slow': 'bounce 2s infinite',
        'pulse-green': 'pulseGreen 2s ease-in-out infinite',
        'float': 'float 3s ease-in-out infinite',
        'shimmer': 'shimmer 1.5s linear infinite',
        'wave': 'wave 1.2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideInRight: {
          '0%': { opacity: '0', transform: 'translateX(20px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        pulseGreen: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(46, 125, 50, 0.4)' },
          '50%': { boxShadow: '0 0 0 12px rgba(46, 125, 50, 0)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        wave: {
          '0%, 100%': { transform: 'scaleY(0.5)' },
          '50%': { transform: 'scaleY(1.5)' },
        },
      },
    },
  },
  plugins: [],
}
