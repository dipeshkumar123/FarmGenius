// src/pages/LoginPage.tsx
import { useState, useRef, useEffect } from 'react';
import type { KeyboardEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Leaf as Plant,
  Phone,
  ArrowRight,
  ShieldCheck,
  CaretLeft,
  Spinner,
  DeviceMobile,
  CheckCircle,
  CloudSun,
  ChartLine,
  Robot,
} from 'phosphor-react';
import toast from 'react-hot-toast';
import { useAppStore } from '../store/appStore';
import apiClient from '../api/client';

// ─── Types ────────────────────────────────────────────────────────────────────

type AuthStep = 'phone' | 'otp' | 'success';

interface OtpInputProps {
  value: string[];
  onChange: (index: number, value: string) => void;
  onKeyDown: (index: number, e: KeyboardEvent<HTMLInputElement>) => void;
  inputRefs: React.RefObject<HTMLInputElement | null>[];
  hasError: boolean;
}

// ─── Mock Demo Farmer Data ────────────────────────────────────────────────────

const DEMO_FARMER = {
  name: 'Demo Farmer',
  phone: '9999999999',
  district: 'Dharwad',
  state: 'Karnataka',
  language: 'en',
  crops: ['Wheat', 'Tomato', 'Onion'],
};

// ─── Animation Variants ────────────────────────────────────────────────────────

const slideVariants = {
  enterFromRight: {
    x: 40,
    opacity: 0,
  },
  center: {
    x: 0,
    opacity: 1,
    transition: { duration: 0.4 },
  },
  exitToLeft: {
    x: -40,
    opacity: 0,
    transition: { duration: 0.3 },
  },
};

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (delay: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, delay },
  }),
};

// ─── OTP Input Component ──────────────────────────────────────────────────────

function OtpInput({ value, onChange, onKeyDown, inputRefs, hasError }: OtpInputProps) {
  return (
    <div className="flex gap-3 justify-center">
      {Array.from({ length: 6 }).map((_, i) => (
        <motion.input
          key={i}
          ref={inputRefs[i]}
          type="text"
          inputMode="numeric"
          maxLength={1}
          value={value[i] || ''}
          onChange={(e) => {
            const val = e.target.value.replace(/\D/g, '').slice(-1);
            onChange(i, val);
          }}
          onKeyDown={(e) => onKeyDown(i, e)}
          onFocus={(e) => e.target.select()}
          initial={{ scale: 0.85, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: i * 0.06, type: 'spring', stiffness: 300, damping: 20 }}
          className={`w-11 h-14 sm:w-13 sm:h-16 text-center text-xl font-poppins font-bold rounded-xl border-2 transition-all duration-200 outline-none focus:ring-2 focus:ring-primary/30 bg-surface
            ${value[i]
              ? hasError
                ? 'border-farm-error text-farm-error bg-red-50'
                : 'border-primary text-primary bg-surface-variant'
              : 'border-farm-divider text-text-primary'
            }
            ${hasError ? 'animate-shake' : ''}
          `}
          style={{ width: '3rem', height: '3.5rem' }}
        />
      ))}
    </div>
  );
}

// ─── Left Illustration Panel ──────────────────────────────────────────────────

function LeftPanel() {
  const features = [
    { icon: <Robot size={18} weight="fill" />, text: 'AI Crop Advisor in your language' },
    { icon: <CloudSun size={18} weight="fill" />, text: '7-day farming weather forecast' },
    { icon: <ChartLine size={18} weight="fill" />, text: 'Live mandi prices from 500+ APMCs' },
  ];

  return (
    <div
      className="hidden lg:flex flex-col justify-between h-full p-10 relative overflow-hidden"
      style={{
        background: 'linear-gradient(160deg, #1B5E20 0%, #2E7D32 45%, #388E3C 100%)',
      }}
    >
      {/* Blobs */}
      <motion.div
        className="absolute w-72 h-72 rounded-full bg-white/5 blur-3xl -top-20 -right-20 pointer-events-none"
        animate={{ scale: [1, 1.15, 1] }}
        transition={{ duration: 7, repeat: Infinity }}
      />
      <motion.div
        className="absolute w-56 h-56 rounded-full bg-accent/10 blur-3xl bottom-10 -left-10 pointer-events-none"
        animate={{ scale: [1, 1.2, 1] }}
        transition={{ duration: 9, repeat: Infinity, delay: 2 }}
      />

      {/* Grid overlay */}
      <div
        className="absolute inset-0 opacity-5 pointer-events-none"
        style={{
          backgroundImage: 'radial-gradient(circle at 1px 1px, white 1px, transparent 0)',
          backgroundSize: '32px 32px',
        }}
      />

      {/* Top: Logo */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="flex items-center gap-3 relative z-10"
      >
        <div className="w-11 h-11 rounded-full bg-accent flex items-center justify-center shadow-accent">
          <Plant size={24} weight="fill" className="text-white" />
        </div>
        <div>
          <span className="font-poppins font-bold text-white text-xl block leading-none">FarmGenius</span>
          <span className="font-noto text-white/60 text-xs">Smart Farming Assistant</span>
        </div>
      </motion.div>

      {/* Middle: Illustration card */}
      <div className="relative z-10 flex-1 flex flex-col justify-center">
        <motion.h2
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}
          className="font-poppins font-bold text-white text-3xl xl:text-4xl leading-tight mb-4"
        >
          Your Smart Farm<br />Starts Here 🌾
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.35 }}
          className="font-noto text-white/70 text-base leading-relaxed mb-8 max-w-xs"
        >
          Join 50,000+ Indian farmers who are growing more with less — powered by AI.
        </motion.p>

        {/* Feature list */}
        <div className="flex flex-col gap-4">
          {features.map((f, i) => (
            <motion.div
              key={f.text}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.5 + i * 0.1 }}
              className="flex items-center gap-3"
            >
              <div className="w-9 h-9 rounded-full bg-white/15 backdrop-blur-sm flex items-center justify-center shrink-0 text-accent">
                {f.icon}
              </div>
              <span className="font-noto text-white/80 text-sm">{f.text}</span>
            </motion.div>
          ))}
        </div>

        {/* Mock mini dashboard */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.8 }}
          className="mt-8 bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl p-4"
        >
          <div className="flex items-center gap-2 mb-3">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="font-noto text-white/60 text-xs">Live Dashboard Preview</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {[
              { label: 'Wheat Price', value: '₹2,180', change: '↑ 3%', color: 'text-green-300' },
              { label: 'Rain Today', value: '12mm', change: '⚠ Plan', color: 'text-yellow-300' },
              { label: 'Crop Health', value: '87%', change: '✓ Good', color: 'text-green-300' },
            ].map((stat) => (
              <div key={stat.label} className="bg-white/10 rounded-xl p-2 text-center">
                <p className={`font-poppins font-bold text-sm ${stat.color}`}>{stat.value}</p>
                <p className="font-noto text-white/50 text-xs mt-0.5">{stat.label}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Bottom: Testimonial pill */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 1 }}
        className="relative z-10 bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl p-4 flex items-start gap-3"
      >
        <div className="w-10 h-10 rounded-full bg-accent/80 shrink-0 flex items-center justify-center font-poppins font-bold text-white text-sm">
          RS
        </div>
        <div>
          <p className="font-noto text-white/80 text-sm leading-relaxed">
            "FarmGenius ने मेरी गेहूं की फसल बचाई! <span className="text-white font-medium">Amazing app.</span>"
          </p>
          <p className="font-noto text-white/50 text-xs mt-1">Ramesh Singh, Bareilly UP · ★★★★★</p>
        </div>
      </motion.div>
    </div>
  );
}

// ─── Main Login Page ──────────────────────────────────────────────────────────

export default function LoginPage() {
  const navigate = useNavigate();
  const { setAuthenticated, setFarmer } = useAppStore();

  const [step, setStep] = useState<AuthStep>('phone');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [otp, setOtp] = useState<string[]>(Array(6).fill(''));
  const [otpError, setOtpError] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [agreedToTerms, setAgreedToTerms] = useState(false);

  // OTP input refs for auto-advance (6 individual refs — hooks must not be in loops)
  const otpRef0 = useRef<HTMLInputElement>(null);
  const otpRef1 = useRef<HTMLInputElement>(null);
  const otpRef2 = useRef<HTMLInputElement>(null);
  const otpRef3 = useRef<HTMLInputElement>(null);
  const otpRef4 = useRef<HTMLInputElement>(null);
  const otpRef5 = useRef<HTMLInputElement>(null);
  const otpRefs = [otpRef0, otpRef1, otpRef2, otpRef3, otpRef4, otpRef5];

  // Countdown timer for resend OTP
  useEffect(() => {
    if (countdown <= 0) return;
    const timer = setTimeout(() => setCountdown((c) => c - 1), 1000);
    return () => clearTimeout(timer);
  }, [countdown]);

  // Phone validation
  const isPhoneValid = /^[6-9]\d{9}$/.test(phoneNumber.trim());

  // Handle phone submit
  const handleSendOtp = async () => {
    if (!isPhoneValid) {
      toast.error('Please enter a valid 10-digit Indian mobile number');
      return;
    }
    if (!agreedToTerms) {
      toast.error('Please agree to the Terms & Privacy Policy');
      return;
    }
    setIsLoading(true);
    try {
      await apiClient.post('/auth/send-otp', { phone: phoneNumber });
      setIsLoading(false);
      setStep('otp');
      setCountdown(30);
      toast.success(`OTP sent to +91 ${phoneNumber}!`);
      // Focus first OTP box after transition
      setTimeout(() => otpRefs[0].current?.focus(), 400);
    } catch (err) {
      setIsLoading(false);
      toast.error('Failed to send OTP. Please try again later.');
    }
  };

  // Handle OTP input change with auto-advance
  const handleOtpChange = (index: number, val: string) => {
    const newOtp = [...otp];
    newOtp[index] = val;
    setOtp(newOtp);
    setOtpError(false);

    if (val && index < 5) {
      otpRefs[index + 1].current?.focus();
    }

    // Auto-verify when all 6 digits filled
    if (val && index === 5) {
      const fullOtp = [...newOtp.slice(0, 5), val].join('');
      if (fullOtp.length === 6) {
        handleVerifyOtp([...newOtp.slice(0, 5), val]);
      }
    }
  };

  // Handle OTP keyboard navigation
  const handleOtpKeyDown = (index: number, e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Backspace') {
      if (!otp[index] && index > 0) {
        otpRefs[index - 1].current?.focus();
        const newOtp = [...otp];
        newOtp[index - 1] = '';
        setOtp(newOtp);
      }
    } else if (e.key === 'ArrowLeft' && index > 0) {
      otpRefs[index - 1].current?.focus();
    } else if (e.key === 'ArrowRight' && index < 5) {
      otpRefs[index + 1].current?.focus();
    }
  };

  // Handle OTP paste
  const handleOtpPaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const pasted = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, 6);
    if (pasted.length === 6) {
      const newOtp = pasted.split('');
      setOtp(newOtp);
      otpRefs[5].current?.focus();
      handleVerifyOtp(newOtp);
    }
  };

  // Verify OTP (hitting backend)
  const handleVerifyOtp = async (otpValue: string[]) => {
    const code = otpValue.join('');
    if (code.length < 6) return;

    setIsLoading(true);
    
    try {
      const res = await apiClient.post('/auth/verify-otp', {
        phone: phoneNumber,
        otp: code
      });
      
      const { token, farmer_id } = res.data;
      localStorage.setItem('fg_token', token);
      
      // Success
      setFarmer({ ...DEMO_FARMER, phone: farmer_id });
      setAuthenticated(true);
      setIsLoading(false);
      setStep('success');

      setTimeout(() => {
        navigate('/dashboard');
      }, 1800);
      
    } catch (err: any) {
      setIsLoading(false);
      setOtpError(true);
      toast.error('Incorrect OTP. Please try again.');
      setOtp(Array(6).fill(''));
      otpRefs[0].current?.focus();
    }
  };

  // Handle manual verify button
  const handleVerifyButtonClick = () => {
    handleVerifyOtp(otp);
  };

  // Handle resend OTP
  const handleResendOtp = async () => {
    if (countdown > 0) return;
    setOtp(Array(6).fill(''));
    setOtpError(false);
    setIsLoading(true);
    try {
      await apiClient.post('/auth/send-otp', { phone: phoneNumber });
      setIsLoading(false);
      setCountdown(30);
      toast.success('OTP resent!');
      setTimeout(() => otpRefs[0].current?.focus(), 300);
    } catch (err) {
      setIsLoading(false);
      toast.error('Failed to resend OTP. Please try again later.');
    }
  };

  // Demo login
  const handleDemoLogin = async () => {
    setIsLoading(true);
    try {
      const res = await apiClient.post('/auth/verify-otp', {
        phone: '9999999999',
        otp: '123456'
      });
      localStorage.setItem('fg_token', res.data.token);
      setFarmer(DEMO_FARMER);
      setAuthenticated(true);
      setIsLoading(false);
      toast.success('Welcome to FarmGenius Demo! 🌱');
      navigate('/dashboard');
    } catch (err) {
      setIsLoading(false);
      toast.error('Could not connect to backend.');
    }
  };

  // Google login (mock)
  const handleGoogleLogin = async () => {
    setIsLoading(true);
    await new Promise((r) => setTimeout(r, 1200));
    setIsLoading(false);
    toast('Google login coming soon!', { icon: '🚧' });
  };

  const otpComplete = otp.every(Boolean);

  return (
    <div className="min-h-screen flex bg-bg">

      {/* ── Left illustration panel (desktop only) ─────────────────────────── */}
      <div className="lg:w-[48%] xl:w-[45%] lg:min-h-screen shrink-0">
        <LeftPanel />
      </div>

      {/* ── Right: Auth Form ─────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col items-center justify-center p-4 sm:p-8 min-h-screen">

        {/* Mobile-only logo */}
        <motion.div
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="lg:hidden flex items-center gap-3 mb-8"
        >
          <div className="w-11 h-11 rounded-full bg-gradient-primary flex items-center justify-center shadow-card">
            <Plant size={24} weight="fill" className="text-white" />
          </div>
          <span className="font-poppins font-bold text-text-primary text-xl">FarmGenius</span>
        </motion.div>

        {/* Card */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="w-full max-w-md bg-white rounded-2xl shadow-card-hover border border-farm-divider p-8"
        >
          <AnimatePresence mode="wait">

            {/* ── PHONE STEP ─────────────────────────────────────────────── */}
            {step === 'phone' && (
              <motion.div
                key="phone"
                variants={slideVariants}
                initial="enterFromRight"
                animate="center"
                exit="exitToLeft"
                className="flex flex-col gap-6"
              >
                {/* Header */}
                <div>
                  <motion.h1
                    variants={fadeUp}
                    initial="hidden"
                    animate="visible"
                    custom={0}
                    className="font-poppins font-bold text-text-primary text-2xl sm:text-3xl"
                  >
                    Welcome back, Farmer! 👋
                  </motion.h1>
                  <motion.p
                    variants={fadeUp}
                    initial="hidden"
                    animate="visible"
                    custom={0.08}
                    className="font-noto text-text-secondary text-base mt-1"
                  >
                    स्वागत है, किसान!
                  </motion.p>
                  <motion.p
                    variants={fadeUp}
                    initial="hidden"
                    animate="visible"
                    custom={0.16}
                    className="font-noto text-text-secondary text-sm mt-3 leading-relaxed"
                  >
                    Enter your mobile number to get started. We'll send you a quick OTP.
                  </motion.p>
                </div>

                {/* Phone input */}
                <motion.div variants={fadeUp} initial="hidden" animate="visible" custom={0.24}>
                  <label className="font-poppins font-semibold text-text-primary text-sm block mb-2">
                    Mobile Number
                  </label>
                  <div
                    className={`flex items-center border-2 rounded-xl overflow-hidden transition-colors duration-200 ${
                      phoneNumber && !isPhoneValid
                        ? 'border-farm-error'
                        : phoneNumber && isPhoneValid
                        ? 'border-primary'
                        : 'border-farm-divider focus-within:border-primary'
                    }`}
                  >
                    {/* Country code */}
                    <div className="flex items-center gap-2 bg-surface-variant px-4 py-4 border-r-2 border-farm-divider shrink-0">
                      <span className="text-lg">🇮🇳</span>
                      <span className="font-poppins font-semibold text-text-primary text-sm">+91</span>
                    </div>
                    {/* Input */}
                    <input
                      type="tel"
                      inputMode="numeric"
                      placeholder="9876543210"
                      value={phoneNumber}
                      onChange={(e) => setPhoneNumber(e.target.value.replace(/\D/g, '').slice(0, 10))}
                      onKeyDown={(e) => e.key === 'Enter' && handleSendOtp()}
                      className="flex-1 px-4 py-4 font-poppins text-text-primary text-lg outline-none bg-transparent placeholder:text-text-secondary/40 min-h-[56px]"
                    />
                    {/* Indicator */}
                    {phoneNumber && (
                      <div className="pr-4">
                        {isPhoneValid ? (
                          <CheckCircle size={22} weight="fill" className="text-primary" />
                        ) : (
                          <Phone size={22} className="text-text-secondary/40" />
                        )}
                      </div>
                    )}
                  </div>
                  {phoneNumber && !isPhoneValid && (
                    <p className="font-noto text-farm-error text-xs mt-1.5 ml-1">
                      Please enter a valid 10-digit mobile number
                    </p>
                  )}
                </motion.div>

                {/* Terms checkbox */}
                <motion.label
                  variants={fadeUp}
                  initial="hidden"
                  animate="visible"
                  custom={0.32}
                  className="flex items-start gap-3 cursor-pointer"
                >
                  <div
                    role="checkbox"
                    aria-checked={agreedToTerms}
                    aria-labelledby="terms-label"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === ' ' || e.key === 'Enter') {
                        e.preventDefault();
                        setAgreedToTerms(!agreedToTerms);
                      }
                    }}
                    onClick={() => setAgreedToTerms(!agreedToTerms)}
                    className={`w-5 h-5 rounded border-2 shrink-0 mt-0.5 flex items-center justify-center transition-all cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 ${
                      agreedToTerms
                        ? 'bg-primary border-primary'
                        : 'border-farm-divider bg-white'
                    }`}
                  >
                    {agreedToTerms && (
                      <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
                        <path
                          d="M1 4L3.5 6.5L9 1"
                          stroke="white"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    )}
                  </div>
                  <span id="terms-label" className="font-noto text-text-secondary text-sm leading-relaxed">
                    I agree to the{' '}
                    <a href="#" className="text-primary underline hover:text-primary-dark">
                      Terms of Service
                    </a>{' '}
                    and{' '}
                    <a href="#" className="text-primary underline hover:text-primary-dark">
                      Privacy Policy
                    </a>
                  </span>
                </motion.label>

                {/* Send OTP button */}
                <motion.button
                  variants={fadeUp}
                  initial="hidden"
                  animate="visible"
                  custom={0.4}
                  whileHover={isPhoneValid && agreedToTerms && !isLoading ? { scale: 1.02, boxShadow: '0 8px 24px rgba(46, 125, 50, 0.3)' } : {}}
                  whileTap={isPhoneValid && agreedToTerms && !isLoading ? { scale: 0.98 } : {}}
                  onClick={handleSendOtp}
                  disabled={!isPhoneValid || !agreedToTerms || isLoading}
                  className={`w-full flex items-center justify-center gap-3 py-4 rounded-xl font-poppins font-bold text-base min-h-[56px] transition-all duration-200 ${
                    isPhoneValid && agreedToTerms && !isLoading
                      ? 'bg-primary text-white hover:bg-primary-dark shadow-card cursor-pointer'
                      : 'bg-farm-divider text-text-secondary cursor-not-allowed'
                  }`}
                >
                  {isLoading ? (
                    <>
                      <Spinner size={20} className="animate-spin" />
                      Sending OTP...
                    </>
                  ) : (
                    <>
                      <DeviceMobile size={20} weight="fill" />
                      Send OTP
                      <ArrowRight size={18} weight="bold" />
                    </>
                  )}
                </motion.button>

                {/* Divider */}
                <motion.div
                  variants={fadeUp}
                  initial="hidden"
                  animate="visible"
                  custom={0.48}
                  className="flex items-center gap-4"
                >
                  <div className="flex-1 h-px bg-farm-divider" />
                  <span className="font-noto text-text-secondary text-sm whitespace-nowrap">
                    or continue with
                  </span>
                  <div className="flex-1 h-px bg-farm-divider" />
                </motion.div>

                {/* Google button */}
                <motion.button
                  variants={fadeUp}
                  initial="hidden"
                  animate="visible"
                  custom={0.56}
                  whileHover={{ scale: 1.02, borderColor: '#4285F4' }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleGoogleLogin}
                  disabled={isLoading}
                  className="w-full flex items-center justify-center gap-3 py-4 rounded-xl font-poppins font-semibold text-sm text-text-primary bg-white border-2 border-farm-divider hover:bg-surface-variant transition-all min-h-[52px] cursor-pointer"
                >
                  {/* Google icon */}
                  <svg width="20" height="20" viewBox="0 0 24 24">
                    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
                    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
                    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
                  </svg>
                  Continue with Google
                </motion.button>

                {/* Demo mode */}
                <motion.div
                  variants={fadeUp}
                  initial="hidden"
                  animate="visible"
                  custom={0.64}
                  className="text-center"
                >
                  <button
                    onClick={handleDemoLogin}
                    disabled={isLoading}
                    className="font-noto text-primary text-sm underline underline-offset-2 hover:text-primary-dark transition-colors cursor-pointer"
                  >
                    🌱 Try Demo (no sign-up needed)
                  </button>
                </motion.div>
              </motion.div>
            )}

            {/* ── OTP STEP ───────────────────────────────────────────────── */}
            {step === 'otp' && (
              <motion.div
                key="otp"
                variants={slideVariants}
                initial="enterFromRight"
                animate="center"
                exit="exitToLeft"
                className="flex flex-col gap-6"
              >
                {/* Back button + header */}
                <div>
                  <button
                    onClick={() => {
                      setStep('phone');
                      setOtp(Array(6).fill(''));
                      setOtpError(false);
                    }}
                    className="flex items-center gap-1.5 text-text-secondary hover:text-primary transition-colors mb-4 font-noto text-sm"
                  >
                    <CaretLeft size={16} weight="bold" />
                    Change number
                  </button>
                  <h1 className="font-poppins font-bold text-text-primary text-2xl sm:text-3xl">
                    Verify your number
                  </h1>
                  <p className="font-noto text-text-secondary text-sm mt-2">
                    We sent a 6-digit OTP to{' '}
                    <span className="font-poppins font-semibold text-text-primary">
                      +91 {phoneNumber}
                    </span>
                  </p>
                </div>

                {/* OTP boxes */}
                <div onPaste={handleOtpPaste}>
                  <OtpInput
                    value={otp}
                    onChange={handleOtpChange}
                    onKeyDown={handleOtpKeyDown}
                    inputRefs={otpRefs}
                    hasError={otpError}
                  />
                  {otpError && (
                    <motion.p
                      initial={{ opacity: 0, y: -6 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="font-noto text-farm-error text-sm text-center mt-3"
                    >
                      Incorrect OTP. Please check and try again.
                    </motion.p>
                  )}
                </div>

                {/* Info note */}
                <div className="bg-surface-variant border border-farm-divider rounded-xl px-4 py-3 flex items-start gap-3">
                  <ShieldCheck size={20} weight="fill" className="text-primary shrink-0 mt-0.5" />
                  <p className="font-noto text-text-secondary text-sm leading-relaxed">
                    Demo mode: enter{' '}
                    <span className="font-poppins font-bold text-primary">123456</span>{' '}
                    as the OTP to sign in. A real SMS OTP will be used in production.
                  </p>
                </div>

                {/* Verify button */}
                <motion.button
                  whileHover={otpComplete && !isLoading ? { scale: 1.02, boxShadow: '0 8px 24px rgba(46, 125, 50, 0.3)' } : {}}
                  whileTap={otpComplete && !isLoading ? { scale: 0.98 } : {}}
                  onClick={handleVerifyButtonClick}
                  disabled={!otpComplete || isLoading}
                  className={`w-full flex items-center justify-center gap-3 py-4 rounded-xl font-poppins font-bold text-base min-h-[56px] transition-all duration-200 ${
                    otpComplete && !isLoading
                      ? 'bg-primary text-white hover:bg-primary-dark shadow-card cursor-pointer'
                      : 'bg-farm-divider text-text-secondary cursor-not-allowed'
                  }`}
                >
                  {isLoading ? (
                    <>
                      <Spinner size={20} className="animate-spin" />
                      Verifying...
                    </>
                  ) : (
                    <>
                      <ShieldCheck size={20} weight="fill" />
                      Verify & Sign In
                    </>
                  )}
                </motion.button>

                {/* Resend OTP */}
                <div className="text-center">
                  {countdown > 0 ? (
                    <p className="font-noto text-text-secondary text-sm">
                      Resend OTP in{' '}
                      <span className="font-poppins font-semibold text-primary">{countdown}s</span>
                    </p>
                  ) : (
                    <button
                      onClick={handleResendOtp}
                      className="font-noto text-primary text-sm underline underline-offset-2 hover:text-primary-dark transition-colors cursor-pointer"
                    >
                      Resend OTP
                    </button>
                  )}
                </div>
              </motion.div>
            )}

            {/* ── SUCCESS STEP ────────────────────────────────────────────── */}
            {step === 'success' && (
              <motion.div
                key="success"
                variants={slideVariants}
                initial="enterFromRight"
                animate="center"
                exit="exitToLeft"
                className="flex flex-col items-center gap-6 py-8"
              >
                <motion.div
                  initial={{ scale: 0, rotate: -20 }}
                  animate={{ scale: 1, rotate: 0 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 18 }}
                  className="w-24 h-24 rounded-full bg-gradient-primary flex items-center justify-center shadow-card"
                >
                  <CheckCircle size={52} weight="fill" className="text-white" />
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3, duration: 0.5 }}
                  className="text-center"
                >
                  <h2 className="font-poppins font-bold text-text-primary text-2xl mb-2">
                    Welcome to FarmGenius! 🌱
                  </h2>
                  <p className="font-noto text-text-secondary text-base">
                    You're verified. Redirecting to your dashboard...
                  </p>
                </motion.div>

                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: '100%' }}
                  transition={{ duration: 1.8, delay: 0.2 }}
                  className="h-1.5 bg-gradient-primary rounded-full"
                  style={{ maxWidth: '300px' }}
                />

                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5 }}
                  className="flex items-center gap-2 text-text-secondary"
                >
                  <Spinner size={18} className="animate-spin text-primary" />
                  <span className="font-noto text-sm">Loading your farm data...</span>
                </motion.div>
              </motion.div>
            )}

          </AnimatePresence>
        </motion.div>

        {/* Bottom note */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="font-noto text-text-secondary text-xs text-center mt-6 max-w-sm"
        >
          🔒 Your data is secure and encrypted. FarmGenius will never share your information with third parties.
        </motion.p>
      </div>
    </div>
  );
}
