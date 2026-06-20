// src/pages/LandingPage.tsx
import { useRef, useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  motion,
  useInView,
  AnimatePresence,
  useScroll,
  useTransform,
} from 'framer-motion';
import {
  Robot,
  Scan,
  CloudSun,
  ChartLine,
  Leaf as Plant,
  Bank,
  ArrowRight,
  CheckCircle,
  Star,
  WhatsappLogo,
  TwitterLogo,
  InstagramLogo,
} from 'phosphor-react';

// ─── Types ────────────────────────────────────────────────────────────────────

interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  color: string;
  delay: number;
}

interface StatItemProps {
  value: string;
  label: string;
  suffix?: string;
}

interface TestimonialCardProps {
  initials: string;
  color: string;
  quote: string;
  translation: string;
  name: string;
  location: string;
  crop: string;
  rating: number;
  delay: number;
}

interface StepProps {
  number: number;
  title: string;
  description: string;
  delay: number;
  isLast?: boolean;
}

// ─── Animation Variants ────────────────────────────────────────────────────────

const fadeUp = {
  hidden: { opacity: 0, y: 40 },
  visible: (delay: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, delay },
  }),
};

const fadeIn = {
  hidden: { opacity: 0 },
  visible: (delay: number = 0) => ({
    opacity: 1,
    transition: { duration: 0.5, delay },
  }),
};

const scaleIn = {
  hidden: { opacity: 0, scale: 0.85 },
  visible: (delay: number = 0) => ({
    opacity: 1,
    scale: 1,
    transition: { duration: 0.5, delay },
  }),
};

// ─── Count-Up Hook ────────────────────────────────────────────────────────────

function useCountUp(target: number, duration: number = 2000, active: boolean = false) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!active) return;
    const start = performance.now();
    const step = (now: number) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setCount(Math.floor(eased * target));
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [active, target, duration]);

  return count;
}

// ─── Sub-components ────────────────────────────────────────────────────────────

function FloatingBlob({ className }: { className: string }) {
  return (
    <motion.div
      className={`absolute rounded-full blur-3xl pointer-events-none ${className}`}
      animate={{
        scale: [1, 1.15, 1],
        x: [0, 20, 0],
        y: [0, -15, 0],
      }}
      transition={{ duration: 8, repeat: Infinity }}
    />
  );
}

function FeatureCard({ icon, title, description, color, delay }: FeatureCardProps) {
  return (
    <motion.div
      variants={scaleIn}
      custom={delay}
      whileHover={{ y: -6, boxShadow: '0 12px 32px 0 rgba(46, 125, 50, 0.18)' }}
      className="bg-white rounded-md p-6 shadow-card flex flex-col gap-4 cursor-default group"
    >
      <div
        className="w-14 h-14 rounded-md flex items-center justify-center shrink-0 transition-transform duration-300 group-hover:scale-110"
        style={{ backgroundColor: color + '20' }}
      >
        <span style={{ color }}>{icon}</span>
      </div>
      <div>
        <h3 className="font-poppins font-semibold text-text-primary text-lg mb-1">
          {title}
        </h3>
        <p className="font-noto text-text-secondary text-sm leading-relaxed">
          {description}
        </p>
      </div>
    </motion.div>
  );
}

function StatItem({ value, label, suffix = '' }: StatItemProps) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true });
  const numericTarget = parseInt(value.replace(/\D/g, ''), 10);
  const count = useCountUp(numericTarget, 2200, isInView);

  return (
    <div ref={ref} className="flex flex-col items-center text-center px-4">
      <span className="font-poppins font-bold text-4xl md:text-5xl text-white">
        {count.toLocaleString()}
        {suffix}
      </span>
      <span className="font-noto text-green-200 text-sm mt-1">{label}</span>
    </div>
  );
}

function StepItem({ number, title, description, delay, isLast = false }: StepProps) {
  return (
    <div className="flex md:flex-col items-start md:items-center gap-4 md:gap-0 relative">
      <motion.div
        variants={scaleIn}
        custom={delay}
        className="flex flex-col md:flex-row items-center"
      >
        <div className="w-14 h-14 rounded-full bg-gradient-primary flex items-center justify-center shrink-0 shadow-card z-10">
          <span className="font-poppins font-bold text-white text-xl">{number}</span>
        </div>
        {!isLast && (
          <div className="hidden md:block h-0.5 w-32 lg:w-48 border-t-2 border-dashed border-primary-light/50 mx-2" />
        )}
        {!isLast && (
          <div className="md:hidden w-0.5 h-12 border-l-2 border-dashed border-primary-light/50 ml-6" />
        )}
      </motion.div>
      <motion.div variants={fadeUp} custom={delay + 0.1} className="md:mt-6 md:text-center max-w-xs">
        <h3 className="font-poppins font-semibold text-text-primary text-lg mb-2">
          {title}
        </h3>
        <p className="font-noto text-text-secondary text-sm leading-relaxed">
          {description}
        </p>
      </motion.div>
    </div>
  );
}

function TestimonialCard({ initials, color, quote, translation, name, location, crop, rating, delay }: TestimonialCardProps) {
  return (
    <motion.div
      variants={fadeUp}
      custom={delay}
      whileHover={{ y: -4 }}
      className="bg-white rounded-md p-6 shadow-card flex flex-col gap-4"
    >
      <div className="flex items-center gap-3">
        <div
          className="w-12 h-12 rounded-full flex items-center justify-center shrink-0 font-poppins font-bold text-white text-lg"
          style={{ backgroundColor: color }}
        >
          {initials}
        </div>
        <div>
          <p className="font-poppins font-semibold text-text-primary text-sm">{name}</p>
          <p className="font-noto text-text-secondary text-xs">{location} · {crop}</p>
        </div>
        <div className="ml-auto flex gap-0.5">
          {Array.from({ length: rating }).map((_, i) => (
            <Star key={i} size={14} weight="fill" className="text-accent" />
          ))}
        </div>
      </div>
      <blockquote>
        <p className="font-noto text-text-primary text-sm leading-relaxed italic">"{quote}"</p>
        {translation && (
          <p className="font-noto text-text-secondary text-xs mt-2 leading-relaxed">
            "{translation}"
          </p>
        )}
      </blockquote>
    </motion.div>
  );
}

// ─── Mock Dashboard Mockup Card ────────────────────────────────────────────────

function DashboardMockup() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30, rotateY: -8 }}
      animate={{ opacity: 1, y: 0, rotateY: 0 }}
      transition={{ duration: 0.9, delay: 0.4 }}
      className="relative perspective-1000"
      style={{ perspective: '1000px' }}
    >
      <motion.div
        animate={{ y: [0, -8, 0] }}
        transition={{ duration: 4, repeat: Infinity }}
        className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl p-4 shadow-2xl max-w-sm mx-auto"
      >
        {/* App top bar */}
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 rounded-full bg-accent flex items-center justify-center">
            <Plant size={16} weight="fill" className="text-white" />
          </div>
          <span className="font-poppins font-semibold text-white text-sm">FarmGenius</span>
          <div className="ml-auto flex gap-1.5">
            <div className="w-2 h-2 rounded-full bg-red-400" />
            <div className="w-2 h-2 rounded-full bg-yellow-400" />
            <div className="w-2 h-2 rounded-full bg-green-400" />
          </div>
        </div>

        {/* Weather strip */}
        <div className="bg-white/10 rounded-xl p-3 mb-3 flex items-center gap-3">
          <CloudSun size={28} weight="fill" className="text-accent shrink-0" />
          <div>
            <p className="font-poppins font-bold text-white text-lg">28°C</p>
            <p className="font-noto text-white/70 text-xs">Partly cloudy · Dharwad</p>
          </div>
          <div className="ml-auto text-right">
            <p className="font-noto text-white/70 text-xs">Rain in 3 days</p>
            <p className="font-poppins font-semibold text-accent text-xs">Plan irrigation</p>
          </div>
        </div>

        {/* Crop health row */}
        <div className="grid grid-cols-2 gap-2 mb-3">
          {[
            { crop: '🌾 Wheat', status: 'Healthy', color: 'text-green-300', bar: 85 },
            { crop: '🍅 Tomato', status: 'Check leaf', color: 'text-yellow-300', bar: 62 },
          ].map((item) => (
            <div key={item.crop} className="bg-white/10 rounded-xl p-3">
              <p className="font-noto text-white text-xs mb-1">{item.crop}</p>
              <p className={`font-poppins font-semibold text-xs ${item.color} mb-2`}>
                {item.status}
              </p>
              <div className="h-1 rounded-full bg-white/20">
                <div
                  className="h-1 rounded-full bg-accent"
                  style={{ width: `${item.bar}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Price card */}
        <div className="bg-white/10 rounded-xl p-3 mb-3">
          <div className="flex justify-between items-center mb-2">
            <span className="font-poppins text-white/80 text-xs font-medium">Mandi Prices</span>
            <span className="font-noto text-green-300 text-xs">↑ Today</span>
          </div>
          <div className="flex justify-between">
            {[
              { name: 'Wheat', price: '₹2,180' },
              { name: 'Tomato', price: '₹1,420' },
              { name: 'Onion', price: '₹890' },
            ].map((p) => (
              <div key={p.name} className="text-center">
                <p className="font-poppins font-bold text-white text-sm">{p.price}</p>
                <p className="font-noto text-white/60 text-xs">{p.name}</p>
              </div>
            ))}
          </div>
        </div>

        {/* AI Chat preview */}
        <div className="bg-white/10 rounded-xl p-3">
          <div className="flex gap-2 mb-2">
            <div className="w-6 h-6 rounded-full bg-primary shrink-0 flex items-center justify-center">
              <Robot size={12} weight="fill" className="text-white" />
            </div>
            <div className="bg-white/20 rounded-lg px-3 py-1.5 flex-1">
              <p className="font-noto text-white text-xs">मेरी wheat पीली हो रही है?</p>
            </div>
          </div>
          <div className="bg-accent/20 rounded-lg px-3 py-1.5 border border-accent/30">
            <p className="font-noto text-white text-xs">
              Nitrogen deficiency likely. Apply urea @ 50 kg/acre now. 🌱
            </p>
          </div>
        </div>
      </motion.div>

      {/* Floating badges */}
      <motion.div
        animate={{ x: [0, 5, 0], y: [0, -4, 0] }}
        transition={{ duration: 3, repeat: Infinity, delay: 0.5 }}
        className="absolute -right-4 top-8 bg-white rounded-xl px-3 py-2 shadow-accent flex items-center gap-2"
      >
        <Scan size={16} className="text-primary" />
        <span className="font-poppins font-semibold text-primary text-xs">Disease Detected!</span>
      </motion.div>

      <motion.div
        animate={{ x: [0, -4, 0], y: [0, 5, 0] }}
        transition={{ duration: 3.5, repeat: Infinity, delay: 1 }}
        className="absolute -left-4 bottom-12 bg-accent rounded-xl px-3 py-2 shadow-accent flex items-center gap-2"
      >
        <ChartLine size={16} className="text-white" />
        <span className="font-poppins font-semibold text-white text-xs">Price ↑ 12%</span>
      </motion.div>
    </motion.div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function LandingPage() {
  const navigate = useNavigate();
  const heroRef = useRef<HTMLDivElement>(null);
  const featuresRef = useRef<HTMLDivElement>(null);
  const statsRef = useRef<HTMLDivElement>(null);
  const howRef = useRef<HTMLDivElement>(null);
  const langRef = useRef<HTMLDivElement>(null);
  const testimonialsRef = useRef<HTMLDivElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);

  const featuresInView = useInView(featuresRef, { once: true, margin: '-80px' });
  const howInView = useInView(howRef, { once: true, margin: '-80px' });
  const testimonialInView = useInView(testimonialsRef, { once: true, margin: '-80px' });

  const { scrollYProgress } = useScroll({ target: heroRef });
  const heroY = useTransform(scrollYProgress, [0, 1], [0, 120]);
  const heroOpacity = useTransform(scrollYProgress, [0, 0.6], [1, 0]);

  const features = [
    {
      icon: <Robot size={28} weight="fill" />,
      title: 'AI Farm Advisor',
      description: 'Ask anything about your crops in your language. Get expert advice instantly.',
      color: '#2E7D32',
    },
    {
      icon: <Scan size={28} weight="fill" />,
      title: 'Disease Scanner',
      description: 'Point your camera at a leaf for instant AI-powered disease diagnosis.',
      color: '#EF6C00',
    },
    {
      icon: <CloudSun size={28} weight="fill" />,
      title: 'Weather Forecast',
      description: '7-day forecast with farming-specific advisories for irrigation and spraying.',
      color: '#1565C0',
    },
    {
      icon: <ChartLine size={28} weight="fill" />,
      title: 'Market Prices',
      description: 'Live mandi prices from 500+ APMCs across India. Know when to sell.',
      color: '#558B2F',
    },
    {
      icon: <Plant size={28} weight="fill" />,
      title: 'Crop Recommender',
      description: 'AI picks the best crops for your soil type and current season.',
      color: '#F9A825',
    },
    {
      icon: <Bank size={28} weight="fill" />,
      title: 'Govt Schemes',
      description: 'Find PM-KISAN, crop insurance, subsidies and more — personalized to you.',
      color: '#7B1FA2',
    },
  ];

  const languages = [
    'English', 'हिंदी', 'ಕನ್ನಡ', 'தமிழ்', 'తెలుగు', 'मराठी',
    'ਪੰਜਾਬੀ', 'বাংলা', 'ગુજરાતી', 'ଓଡ଼ିଆ', 'മലയാളം', 'অসমীয়া',
    'English', 'हिंदी', 'ಕನ್ನಡ', 'தமிழ்', 'తెలుగు', 'मराठी',
    'ਪੰਜਾਬੀ', 'বাংলা', 'ગુજરાતી', 'ଓଡ଼ିଆ', 'മലയാളം', 'অসমীয়া',
  ];

  const testimonials = [
    {
      initials: 'RS',
      color: '#2E7D32',
      quote: 'मेरी गेहूं की फसल में पीलापन आ रहा था। FarmGenius ने 5 मिनट में बताया कि nitrogen की कमी है। उर्वरक डाला, 10 दिन में फसल हरी हो गई।',
      translation:
        'My wheat crop was turning yellow. FarmGenius told me in 5 minutes it was nitrogen deficiency. I applied fertilizer and the crop turned green in 10 days.',
      name: 'Ramesh Singh',
      location: 'Bareilly, UP',
      crop: 'Wheat',
      rating: 5,
    },
    {
      initials: 'PK',
      color: '#F57F17',
      quote: 'ನನ್ನ ಟೊಮೇಟೊ ಗಿಡಗಳಲ್ಲಿ ರೋಗ ಬಂದಿತ್ತು. ಫೋಟೋ ತೆಗೆದು ಅಪ್ಲೋಡ್ ಮಾಡಿದೆ, ತಕ್ಷಣ ಉತ್ತರ ಬಂತು.',
      translation:
        'My tomato plants had a disease. I uploaded a photo and got an instant answer. Saved my entire crop!',
      name: 'Prakash Kumar',
      location: 'Dharwad, Karnataka',
      crop: 'Tomato',
      rating: 5,
    },
    {
      initials: 'SB',
      color: '#1565C0',
      quote: 'Mandi price alert ne mujhe sahi time pe bechne mein help kiya. ₹3,000 per quintal zyada mila is baar!',
      translation:
        'The mandi price alert helped me sell at the right time. I got ₹3,000 per quintal more this time!',
      name: 'Sunita Bai',
      location: 'Nagpur, Maharashtra',
      crop: 'Onion',
      rating: 5,
    },
  ];

  return (
    <div className="min-h-screen bg-bg font-noto overflow-x-hidden">

      {/* ── Navbar ─────────────────────────────────────────────────────────── */}
      <motion.nav
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-4 sm:px-8 py-3 bg-primary-dark/90 backdrop-blur-md border-b border-white/10"
      >
        <div className="flex items-center gap-2">
          <div className="w-9 h-9 rounded-full bg-accent flex items-center justify-center">
            <Plant size={20} weight="fill" className="text-white" />
          </div>
          <span className="font-poppins font-bold text-white text-lg">FarmGenius</span>
        </div>
        <div className="hidden sm:flex items-center gap-6">
          {['Features', 'How it Works', 'Languages', 'Testimonials'].map((link) => (
            <a
              key={link}
              href={`#${link.toLowerCase().replace(/ /g, '-')}`}
              className="font-noto text-white/80 text-sm hover:text-white transition-colors"
            >
              {link}
            </a>
          ))}
        </div>
        <div className="flex items-center gap-3">
          <Link
            to="/login"
            className="font-poppins font-medium text-white/80 text-sm hover:text-white transition-colors hidden sm:block"
          >
            Login
          </Link>
          <Link
            to="/login"
            className="font-poppins font-semibold text-sm bg-accent text-white px-4 py-2 rounded-full hover:bg-accent-dark transition-colors shadow-accent min-h-[44px] flex items-center"
          >
            Get Started
          </Link>
        </div>
      </motion.nav>

      {/* ── Hero Section ───────────────────────────────────────────────────── */}
      <section
        ref={heroRef}
        className="relative min-h-screen flex items-center overflow-hidden pt-16"
        style={{
          background: 'linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #388E3C 100%)',
        }}
      >
        {/* Animated blobs */}
        <FloatingBlob className="w-96 h-96 bg-white/5 -top-24 -left-24" />
        <FloatingBlob className="w-72 h-72 bg-accent/10 top-1/3 right-0" />
        <FloatingBlob className="w-56 h-56 bg-white/5 bottom-0 left-1/4" />

        {/* Subtle grid overlay */}
        <div
          className="absolute inset-0 opacity-5 pointer-events-none"
          style={{
            backgroundImage: 'radial-gradient(circle at 1px 1px, white 1px, transparent 0)',
            backgroundSize: '40px 40px',
          }}
        />

        <motion.div
          style={{ y: heroY, opacity: heroOpacity }}
          className="relative z-10 w-full max-w-7xl mx-auto px-4 sm:px-8 py-16 grid md:grid-cols-2 gap-12 items-center"
        >
          {/* Left column */}
          <div className="flex flex-col gap-6">
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="inline-flex items-center gap-2 bg-white/15 backdrop-blur-sm border border-white/25 rounded-full px-4 py-2 w-fit"
            >
              <span className="text-base">🌱</span>
              <span className="font-poppins font-medium text-white text-sm">
                Trusted by 50,000+ Farmers
              </span>
            </motion.div>

            {/* Headline */}
            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.2 }}
              className="font-poppins font-bold text-white text-4xl sm:text-5xl lg:text-6xl leading-tight"
            >
              Farm Smarter with{' '}
              <span className="text-accent">AI-Powered</span>{' '}
              Insights
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.35 }}
              className="font-noto text-white/80 text-base sm:text-lg leading-relaxed max-w-lg"
            >
              Get crop advice, disease detection, weather forecasts, and market prices — all in one place.
              Now in <strong className="text-white">25+ Indian languages</strong>.
            </motion.p>

            {/* CTA buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.5 }}
              className="flex flex-wrap gap-3"
            >
              <motion.button
                whileHover={{ scale: 1.04, boxShadow: '0 8px 24px rgba(249, 168, 37, 0.45)' }}
                whileTap={{ scale: 0.97 }}
                onClick={() => navigate('/login')}
                className="font-poppins font-bold text-white bg-accent px-7 py-3.5 rounded-full shadow-accent min-h-[52px] flex items-center gap-2 text-base transition-all"
              >
                Get Started Free
                <ArrowRight size={20} weight="bold" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.04, backgroundColor: 'rgba(255,255,255,0.12)' }}
                whileTap={{ scale: 0.97 }}
                className="font-poppins font-semibold text-white border-2 border-white/50 px-7 py-3.5 rounded-full min-h-[52px] flex items-center gap-2 text-base transition-all"
              >
                ▶ Watch Demo
              </motion.button>
            </motion.div>

            {/* Trust indicators */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.65 }}
              className="flex flex-wrap gap-4"
            >
              {['Free to use', 'No registration required', 'Works offline'].map((text) => (
                <div key={text} className="flex items-center gap-1.5">
                  <CheckCircle size={16} weight="fill" className="text-accent" />
                  <span className="font-noto text-white/75 text-sm">{text}</span>
                </div>
              ))}
            </motion.div>
          </div>

          {/* Right column — Dashboard Mockup */}
          <div className="hidden md:flex justify-center items-center">
            <DashboardMockup />
          </div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1 text-white/50"
        >
          <span className="font-noto text-xs">Scroll to explore</span>
          <div className="w-5 h-8 border border-white/40 rounded-full flex justify-center pt-1">
            <div className="w-1 h-2 bg-white/60 rounded-full" />
          </div>
        </motion.div>
      </section>

      {/* ── Features Section ────────────────────────────────────────────────── */}
      <section id="features" ref={featuresRef} className="py-20 px-4 sm:px-8 max-w-7xl mx-auto">
        <motion.div
          initial="hidden"
          animate={featuresInView ? 'visible' : 'hidden'}
          variants={{ visible: { transition: { staggerChildren: 0.1 } } }}
          className="text-center mb-14"
        >
          <motion.div variants={fadeUp} custom={0}>
            <span className="font-poppins font-semibold text-primary text-sm uppercase tracking-widest">
              Features
            </span>
          </motion.div>
          <motion.h2
            variants={fadeUp}
            custom={0.1}
            className="font-poppins font-bold text-text-primary text-3xl sm:text-4xl mt-2 mb-4"
          >
            Everything a Farmer Needs
          </motion.h2>
          <motion.p
            variants={fadeUp}
            custom={0.2}
            className="font-noto text-text-secondary text-base max-w-xl mx-auto"
          >
            One app. All the tools. Designed for Indian farmers working in the field.
          </motion.p>
        </motion.div>

        <motion.div
          initial="hidden"
          animate={featuresInView ? 'visible' : 'hidden'}
          variants={{ visible: { transition: { staggerChildren: 0.1 } } }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5"
        >
          {features.map((f, i) => (
            <FeatureCard key={f.title} {...f} delay={i * 0.08} />
          ))}
        </motion.div>
      </section>

      {/* ── Stats Bar ───────────────────────────────────────────────────────── */}
      <section
        ref={statsRef}
        className="py-14"
        style={{ background: 'linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%)' }}
      >
        <div className="max-w-5xl mx-auto px-4 sm:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <StatItem value="50000" suffix="+" label="Farmers Empowered" />
            <StatItem value="25" suffix="+" label="Indian Languages" />
            <StatItem value="500" suffix="+" label="APMCs Covered" />
            <StatItem value="22" suffix="" label="Diseases Detected" />
          </div>
        </div>
      </section>

      {/* ── How It Works ─────────────────────────────────────────────────────── */}
      <section id="how-it-works" ref={howRef} className="py-20 px-4 sm:px-8 bg-surface-variant">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial="hidden"
            animate={howInView ? 'visible' : 'hidden'}
            variants={{ visible: { transition: { staggerChildren: 0.12 } } }}
            className="text-center mb-14"
          >
            <motion.span variants={fadeUp} custom={0} className="font-poppins font-semibold text-primary text-sm uppercase tracking-widest">
              Simple Steps
            </motion.span>
            <motion.h2
              variants={fadeUp}
              custom={0.1}
              className="font-poppins font-bold text-text-primary text-3xl sm:text-4xl mt-2"
            >
              How It Works
            </motion.h2>
          </motion.div>

          <motion.div
            initial="hidden"
            animate={howInView ? 'visible' : 'hidden'}
            variants={{ visible: { transition: { staggerChildren: 0.15 } } }}
            className="flex flex-col md:flex-row gap-6 md:gap-0 md:items-start justify-between"
          >
            <StepItem
              number={1}
              title="Tell Us About Your Farm"
              description="Share your location, crops grown, and soil type. Takes less than 2 minutes."
              delay={0}
            />
            <StepItem
              number={2}
              title="Ask or Scan"
              description="Ask a question in your language, or take a photo of a diseased leaf."
              delay={0.15}
            />
            <StepItem
              number={3}
              title="Get Instant Answers"
              description="Receive AI-powered advice, disease diagnosis, and treatment plans immediately."
              delay={0.3}
              isLast
            />
          </motion.div>
        </div>
      </section>

      {/* ── Language Section ─────────────────────────────────────────────────── */}
      <section id="languages" ref={langRef} className="py-16 px-4 sm:px-8 overflow-hidden">
        <div className="max-w-5xl mx-auto text-center mb-10">
          <span className="font-poppins font-semibold text-primary text-sm uppercase tracking-widest">
            Multilingual
          </span>
          <h2 className="font-poppins font-bold text-text-primary text-3xl sm:text-4xl mt-2 mb-3">
            Available in Your Language
          </h2>
          <p className="font-noto text-text-secondary">
            Because every farmer deserves advice in the language they think in.
          </p>
        </div>

        {/* Marquee */}
        <div className="relative">
          <div className="absolute left-0 top-0 bottom-0 w-16 bg-gradient-to-r from-bg to-transparent z-10 pointer-events-none" />
          <div className="absolute right-0 top-0 bottom-0 w-16 bg-gradient-to-l from-bg to-transparent z-10 pointer-events-none" />
          <div className="flex overflow-hidden">
            <motion.div
              className="flex gap-3 shrink-0"
              animate={{ x: ['0%', '-50%'] }}
              transition={{ duration: 30, repeat: Infinity }}
            >
              {languages.map((lang, i) => (
                <div
                  key={`${lang}-${i}`}
                  className="shrink-0 bg-white border border-farm-divider rounded-full px-5 py-2.5 font-noto text-text-primary text-sm font-medium shadow-card hover:border-primary hover:text-primary transition-colors cursor-default whitespace-nowrap"
                >
                  {lang}
                </div>
              ))}
            </motion.div>
          </div>
        </div>

        {/* Second row — reverse */}
        <div className="relative mt-3">
          <div className="absolute left-0 top-0 bottom-0 w-16 bg-gradient-to-r from-bg to-transparent z-10 pointer-events-none" />
          <div className="absolute right-0 top-0 bottom-0 w-16 bg-gradient-to-l from-bg to-transparent z-10 pointer-events-none" />
          <div className="flex overflow-hidden">
            <motion.div
              className="flex gap-3 shrink-0"
              animate={{ x: ['-50%', '0%'] }}
              transition={{ duration: 28, repeat: Infinity }}
            >
              {[...languages].reverse().map((lang, i) => (
                <div
                  key={`rev-${lang}-${i}`}
                  className="shrink-0 bg-surface-variant border border-farm-divider rounded-full px-5 py-2.5 font-noto text-text-secondary text-sm font-medium shadow-card whitespace-nowrap cursor-default"
                >
                  {lang}
                </div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* ── Testimonials ─────────────────────────────────────────────────────── */}
      <section
        id="testimonials"
        ref={testimonialsRef}
        className="py-20 px-4 sm:px-8 bg-surface-variant"
      >
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial="hidden"
            animate={testimonialInView ? 'visible' : 'hidden'}
            variants={{ visible: { transition: { staggerChildren: 0.1 } } }}
            className="text-center mb-14"
          >
            <motion.span variants={fadeUp} custom={0} className="font-poppins font-semibold text-primary text-sm uppercase tracking-widest">
              Farmer Stories
            </motion.span>
            <motion.h2
              variants={fadeUp}
              custom={0.1}
              className="font-poppins font-bold text-text-primary text-3xl sm:text-4xl mt-2"
            >
              What Farmers Are Saying
            </motion.h2>
          </motion.div>

          <motion.div
            initial="hidden"
            animate={testimonialInView ? 'visible' : 'hidden'}
            variants={{ visible: { transition: { staggerChildren: 0.15 } } }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
          >
            {testimonials.map((t, i) => (
              <TestimonialCard key={t.name} {...t} delay={i * 0.12} />
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── CTA Section ──────────────────────────────────────────────────────── */}
      <section
        ref={ctaRef}
        id="cta"
        className="py-24 px-4 sm:px-8 text-center relative overflow-hidden"
        style={{
          background: 'linear-gradient(135deg, #1B5E20 0%, #2E7D32 60%, #F9A825 200%)',
        }}
      >
        <FloatingBlob className="w-80 h-80 bg-white/5 -top-20 right-0" />
        <FloatingBlob className="w-56 h-56 bg-accent/10 bottom-0 left-10" />

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={{ visible: { transition: { staggerChildren: 0.12 } } }}
          className="relative z-10 max-w-3xl mx-auto"
        >
          <motion.h2
            variants={fadeUp}
            custom={0}
            className="font-poppins font-bold text-white text-3xl sm:text-5xl mb-4 leading-tight"
          >
            Ready to Transform Your Farming?
          </motion.h2>
          <motion.p
            variants={fadeUp}
            custom={0.1}
            className="font-noto text-white/80 text-base sm:text-lg mb-8"
          >
            Join 50,000+ farmers who are already farming smarter with FarmGenius.
            Free forever. No credit card. No registration.
          </motion.p>
          <motion.button
            variants={scaleIn}
            custom={0.2}
            whileHover={{ scale: 1.05, boxShadow: '0 12px 32px rgba(249, 168, 37, 0.5)' }}
            whileTap={{ scale: 0.97 }}
            onClick={() => navigate('/login')}
            className="font-poppins font-bold text-primary-dark bg-accent px-10 py-4 rounded-full text-lg shadow-accent min-h-[56px] inline-flex items-center gap-3 transition-all"
          >
            🌱 Start for Free
            <ArrowRight size={22} weight="bold" />
          </motion.button>
          <motion.p variants={fadeIn} custom={0.4} className="font-noto text-white/50 text-sm mt-6">
            Available in Hindi · Kannada · Tamil · Telugu · Marathi · Punjabi and 20 more languages
          </motion.p>
        </motion.div>
      </section>

      {/* ── Footer ───────────────────────────────────────────────────────────── */}
      <footer className="bg-primary-dark py-16 px-4 sm:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-10 mb-12">
            {/* Brand */}
            <div className="col-span-2 md:col-span-1">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-9 h-9 rounded-full bg-accent flex items-center justify-center">
                  <Plant size={20} weight="fill" className="text-white" />
                </div>
                <span className="font-poppins font-bold text-white text-lg">FarmGenius</span>
              </div>
              <p className="font-noto text-white/50 text-sm leading-relaxed mb-4">
                Empowering Indian farmers with AI-powered insights for a better harvest.
              </p>
              <div className="flex gap-3">
                {[WhatsappLogo, TwitterLogo, InstagramLogo].map((Icon, i) => (
                  <motion.button
                    key={i}
                    whileHover={{ scale: 1.15, color: '#F9A825' }}
                    className="text-white/50 transition-colors"
                  >
                    <Icon size={22} weight="fill" />
                  </motion.button>
                ))}
              </div>
            </div>

            {/* Links */}
            {[
              {
                title: 'Product',
                links: ['AI Advisor', 'Disease Scanner', 'Market Prices', 'Weather', 'Crop Recommender'],
              },
              {
                title: 'Resources',
                links: ['Documentation', 'Farmer Guide', 'Blog', 'KVK Directory', 'API Access'],
              },
              {
                title: 'Languages',
                links: ['हिंदी', 'ಕನ್ನಡ', 'தமிழ்', 'తెలుగు', 'मराठी'],
              },
              {
                title: 'Legal',
                links: ['Privacy Policy', 'Terms of Use', 'Cookie Policy', 'Disclaimer'],
              },
            ].map((col) => (
              <div key={col.title}>
                <h4 className="font-poppins font-semibold text-white text-sm mb-4">{col.title}</h4>
                <ul className="flex flex-col gap-2.5">
                  {col.links.map((link) => (
                    <li key={link}>
                      <a
                        href="#"
                        className="font-noto text-white/50 text-sm hover:text-white transition-colors"
                      >
                        {link}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          <div className="border-t border-white/10 pt-8 flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="font-noto text-white/40 text-sm text-center sm:text-left">
              © 2026 FarmGenius. Made with ❤️ for Indian Farmers. All rights reserved.
            </p>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span className="font-noto text-white/40 text-sm">All systems operational</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
