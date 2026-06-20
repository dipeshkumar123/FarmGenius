// src/pages/ScanPage.tsx
import React, { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import {
  Camera,
  Image as ImageIcon,
  ArrowCounterClockwise,
  Chat,
  CheckCircle,
  WarningCircle,
  ShieldWarning,
  Pill,
  Leaf,
  X,
} from 'phosphor-react';
import PageWrapper from '../components/ui/PageWrapper';
import apiClient from '../api/client';

// ─── Types ───────────────────────────────────────────────────────────────────

type ScanState = 'idle' | 'uploaded' | 'processing' | 'result' | 'error';

interface Treatment {
  name: string;
  dosage: string;
  frequency: string;
  cost: string;
}

interface ScanResult {
  disease: string;
  confidence: number;
  type: string;
  severity: 'Mild' | 'Moderate' | 'Severe';
  description: string;
  immediateActions: string[];
  treatments: Treatment[];
  prevention: string[];
}

type ResultTab = 'about' | 'actions' | 'treatment' | 'prevention';

// ─── Mock result data ─────────────────────────────────────────────────────────

const mockResult: ScanResult = {
  disease: 'Tomato Late Blight',
  confidence: 0.94,
  type: 'Fungal',
  severity: 'Severe',
  description:
    'A fungal disease caused by Phytophthora infestans. Spreads rapidly in humid conditions above 80% relative humidity. Can destroy an entire crop within days if untreated.',
  immediateActions: [
    'Remove and destroy all affected leaves immediately',
    'Avoid overhead watering — water at soil level only',
    'Ensure good air circulation between plants',
  ],
  treatments: [
    {
      name: 'Mancozeb 75% WP',
      dosage: '2g per liter',
      frequency: 'Every 7 days',
      cost: '~₹120/kg',
    },
    {
      name: 'Cymoxanil + Mancozeb',
      dosage: '3g per liter',
      frequency: 'Every 10 days for severe cases',
      cost: '~₹250/kg',
    },
  ],
  prevention: [
    'Crop rotation every season',
    'Use resistant varieties like Arka Rakshak',
    'Avoid working in wet conditions',
    'Apply copper-based fungicides preventatively',
  ],
};

// ─── Sub-components ──────────────────────────────────────────────────────────

function ConfidenceBar({ value, label }: { value: number; label?: string }) {
  const pct = Math.round(value * 100);
  const color = pct >= 85 ? '#2E7D32' : pct >= 70 ? '#F9A825' : '#C62828';

  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between mb-1">
          <span className="text-xs font-noto text-text-secondary">{label}</span>
          <span className="text-xs font-poppins font-bold" style={{ color }}>
            {pct}%
          </span>
        </div>
      )}
      <div className="h-2.5 rounded-full bg-surface-variant overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.9,  delay: 0.2 }}
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
      {!label && (
        <p className="text-right text-xs font-poppins font-bold mt-1" style={{ color }}>
          {pct}% confidence
        </p>
      )}
    </div>
  );
}

function SeverityBadge({ severity }: { severity: ScanResult['severity'] }) {
  const styles: Record<ScanResult['severity'], string> = {
    Mild: 'bg-green-100 text-green-700',
    Moderate: 'bg-amber-100 text-amber-700',
    Severe: 'bg-red-100 text-red-700',
  };
  const icons: Record<ScanResult['severity'], React.ReactNode> = {
    Mild: <CheckCircle size={12} weight="fill" />,
    Moderate: <WarningCircle size={12} weight="fill" />,
    Severe: <ShieldWarning size={12} weight="fill" />,
  };

  return (
    <span className={`badge ${styles[severity]} text-xs`}>
      {icons[severity]} {severity}
    </span>
  );
}

// ─── Upload tips ──────────────────────────────────────────────────────────────

const tips = [
  { icon: '☀️', text: 'Good lighting' },
  { icon: '🍃', text: 'Single leaf' },
  { icon: '🔍', text: 'Show symptoms' },
];

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function ScanPage() {
  const navigate = useNavigate();
  const [scanState, setScanState] = useState<ScanState>('idle');
  const [preview, setPreview] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [activeTab, setActiveTab] = useState<ResultTab>('about');
  const progressRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Clean up preview URL on unmount
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
      if (progressRef.current) clearInterval(progressRef.current);
    };
  }, [preview]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    setScanState('uploaded');
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp'] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
  });

  const [result, setResult] = useState<ScanResult | null>(null);

  const startProcessing = async () => {
    if (!preview) return;
    setScanState('processing');
    setProgress(0);

    // Fake progress bar animation
    let p = 0;
    progressRef.current = setInterval(() => {
      p += 100 / 30; // 3 seconds at 100ms intervals
      setProgress(Math.min(p, 90)); // Cap at 90% until real API returns
    }, 100);

    try {
      // Get the file from the preview URL (we stored it in dropzone but didn't save it to state. Let's assume we can fetch it, or better, we should store the File object)
      // Since we didn't store the File object, let's fetch it from the blob URL
      const response = await fetch(preview);
      const blob = await response.blob();
      const formData = new FormData();
      formData.append('file', blob, 'leaf.jpg');

      const apiRes = await apiClient.post('/disease/detect', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const data = apiRes.data; // DiseaseResponse

      // Map backend response to UI schema
      const scanRes: ScanResult = {
        disease: data.disease_name,
        confidence: data.confidence,
        type: 'Identified by AI Vision',
        severity: 'Moderate', // Default as backend doesn't provide this yet
        description: data.disease_name_hi ? `Also known in Hindi as: ${data.disease_name_hi}. Sourced from: ${data.source_name}` : `Sourced from: ${data.source_name}`,
        immediateActions: [
          'Isolate the affected plant if possible',
          'Avoid overhead watering',
        ],
        treatments: [
          {
            name: 'Organic Treatment',
            dosage: 'As prescribed',
            frequency: 'Regularly',
            cost: 'Low',
          },
          {
            name: 'Chemical Treatment',
            dosage: data.dosage || 'Standard dosage',
            frequency: 'As needed',
            cost: 'Medium',
          }
        ],
        prevention: [
          'Rotate crops regularly',
          'Ensure good soil drainage',
        ],
      };
      
      // Override details with API text
      scanRes.treatments[0].name = data.organic_treatment;
      scanRes.treatments[1].name = data.chemical_treatment;

      setResult(scanRes);
      
      if (progressRef.current) clearInterval(progressRef.current);
      setProgress(100);
      setTimeout(() => setScanState('result'), 400);

    } catch (error) {
      console.error('API Error:', error);
      if (progressRef.current) clearInterval(progressRef.current);
      setProgress(0);
      setScanState('error');
    }
  };

  const reset = () => {
    setScanState('idle');
    setPreview(null);
    setResult(null);
    setProgress(0);
    setActiveTab('about');
    if (progressRef.current) clearInterval(progressRef.current);
  };

  const askAI = () => {
    if (!result) return;
    navigate('/chat', {
      state: {
        prefill: `I found ${result.disease} on my plants. What should I do?`,
      },
    });
  };

  const tabLabels: { key: ResultTab; label: string }[] = [
    { key: 'about', label: 'What is this?' },
    { key: 'actions', label: 'Immediate Actions' },
    { key: 'treatment', label: 'Treatment' },
    { key: 'prevention', label: 'Prevention' },
  ];

  const displayResult = result;

  return (
    <PageWrapper noPadding>
      <div className="px-4 pt-4 pb-28">
        {/* Page header */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6"
        >
          <h1 className="font-poppins font-bold text-2xl text-text-primary">
            🔬 Crop Disease Scanner
          </h1>
          <p className="text-text-secondary font-noto text-sm mt-0.5">
            Take or upload a photo of your crop leaf
          </p>
        </motion.div>

        <AnimatePresence mode="wait">
          {/* ── IDLE STATE ─────────────────────────────────────────────────── */}
          {scanState === 'idle' && (
            <motion.div
              key="idle"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.35 }}
            >
              {/* Dropzone */}
              <div
                {...(getRootProps() as any)}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200 min-h-[260px] flex flex-col items-center justify-center gap-4
                  ${isDragActive
                    ? 'border-primary bg-surface-variant shadow-inner'
                    : 'border-farm-divider bg-white hover:border-primary hover:bg-surface-variant/50'
                  }`}
              >
                <input {...getInputProps()} />
                <motion.div
                  animate={isDragActive ? { scale: 1.1, rotate: 5 } : { scale: 1, rotate: 0 }}
                  transition={{ type: 'spring', stiffness: 300 }}
                >
                  <Camera
                    size={64}
                    weight="light"
                    className={isDragActive ? 'text-primary' : 'text-text-secondary'}
                  />
                </motion.div>
                <div>
                  <p className="font-poppins font-semibold text-text-primary text-base">
                    {isDragActive ? 'Drop your photo here!' : 'Drag a leaf photo here'}
                  </p>
                  <p className="text-text-secondary font-noto text-sm mt-1">
                    or tap to browse your gallery
                  </p>
                </div>
                <div className="flex gap-4 mt-2">
                  {tips.map(({ icon, text }) => (
                    <div key={text} className="flex items-center gap-1.5 text-xs font-noto text-text-secondary">
                      <span>{icon}</span>
                      <span>{text}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Action buttons */}
              <div className="flex gap-3 mt-4">
                <motion.button
                  {...(getRootProps() as any)}
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.96 }}
                  className="flex-1 flex items-center justify-center gap-2 bg-primary text-white font-poppins font-semibold py-3 px-4 rounded-full shadow-card hover:shadow-card-hover transition-all"
                >
                  <input {...getInputProps()} />
                  <Camera size={20} weight="fill" />
                  <span className="sm:hidden">Take Photo</span>
                  <span className="hidden sm:inline">Take Photo</span>
                </motion.button>
                <motion.button
                  {...(getRootProps() as any)}
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.96 }}
                  className="flex-1 flex items-center justify-center gap-2 btn-secondary"
                >
                  <input {...getInputProps()} />
                  <ImageIcon size={20} weight="fill" />
                  Browse Gallery
                </motion.button>
              </div>

              {/* Info card */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="mt-5 bg-surface-variant rounded-md p-4 flex gap-3"
              >
                <Leaf size={22} weight="fill" className="text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-poppins font-semibold text-text-primary text-sm">
                    22 diseases detected
                  </p>
                  <p className="text-text-secondary font-noto text-xs mt-0.5">
                    Covers tomato, potato, rice, wheat, cotton, chickpea &amp; maize
                  </p>
                </div>
              </motion.div>
            </motion.div>
          )}

          {/* ── UPLOADED STATE ─────────────────────────────────────────────── */}
          {scanState === 'uploaded' && preview && (
            <motion.div
              key="uploaded"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.35 }}
              className="flex flex-col gap-4"
            >
              <div className="relative rounded-lg overflow-hidden shadow-card">
                <img
                  src={preview}
                  alt="Uploaded leaf"
                  className="w-full max-h-72 object-cover"
                />
                <button
                  onClick={reset}
                  className="absolute top-3 right-3 bg-black/50 text-white rounded-full p-1.5 hover:bg-black/70 transition"
                >
                  <X size={16} weight="bold" />
                </button>
              </div>
              <p className="text-center font-noto text-text-secondary text-sm">
                Photo looks good! Ready to scan?
              </p>
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={startProcessing}
                className="btn-primary w-full text-base flex items-center justify-center gap-2 py-4"
              >
                🔬 Analyze Photo
              </motion.button>
              <button onClick={reset} className="btn-ghost w-full">
                Choose a different photo
              </button>
            </motion.div>
          )}

          {/* ── PROCESSING STATE ───────────────────────────────────────────── */}
          {scanState === 'processing' && preview && (
            <motion.div
              key="processing"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.35 }}
              className="flex flex-col items-center gap-6"
            >
              {/* Blurred image with scanning overlay */}
              <div className="relative w-full rounded-lg overflow-hidden shadow-card" style={{ height: 280 }}>
                <img
                  src={preview}
                  alt="Scanning"
                  className="w-full h-full object-cover"
                  style={{ filter: 'blur(3px) brightness(0.7)' }}
                />
                {/* Scanning line */}
                <motion.div
                  animate={{ top: ['0%', '100%'] }}
                  transition={{ duration: 2,  repeat: Infinity, repeatType: 'reverse' }}
                  className="absolute left-0 right-0 h-1 bg-primary/80 shadow-lg"
                  style={{ boxShadow: '0 0 16px 4px rgba(46,125,50,0.6)' }}
                />
                {/* Center overlay text */}
                <div className="absolute inset-0 flex flex-col items-center justify-center text-white">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2,  repeat: Infinity }}
                    className="w-12 h-12 rounded-full border-4 border-white/30 border-t-white mb-3"
                  />
                  <p className="font-poppins font-bold text-lg drop-shadow">🔬 Analyzing your crop...</p>
                  <p className="font-noto text-sm text-white/80 drop-shadow">
                    Checking against 22 known diseases
                  </p>
                </div>
              </div>

              {/* Progress bar */}
              <div className="w-full">
                <div className="flex justify-between text-xs font-noto text-text-secondary mb-1.5">
                  <span>Analyzing patterns...</span>
                  <span className="font-poppins font-bold text-primary">
                    {Math.round(progress)}%
                  </span>
                </div>
                <div className="h-3 rounded-full bg-surface-variant overflow-hidden">
                  <motion.div
                    className="h-full rounded-full bg-gradient-primary"
                    style={{ width: `${progress}%` }}
                    transition={{ duration: 0.1 }}
                  />
                </div>
              </div>

              <p className="text-text-secondary font-noto text-sm text-center">
                This usually takes 3–5 seconds
              </p>
            </motion.div>
          )}

          {/* ── ERROR STATE ────────────────────────────────────────────────── */}
          {scanState === 'error' && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.35 }}
              className="flex flex-col items-center gap-5 py-8 text-center"
            >
              <div className="w-20 h-20 rounded-full bg-red-100 flex items-center justify-center">
                <WarningCircle size={40} weight="fill" className="text-red-500" />
              </div>
              <div>
                <h2 className="font-poppins font-bold text-lg text-text-primary">Detection Failed</h2>
                <p className="font-noto text-sm text-text-secondary mt-1 max-w-xs">
                  Could not analyze your photo. Please ensure the leaf is clearly visible, well-lit, and try again.
                </p>
              </div>
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={reset}
                className="btn-primary w-full flex items-center justify-center gap-2 py-4 text-base"
              >
                <ArrowCounterClockwise size={20} weight="bold" />
                Try Again
              </motion.button>
            </motion.div>
          )}

          {/* ── RESULT STATE ───────────────────────────────────────────────── */}
          {scanState === 'result' && displayResult && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45, }}
              className="flex flex-col gap-5"
            >
              {/* Image with simulated affected area overlay */}
              {preview && (
                <div className="relative rounded-lg overflow-hidden shadow-card">
                  <img
                    src={preview}
                    alt="Scanned crop"
                    className="w-full max-h-60 object-cover"
                  />
                  {/* Simulated red highlight overlay */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <motion.div
                      initial={{ opacity: 0, scale: 0.6 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.4, duration: 0.5, type: 'spring' }}
                      className="w-24 h-24 rounded-full border-4 border-red-500 bg-red-500/20"
                    >
                      <div className="w-full h-full flex items-center justify-center">
                        <span className="text-red-500 font-poppins font-bold text-xs text-center leading-tight px-1">
                          Affected<br />Area
                        </span>
                      </div>
                    </motion.div>
                  </div>
                  {/* Overlay badge */}
                  <div className="absolute top-3 left-3">
                    <SeverityBadge severity={displayResult.severity} />
                  </div>
                </div>
              )}

              {/* Disease info card */}
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.15 }}
                className="card p-4"
              >
                <div className="flex items-start justify-between gap-2 mb-3">
                  <div>
                    <h2 className="font-poppins font-bold text-xl text-text-primary">
                      {displayResult.disease}
                    </h2>
                    <p className="text-text-secondary font-noto text-sm mt-0.5">
                      Type: <span className="text-primary font-semibold">{displayResult.type}</span>
                    </p>
                  </div>
                  <SeverityBadge severity={displayResult.severity} />
                </div>
                <ConfidenceBar value={displayResult.confidence} label="Detection Confidence" />
              </motion.div>

              {/* Tabs */}
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.25 }}
              >
                {/* Tab headers */}
                <div className="flex gap-1 bg-surface-variant rounded-lg p-1 mb-4 overflow-x-auto scrollbar-hide">
                  {tabLabels.map(({ key, label }) => (
                    <button
                      key={key}
                      onClick={() => setActiveTab(key)}
                      className={`shrink-0 px-3 py-2 rounded-md text-xs font-poppins font-semibold transition-all duration-200 whitespace-nowrap
                        ${activeTab === key
                          ? 'bg-white text-primary shadow-sm'
                          : 'text-text-secondary hover:text-text-primary'
                        }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>

                {/* Tab content */}
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeTab}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.22 }}
                  >
                    {activeTab === 'about' && (
                      <div className="card p-4">
                        <p className="font-noto text-text-primary text-sm leading-relaxed">
                          {displayResult.description}
                        </p>
                      </div>
                    )}

                    {activeTab === 'actions' && (
                      <div className="card p-4 flex flex-col gap-3">
                        {displayResult.immediateActions.map((action, i) => (
                          <motion.div
                            key={i}
                            initial={{ opacity: 0, x: -12 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.08 }}
                            className="flex items-start gap-3"
                          >
                            <div className="w-7 h-7 rounded-full bg-red-100 text-red-600 font-poppins font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
                              {i + 1}
                            </div>
                            <p className="font-noto text-text-primary text-sm leading-relaxed">{action}</p>
                          </motion.div>
                        ))}
                      </div>
                    )}

                    {activeTab === 'treatment' && (
                      <div className="flex flex-col gap-3">
                        {displayResult.treatments.map((t, i) => (
                          <motion.div
                            key={t.name}
                            initial={{ opacity: 0, y: 12 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.1 }}
                            className="card p-4"
                          >
                            <div className="flex items-center gap-2 mb-2">
                              <Pill size={18} weight="fill" className="text-primary" />
                              <h3 className="font-poppins font-semibold text-text-primary text-sm">
                                {t.name}
                              </h3>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                              <div className="bg-surface-variant rounded-sm p-2">
                                <p className="text-[10px] font-noto text-text-secondary">Dosage</p>
                                <p className="text-sm font-poppins font-semibold text-text-primary">{t.dosage}</p>
                              </div>
                              <div className="bg-surface-variant rounded-sm p-2">
                                <p className="text-[10px] font-noto text-text-secondary">Frequency</p>
                                <p className="text-sm font-poppins font-semibold text-text-primary">{t.frequency}</p>
                              </div>
                              <div className="bg-amber-50 rounded-sm p-2 col-span-2">
                                <p className="text-[10px] font-noto text-amber-700">Estimated Cost</p>
                                <p className="text-sm font-poppins font-bold text-amber-800">{t.cost}</p>
                              </div>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    )}

                    {activeTab === 'prevention' && (
                      <div className="card p-4 flex flex-col gap-3">
                        {displayResult.prevention.map((item, i) => (
                          <motion.div
                            key={i}
                            initial={{ opacity: 0, x: -12 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.08 }}
                            className="flex items-start gap-2"
                          >
                            <CheckCircle size={18} weight="fill" className="text-primary shrink-0 mt-0.5" />
                            <p className="font-noto text-text-primary text-sm">{item}</p>
                          </motion.div>
                        ))}
                      </div>
                    )}
                  </motion.div>
                </AnimatePresence>
              </motion.div>

              {/* Action buttons */}
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.35 }}
                className="flex flex-col gap-3 pb-4"
              >
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={askAI}
                  className="btn-primary w-full flex items-center justify-center gap-2 py-4 text-base"
                >
                  <Chat size={20} weight="fill" />
                  Ask AI about this
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={reset}
                  className="btn-secondary w-full flex items-center justify-center gap-2 py-4"
                >
                  <ArrowCounterClockwise size={20} weight="bold" />
                  Scan Another
                </motion.button>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </PageWrapper>
  );
}
