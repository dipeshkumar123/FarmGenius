// src/pages/ChatPage.tsx
import {  
  useState,
  useRef,
  useEffect,
  useCallback,
} from 'react';
import type { ChangeEvent } from 'react';
import type { KeyboardEvent } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft,
  PaperPlaneRight,
  Microphone,
  MicrophoneSlash,
  Image as ImageIcon,
  Translate,
  WifiX,
  WifiHigh,
  Robot,
} from 'phosphor-react';
import { useAppStore } from '../store/appStore';
import { apiClient } from '../api/client';

// ─── Types ────────────────────────────────────────────────────────────────────

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  quickReplies?: string[];
  isLoading?: boolean;
  imageUrl?: string;
}

interface LanguageOption {
  code: string;
  label: string;
  nativeLabel: string;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const LANGUAGES: LanguageOption[] = [
  { code: 'en', label: 'English', nativeLabel: 'English' },
  { code: 'hi', label: 'Hindi',   nativeLabel: 'हिंदी'  },
  { code: 'kn', label: 'Kannada', nativeLabel: 'ಕನ್ನಡ' },
  { code: 'te', label: 'Telugu',  nativeLabel: 'తెలుగు' },
  { code: 'ta', label: 'Tamil',   nativeLabel: 'தமிழ்' },
  { code: 'mr', label: 'Marathi', nativeLabel: 'मराठी' },
];

const WELCOME_QUICK_REPLIES: Record<string, string[]> = {
  en: ['What crops to grow?', 'Check wheat price', 'Disease in my tomato', 'Weather this week'],
  hi: ['कौन सी फसल उगाएं?', 'गेहूं का भाव', 'टमाटर में रोग', 'इस हफ्ते मौसम'],
  kn: ['ಯಾವ ಬೆಳೆ ಬೆಳೆಯಬೇಕು?', 'ಗೋಧಿ ಬೆಲೆ', 'ಟೊಮ್ಯಾಟೊ ರೋಗ', 'ಈ ವಾರ ಹವಾಮಾನ'],
  te: ['ఏ పంట వేయాలి?', 'గోధుమ ధర', 'టమాటో వ్యాధి', 'ఈ వారం వాతావరణం'],
  ta: ['என்ன பயிர் வளர்க்கலாம்?', 'கோதுமை விலை', 'தக்காளி நோய்', 'இந்த வார வானிலை'],
  mr: ['कोणते पीक घ्यावे?', 'गव्हाचा भाव', 'टोमॅटोचा रोग', 'या आठवड्याचे हवामान'],
};

const QUICK_REPLIES_BY_CONTEXT: Record<string, Record<string, string[]>> = {
  price: {
    en: ['Check another crop', 'When to sell?', 'Weather this week'],
    hi: ['दूसरी फसल चेक करें', 'कब बेचें?', 'इस हफ्ते मौसम'],
    kn: ['ಇನ್ನೊಂದು ಬೆಳೆ ತಪಾಸಣೆ', 'ಯಾವಾಗ ಮಾರಾಟ ಮಾಡಬೇಕು?', 'ಈ ವಾರ ಹವಾಮಾನ'],
    te: ['మరో పంట తనిఖీ', 'ఎప్పుడు అమ్మాలి?', 'ఈ వారం వాతావరణం'],
    ta: ['வேறு பயிர் சரிபார்க்க', 'எப்போது விற்கலாம்?', 'இந்த வார வானிலை'],
    mr: ['दुसरे पीक तपासा', 'केव्हा विकायचे?', 'या आठवड्याचे हवामान'],
  },
  disease: {
    en: ['What treatment?', 'How to prevent?', 'Contact KVK'],
    hi: ['क्या उपचार?', 'कैसे रोकें?', 'KVK से संपर्क'],
    kn: ['ಏನು ಚಿಕಿತ್ಸೆ?', 'ಹೇಗೆ ತಡೆಯಬೇಕು?', 'KVK ಸಂಪರ್ಕ'],
    te: ['ఏం చికిత్స?', 'ఎలా నివారించాలి?', 'KVK సంప్రదించండి'],
    ta: ['என்ன சிகிச்சை?', 'எப்படி தடுக்கலாம்?', 'KVK தொடர்பு'],
    mr: ['काय उपचार?', 'कसे रोखायचे?', 'KVK शी संपर्क'],
  },
  weather: {
    en: ['Should I irrigate today?', 'Spray tomorrow?', 'Harvest timing'],
    hi: ['आज सिंचाई करूं?', 'कल स्प्रे करें?', 'कटाई का समय'],
    kn: ['ಇಂದು ನೀರಾವರಿ ಮಾಡಬೇಕೇ?', 'ನಾಳೆ ಸ್ಪ್ರೇ?', 'ಕಟಾವು ಸಮಯ'],
    te: ['ఈరోజు నీళ్ళు పెట్టాలా?', 'రేపు స్ప్రే చేయాలా?', 'కోత సమయం'],
    ta: ['இன்று நீர்ப்பாசனம் செய்யலாமா?', 'நாளை தெளிக்கலாமா?', 'அறுவடை நேரம்'],
    mr: ['आज सिंचन करावे का?', 'उद्या फवारणी?', 'कापणीची वेळ'],
  },
};

const MOCK_RESPONSES: Record<string, string> = {
  wheat:
    'Current wheat price in Dharwad APMC is **₹2,180/quintal** — up ₹35 from yesterday. Best time to sell is next week when prices are expected to rise further. 📈',
  maize:
    'Maize (Makka) prices are at **₹1,920/quintal** today — slightly down 1%. Consider holding stock for 3–5 days as prices usually recover after the weekend.',
  soybean:
    'Soybean is trading at **₹4,420/quintal** — stable with no change. Dharwad APMC has good demand. You can sell now or wait for post-festival demand.',
  rice:
    'Rice (Paddy) prices are **₹2,060/quintal** — up 2% this week. Government MSP is ₹2,183 for common grade. Check if your APMC offers better rates.',
  cotton:
    'Cotton prices are at **₹6,150/quintal** — down slightly. Hold if possible; Diwali season usually brings better prices in 2–3 weeks.',
  tomato:
    'Common tomato diseases in Karnataka this season:\n1. **Early Blight** — brown spots with rings, use Mancozeb 2g/L.\n2. **Leaf Curl Virus** — upward curling, remove affected plants.\n\nShare a photo for accurate diagnosis. 📷',
  weather:
    'This week in Dharwad: **32°C** with partly cloudy skies ⛅. Rain expected Thursday–Friday 🌧️. Avoid pesticide spray on Wednesday evening. Good conditions for irrigation on Monday morning.',
  rain:
    '🌧️ Rain forecast: **60% chance Thursday, 75% Friday**. Expected rainfall 12–18mm. This is good for Kharif crops. Avoid any spray treatments from Wednesday evening.',
  pest:
    '⚠️ High **aphid** activity reported in Dharwad district this week. For wheat: spray Imidacloprid 17.8 SL @ 0.5ml/L water. For tomato: use Dimethoate 30 EC @ 2ml/L. Contact your local KVK for a free inspection.',
  irrigation:
    '💧 Irrigation guidance: With current weather (32°C, 65% humidity), irrigate wheat every **5–6 days**. Cotton needs irrigation every **7–8 days**. Drip irrigation saves 40% water — check PM-KUSUM scheme for subsidy.',
  fertilizer:
    '🌱 For wheat at tillering stage — apply **Urea 50 kg/acre** + MOP 25 kg/acre. Best applied in the evening when soil is moist. Source: ICAR wheat cultivation guide.',
  scheme:
    '🏛️ Active government schemes for Karnataka farmers:\n1. **PM-KISAN**: ₹6,000/year direct benefit\n2. **PMFBY**: Crop insurance at 2% premium\n3. **Rythu Bandhu**: ₹5,000/acre input support\n\nAsk me about eligibility for any scheme!',
  crop:
    'Best crops for Dharwad in current season (Kharif): 🌽 Maize, 🫘 Soybean, 🌶️ Red Chilli, 🧅 Onion.\n\nFor Rabi (Oct onwards): 🌾 Wheat, 🫛 Chickpea, 🌻 Sunflower.\n\nWhat\'s your farm size and water source?',
  default:
    'I understand your farming query. For personalized advice, please tell me:\n• Your **crop type**\n• Your **location / district**\n• The **problem** you\'re facing\n\nYou can also upload a photo 📷 or speak in your language 🎙️',
};

function getMockResponse(query: string): string {
  const q = query.toLowerCase();
  for (const key of Object.keys(MOCK_RESPONSES)) {
    if (key !== 'default' && q.includes(key)) return MOCK_RESPONSES[key];
  }
  return MOCK_RESPONSES.default;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function uid(): string {
  return Math.random().toString(36).slice(2, 11);
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  });
}

// ─── Welcome message ──────────────────────────────────────────────────────────

function buildWelcomeMessage(lang: string): Message {
  const welcomeText: Record<string, string> = {
    en: "Namaste! 🙏 I'm **FarmGenius AI**, your smart farming assistant.\n\nAsk me anything about crops, diseases, weather, or market prices — in Hindi, English, Kannada, or any Indian language!",
    hi: "नमस्ते! 🙏 मैं **FarmGenius AI** हूं, आपका स्मार्ट खेती सहायक।\n\nफसल, रोग, मौसम या मंडी भाव के बारे में कुछ भी पूछें!",
    kn: "ನಮಸ್ತೆ! 🙏 ನಾನು **FarmGenius AI**, ನಿಮ್ಮ ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ಸಹಾಯಕ.\n\nಬೆಳೆ, ರೋಗ, ಹವಾಮಾನ ಅಥವಾ ಮಾರುಕಟ್ಟೆ ಬೆಲೆಯ ಬಗ್ಗೆ ಕೇಳಿ!",
    te: "నమస్తే! 🙏 నేను **FarmGenius AI**, మీ స్మార్ట్ వ్యవసాయ సహాయకుడు.\n\nపంట, వ్యాధి, వాతావరణం లేదా మార్కెట్ ధరల గురించి అడగండి!",
    ta: "நமஸ்தே! 🙏 நான் **FarmGenius AI**, உங்கள் ஸ்மார்ட் விவசாய உதவியாளர்.\n\nபயிர், நோய், வானிலை அல்லது சந்தை விலை பற்றி கேளுங்கள்!",
    mr: "नमस्ते! 🙏 मी **FarmGenius AI**, तुमचा स्मार्ट शेती सहाय्यक.\n\nपीक, रोग, हवामान किंवा बाजारभाव बद्दल काहीही विचारा!",
  };
  return {
    id: 'welcome',
    sender: 'bot',
    timestamp: new Date(),
    text: welcomeText[lang] ?? welcomeText['en'],
    quickReplies: WELCOME_QUICK_REPLIES[lang] ?? WELCOME_QUICK_REPLIES['en'],
  };
}

// ─── Sub-components ───────────────────────────────────────────────────────────

/** Animated 3-dot typing indicator */
function TypingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 4 }}
      transition={{ duration: 0.25 }}
      className="flex items-end gap-2 mb-4"
    >
      <div className="w-8 h-8 rounded-full bg-surface-variant border border-farm-divider flex items-center justify-center flex-shrink-0">
        <Robot size={16} weight="fill" className="text-primary" />
      </div>
      <div className="bubble-bot flex items-center gap-1.5 py-4 px-5">
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            animate={{ y: [0, -5, 0] }}
            transition={{
              duration: 0.7,
              repeat: Infinity,
              delay: i * 0.16,
            }}
            className="w-2 h-2 rounded-full bg-text-secondary inline-block"
          />
        ))}
      </div>
    </motion.div>
  );
}

/** Render bot message text with **bold** markdown support */
function BotText({ text }: { text: string }) {
  const segments = text.split(/(\*\*[^*]+\*\*)/g);
  return (
    <>
      {segments.map((seg, i) => {
        if (seg.startsWith('**') && seg.endsWith('**')) {
          return (
            <strong key={i} className="font-semibold text-text-primary">
              {seg.slice(2, -2)}
            </strong>
          );
        }
        return seg.split('\n').map((line, j, arr) => (
          <span key={`${i}-${j}`}>
            {line}
            {j < arr.length - 1 && <br />}
          </span>
        ));
      })}
    </>
  );
}

interface MessageBubbleProps {
  message: Message;
  onQuickReply: (text: string) => void;
}

function MessageBubble({ message, onQuickReply }: MessageBubbleProps) {
  const isBot = message.sender === 'bot';

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 16, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3 }}
      className={`flex items-end gap-2 mb-4 ${isBot ? 'justify-start' : 'justify-end'}`}
    >
      {/* Bot avatar */}
      {isBot && (
        <div className="w-8 h-8 rounded-full bg-surface-variant border border-farm-divider flex items-center justify-center flex-shrink-0 mb-0.5">
          <Robot size={16} weight="fill" className="text-primary" />
        </div>
      )}

      <div
        className={`flex flex-col gap-1 ${isBot ? 'items-start' : 'items-end'} max-w-[80%] sm:max-w-[70%]`}
      >
        {/* Uploaded image preview */}
        {message.imageUrl && (
          <img
            src={message.imageUrl}
            alt="Uploaded crop"
            className="rounded-md rounded-bl-none max-w-[200px] object-cover border border-farm-divider"
          />
        )}

        {/* Bubble */}
        <div className={isBot ? 'bubble-bot' : 'bubble-user'}>
          {isBot ? (
            <p className="font-noto text-sm leading-relaxed text-text-primary">
              <BotText text={message.text} />
            </p>
          ) : (
            <p className="font-noto text-sm leading-relaxed text-white">
              {message.text}
            </p>
          )}
        </div>

        {/* Timestamp */}
        <span className="text-[10px] font-noto text-text-secondary">
          {formatTime(message.timestamp)}
        </span>

        {/* Quick reply chips */}
        {isBot && message.quickReplies && message.quickReplies.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-1">
            {message.quickReplies.map((qr) => (
              <motion.button
                key={qr}
                whileHover={{ scale: 1.04, boxShadow: '0 4px 12px rgba(46,125,50,0.16)' }}
                whileTap={{ scale: 0.96 }}
                onClick={() => onQuickReply(qr)}
                className="text-xs font-poppins font-semibold text-primary bg-surface-variant
                           border border-primary/20 rounded-full px-3 py-1.5
                           hover:bg-primary hover:text-white transition-all duration-200"
              >
                {qr}
              </motion.button>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

// ─── Speech Recognition helper ────────────────────────────────────────────────


const LANG_TO_BCP47: Record<string, string> = {
  en: 'en-IN',
  hi: 'hi-IN',
  kn: 'kn-IN',
  te: 'te-IN',
  ta: 'ta-IN',
  mr: 'mr-IN',
};

// ─── Main Component ───────────────────────────────────────────────────────────

export default function ChatPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const isOffline = useAppStore((s) => s.isOffline);
  const storedLang = useAppStore((s) => s.language);
  const setLanguage = useAppStore((s) => s.setLanguage);
  const farmer = useAppStore((s) => s.farmer);

  const [messages, setMessages] = useState<Message[]>(() => [buildWelcomeMessage(storedLang ?? 'en')]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [showLangPicker, setShowLangPicker] = useState(false);
  const [selectedLang, setSelectedLang] = useState<string>(storedLang ?? 'en');
  const [isLoading, setIsLoading] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<any>(null);

  // Pre-fill from navigation state (e.g. from ScanPage "Ask AI about this")
  useEffect(() => {
    const state = location.state as { prefill?: string } | null;
    if (state?.prefill) {
      setInputText(state.prefill);
      inputRef.current?.focus();
    }
  }, [location.state]);

  // Scroll to bottom whenever messages or typing state changes
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping, scrollToBottom]);

  // Cleanup speech recognition on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, []);

  // ── Core send logic ───────────────────────────────────────────────────────
  const sendMessage = useCallback(
    async (text: string, imageUrl?: string) => {
      if (!text.trim() && !imageUrl) return;

      const userMsg: Message = {
        id: uid(),
        sender: 'user',
        timestamp: new Date(),
        text: text.trim() || 'I uploaded a photo for diagnosis. 📷',
        imageUrl,
      };

      setMessages((prev) => [...prev, userMsg]);
      setInputText('');
      setIsLoading(true);
      setIsTyping(true);

      let botText = '';

      // BUG 15 FIX: If image is attached, call /disease/detect instead of /chat
      if (imageUrl && !text.trim()) {
        try {
          const response = await fetch(imageUrl);
          const blob = await response.blob();
          const formData = new FormData();
          formData.append('file', blob, 'leaf.jpg');

          const apiRes = await apiClient.post('/disease/detect', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
          });

          const data = apiRes.data;
          const conf = Math.round(data.confidence * 100);
          botText = `🔬 **${data.disease_name}** detected with **${conf}% confidence**.\n\n`;
          if (data.organic_treatment) botText += `🌿 Organic: ${data.organic_treatment}\n`;
          if (data.chemical_treatment) botText += `💊 Chemical: ${data.chemical_treatment}\n`;
          if (data.dosage) botText += `📏 Dosage: ${data.dosage}\n`;
          botText += `\nSource: ${data.source_name}`;
        } catch {
          // Fallback to generic mock disease result when backend 503
          botText = '🔬 **Tomato Late Blight** detected (offline/demo result).\n\n🌿 Organic: Neem oil spray 3ml/L every 7 days.\n💊 Chemical: Mancozeb 75% WP @ 2g/L every 7 days.\n\n⚠️ *This is a demo result. Upload a clear leaf photo for accurate diagnosis.*';
        }
      } else if (!isOffline) {
        try {
          const res = await apiClient.post<{ response: string }>('/chat', {
            query: text,
            language: selectedLang,
            farmer_id: farmer?.phone ?? 'demo-001',
          });
          botText = res.data.response;
        } catch {
          botText = getMockResponse(text);
        }
      } else {
        botText = getMockResponse(text);
      }

      setIsTyping(false);
      setIsLoading(false);

      // Derive contextual quick replies in the right language
      const lang = selectedLang as keyof typeof QUICK_REPLIES_BY_CONTEXT.price;
      const qr =
        text.toLowerCase().includes('price') || text.toLowerCase().includes('mandi') || text.toLowerCase().includes('bhav') || text.toLowerCase().includes('ಬೆಲೆ') || text.toLowerCase().includes('ధర')
          ? QUICK_REPLIES_BY_CONTEXT.price[lang] ?? QUICK_REPLIES_BY_CONTEXT.price['en']
          : text.toLowerCase().includes('disease') || text.toLowerCase().includes('pest') || text.toLowerCase().includes('roga') || text.toLowerCase().includes('ರೋಗ') || text.toLowerCase().includes('వ్యాధి')
          ? QUICK_REPLIES_BY_CONTEXT.disease[lang] ?? QUICK_REPLIES_BY_CONTEXT.disease['en']
          : text.toLowerCase().includes('weather') || text.toLowerCase().includes('rain') || text.toLowerCase().includes('mausam') || text.toLowerCase().includes('ಮಳೆ')
          ? QUICK_REPLIES_BY_CONTEXT.weather[lang] ?? QUICK_REPLIES_BY_CONTEXT.weather['en']
          : imageUrl
          ? QUICK_REPLIES_BY_CONTEXT.disease[lang] ?? QUICK_REPLIES_BY_CONTEXT.disease['en']
          : undefined;

      const botMsg: Message = {
        id: uid(),
        sender: 'bot',
        timestamp: new Date(),
        text: botText,
        quickReplies: qr,
      };

      setMessages((prev) => [...prev, botMsg]);
    },
    [isOffline, selectedLang, farmer]
  );

  const handleSend = useCallback(() => {
    if (inputText.trim() && !isLoading) {
      sendMessage(inputText);
    }
  }, [inputText, isLoading, sendMessage]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleQuickReply = useCallback(
    (text: string) => {
      sendMessage(text);
    },
    [sendMessage]
  );

  // ── Image upload ──────────────────────────────────────────────────────────
  const handleImageUpload = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      sendMessage('', url);
      if (fileInputRef.current) fileInputRef.current.value = '';
    },
    [sendMessage]
  );

  // BUG 8 FIX: Real Web Speech API with BCP-47 language codes
  const toggleRecording = useCallback(() => {
    if (isRecording) {
      recognitionRef.current?.abort();
      recognitionRef.current = null;
      setIsRecording(false);
      return;
    }

    const SpeechRecognitionAPI =
      (window as any).SpeechRecognition ?? (window as any).webkitSpeechRecognition;

    if (!SpeechRecognitionAPI) {
      // Fallback stub for browsers without Web Speech API (e.g. Firefox)
      setIsRecording(true);
      setTimeout(() => {
        setIsRecording(false);
        const stubs: Record<string, string> = {
          hi: 'मेरी गेहूं की पत्ती पीली हो रही है, क्या करूं?',
          kn: 'ನನ್ನ ಟೊಮ್ಯಾಟೊ ಎಲೆಗಳು ಹಳದಿ ಆಗುತ್ತಿವೆ, ಏನು ಮಾಡಲಿ?',
          te: 'నా గోధుమ ఆకులు పసుపు రంగులోకి మారుతున్నాయి, ఏమి చేయాలి?',
          ta: 'என் கோதுமை இலைகள் மஞ்சள் ஆகின்றன, என்ன செய்வது?',
          mr: 'माझ्या गव्हाची पाने पिवळी पडत आहेत, काय करू?',
          en: 'My wheat leaves are turning yellow, what should I do?',
        };
        setInputText(stubs[selectedLang] ?? stubs['en']);
        inputRef.current?.focus();
      }, 3000);
      return;
    }

    const recognition: any = new SpeechRecognitionAPI();
    recognition.lang = LANG_TO_BCP47[selectedLang] ?? 'en-IN';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event: any) => {
      const transcript = event.results[0]?.[0]?.transcript ?? '';
      if (transcript) {
        setInputText(transcript);
        inputRef.current?.focus();
      }
    };

    recognition.onerror = () => {
      setIsRecording(false);
      recognitionRef.current = null;
    };

    recognition.onend = () => {
      setIsRecording(false);
      recognitionRef.current = null;
    };

    recognitionRef.current = recognition;
    recognition.start();
    setIsRecording(true);
  }, [isRecording, selectedLang]);

  // BUG 14 FIX: Update welcome message and lang state when language changes
  const handleLangChange = useCallback((langCode: string) => {
    setSelectedLang(langCode);
    setLanguage(langCode);
    setShowLangPicker(false);
    // If no messages beyond welcome, rebuild welcome in new language
    setMessages((prev) => {
      if (prev.length === 1 && prev[0].id === 'welcome') {
        return [buildWelcomeMessage(langCode)];
      }
      return prev;
    });
  }, [setLanguage]);

  const currentLang = LANGUAGES.find((l) => l.code === selectedLang);

  return (
    <div
      className="flex flex-col bg-bg"
      style={{ height: 'calc(100vh - 4rem)' }}
    >
      {/* ── AppBar ─────────────────────────────────────────────────────── */}
      <motion.div
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.35 }}
        className="flex items-center gap-3 px-4 py-3 bg-white border-b border-farm-divider
                   shadow-sm flex-shrink-0 z-10"
      >
        {/* Back */}
        <motion.button
          whileHover={{ scale: 1.08 }}
          whileTap={{ scale: 0.92 }}
          onClick={() => navigate('/dashboard')}
          className="w-9 h-9 rounded-full hover:bg-surface-variant transition-colors
                     flex items-center justify-center"
          aria-label="Go back"
        >
          <ArrowLeft size={20} weight="bold" className="text-text-primary" />
        </motion.button>

        {/* Avatar + title */}
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <div className="w-9 h-9 rounded-full bg-gradient-primary flex items-center justify-center flex-shrink-0">
            <Robot size={20} weight="fill" className="text-white" />
          </div>
          <div className="min-w-0">
            <h1 className="font-poppins font-bold text-base text-text-primary leading-tight truncate">
              FarmGenius AI
            </h1>
            <p className="text-[11px] font-noto text-text-secondary flex items-center gap-1">
              {isOffline ? (
                <>
                  <WifiX size={10} weight="fill" className="text-farm-error" />
                  <span>Offline mode</span>
                </>
              ) : (
                <>
                  <motion.span
                    animate={{ opacity: [1, 0.3, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                    className="w-1.5 h-1.5 rounded-full bg-farm-success inline-block"
                  />
                  <WifiHigh size={10} weight="fill" className="text-farm-success" />
                  <span>Online</span>
                </>
              )}
            </p>
          </div>
        </div>

        {/* Language selector */}
        <div className="relative">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowLangPicker((v) => !v)}
            className="flex items-center gap-1.5 bg-surface-variant rounded-full px-3 py-1.5
                       font-poppins font-semibold text-xs text-primary border border-primary/20
                       hover:bg-primary hover:text-white transition-all duration-200"
            aria-label="Select language"
          >
            <Translate size={14} />
            <span>{currentLang?.nativeLabel ?? 'EN'}</span>
          </motion.button>

          <AnimatePresence>
            {showLangPicker && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9, y: -4 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9, y: -4 }}
                transition={{ duration: 0.18 }}
                className="absolute right-0 top-full mt-2 bg-white rounded-md shadow-card-hover
                           border border-farm-divider z-50 overflow-hidden min-w-[140px]"
              >
                {LANGUAGES.map((lang) => (
                  <button
                    key={lang.code}
                    onClick={() => handleLangChange(lang.code)}
                    className={`w-full flex items-center justify-between px-4 py-2.5 text-sm font-noto
                                hover:bg-surface-variant transition-colors ${
                                  selectedLang === lang.code
                                    ? 'bg-surface-variant text-primary font-semibold'
                                    : 'text-text-primary'
                                }`}
                  >
                    <span>{lang.nativeLabel}</span>
                    {selectedLang === lang.code && (
                      <span className="text-primary text-xs font-poppins font-bold">✓</span>
                    )}
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>

      {/* ── Messages List ──────────────────────────────────────────────── */}
      <div
        className="flex-1 overflow-y-auto px-4 py-4"
        onClick={() => setShowLangPicker(false)}
      >
        <div className="max-w-2xl mx-auto">
          {/* Date chip */}
          <div className="flex items-center gap-3 mb-6">
            <div className="flex-1 h-px bg-farm-divider" />
            <span className="text-xs font-noto text-text-secondary px-2 flex-shrink-0">
              Today
            </span>
            <div className="flex-1 h-px bg-farm-divider" />
          </div>

          {/* Message bubbles */}
          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                onQuickReply={handleQuickReply}
              />
            ))}
          </AnimatePresence>

          {/* Typing indicator */}
          <AnimatePresence>{isTyping && <TypingIndicator key="typing" />}</AnimatePresence>

          {/* Scroll anchor */}
          <div ref={messagesEndRef} className="h-2" />
        </div>
      </div>

      {/* ── Recording Indicator Banner ─────────────────────────────────── */}
      <AnimatePresence>
        {isRecording && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="flex items-center justify-center gap-3 py-2 bg-red-50 border-t border-red-200 flex-shrink-0"
          >
            <motion.div
              animate={{ scale: [1, 1.3, 1], opacity: [1, 0.5, 1] }}
              transition={{ duration: 1.1, repeat: Infinity }}
              className="w-3 h-3 rounded-full bg-farm-error"
            />
            <span className="font-noto text-sm text-farm-error font-semibold">
              Listening… Speak now
            </span>
            <button
              onClick={toggleRecording}
              className="text-farm-error text-xs font-poppins font-bold underline"
            >
              Cancel
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Input Bar ──────────────────────────────────────────────────── */}
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.35, delay: 0.1 }}
        className="flex-shrink-0 bg-white border-t border-farm-divider px-3 py-3
                   shadow-[0_-4px_12px_rgba(0,0,0,0.05)]"
      >
        <div className="max-w-2xl mx-auto flex items-end gap-2">

          {/* Image attach */}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            className="w-10 h-10 rounded-full bg-surface-variant flex items-center justify-center
                       text-text-secondary hover:bg-primary/10 hover:text-primary
                       transition-all duration-200 flex-shrink-0 disabled:opacity-40"
            aria-label="Attach image"
          >
            <ImageIcon size={20} weight="duotone" />
          </motion.button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={handleImageUpload}
          />

          {/* Text area */}
          <div className="flex-1">
            <textarea
              ref={inputRef}
              value={inputText}
              onChange={(e: ChangeEvent<HTMLTextAreaElement>) => {
                setInputText(e.target.value);
                e.target.style.height = 'auto';
                e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
              }}
              onKeyDown={handleKeyDown}
              disabled={isLoading || isRecording}
              placeholder={
                isRecording
                  ? 'Listening...'
                  : 'Ask anything about your farm...'
              }
              rows={1}
              className="input-field py-2.5 px-4 resize-none overflow-hidden leading-relaxed
                         text-sm rounded-full min-h-[44px] max-h-[120px] disabled:opacity-60"
              style={{ height: '44px' }}
            />
          </div>

          {/* Voice button */}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={toggleRecording}
            disabled={isLoading}
            className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0
                        transition-all duration-200 disabled:opacity-40
                        ${
                          isRecording
                            ? 'bg-farm-error text-white shadow-[0_0_0_4px_rgba(198,40,40,0.2)]'
                            : 'bg-surface-variant text-text-secondary hover:bg-primary/10 hover:text-primary'
                        }`}
            aria-label={isRecording ? 'Stop recording' : 'Start voice input'}
          >
            {isRecording ? (
              <MicrophoneSlash size={20} weight="fill" />
            ) : (
              <Microphone size={20} weight="duotone" />
            )}
          </motion.button>

          {/* Send button */}
          <motion.button
            whileHover={inputText.trim() ? { scale: 1.08 } : {}}
            whileTap={inputText.trim() ? { scale: 0.92 } : {}}
            onClick={handleSend}
            disabled={!inputText.trim() || isLoading}
            className="w-11 h-11 rounded-full bg-primary flex items-center justify-center
                       flex-shrink-0 shadow-card hover:bg-primary-dark transition-all duration-200
                       disabled:opacity-40 disabled:cursor-not-allowed"
            aria-label="Send message"
          >
            <AnimatePresence mode="wait">
              {isLoading ? (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0, scale: 0.7 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.7 }}
                  className="w-4 h-4 rounded-full border-2 border-white/40 border-t-white animate-spin"
                />
              ) : (
                <motion.div
                  key="send"
                  initial={{ opacity: 0, scale: 0.7 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.7 }}
                >
                  <PaperPlaneRight size={20} weight="fill" className="text-white" />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.button>
        </div>

        {/* AI disclaimer */}
        <p className="text-center text-[10px] font-noto text-text-secondary mt-2 max-w-2xl mx-auto">
          AI responses are advisory only. For critical decisions, consult your local KVK agronomist.
        </p>
      </motion.div>
    </div>
  );
}
