# FarmGenius 🌾

An AI-powered agricultural assistant platform that combines crop recommendations, plant disease detection, weather analytics, commodity price tracking, government scheme discovery, and an intelligent multilingual chatbot — all in one unified system.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, Vite, TypeScript, Tailwind CSS, Framer Motion, Recharts, Zustand, React Query |
| **Backend** | Python 3.12+, FastAPI, SQLAlchemy, Alembic |
| **ML Inference** | ONNX Runtime, TensorFlow, scikit-learn, RandomForest, MobileNetV2 |
| **LLM Integration** | DeepSeek via OpenRouter API |
| **Database** | PostgreSQL 16 (Neon), SQLite (local dev) |
| **Deployment** | Vercel (frontend + backend serverless) |
| **Infrastructure** | Docker, Docker Compose |

## Key Features

- **Smart Chat** — NLP-powered chatbot with intent routing to specialized ML models (crop, weather, price, disease, FAQ) with DeepSeek fallback for complex queries
- **Crop Advisor** — ML-based crop recommendations from soil parameters (N, P, K, pH, EC, micronutrients) using RandomForest with GridSearchCV optimization
- **Disease Detection** — Image-based plant disease identification using ONNX-optimized MobileNetV2 (22 disease classes across Cashew, Cassava, Maize, Tomato) with detailed treatment recommendations
- **Weather Intelligence** — Current conditions, 7-day forecasts, and crop-specific weather advisories with live API support (WeatherAPI.com) and synthetic fallback
- **Market Prices** — Commodity price lookup, 30-day trends, and analytics with data.gov.in API integration and synthetic fallback for 20+ commodities
- **Government Schemes** — Discover and check eligibility for agricultural schemes and subsidies
- **Multilingual** — Full i18n support for English, Hindi, Kannada, Marathi, Tamil, Telugu with Google Translate integration for 10+ languages
- **Voice Interaction** — Speech-to-text and text-to-speech using Google Speech Recognition and gTTS
- **User System** — JWT-based authentication, per-user chat history, preferences, and activity statistics
- **Offline Resilience** — Graceful degradation with cached data when APIs are unavailable

## Architecture Overview

The system is composed of a React SPA frontend and a FastAPI backend, deployed together on Vercel:

```
┌──────────────────────────────────────────────────────────────────┐
│                          Vercel                                   │
│  ┌─────────────────────┐              ┌────────────────────────┐ │
│  │     Frontend        │              │      Backend           │ │
│  │   React 19 SPA      │──────────────│   FastAPI Serverless   │ │
│  │   (Vite CDN)        │  /api/*      │   (Python 3.12)       │ │
│  │                     │              │                        │ │
│  │  ┌───────────────┐  │              │  ┌──────────────────┐  │ │
│  │  │  Pages:       │  │              │  │  Services:       │  │ │
│  │  │  • Dashboard  │  │              │  │  • Chatbot       │  │ │
│  │  │  • Chat       │  │              │  │  • Crop Model    │  │ │
│  │  │  • Scan       │  │              │  │  • Disease Model │  │ │
│  │  │  • Market     │  │              │  │  • Weather Model │  │ │
│  │  │  • Weather    │  │              │  │  • Price Model   │  │ │
│  │  │  • Crops      │  │              │  │  • Language      │  │ │
│  │  │  • Schemes    │  │              │  │  • Voice         │  │ │
│  │  │  • Profile    │  │              │  │  • DeepSeek/LLM  │  │ │
│  │  └───────────────┘  │              │  └──────────────────┘  │ │
│  │                     │              │                        │ │
│  │  ┌───────────────┐  │              │  ┌──────────────────┐  │ │
│  │  │  State:       │  │              │  │  ML Models:      │  │ │
│  │  │  • Zustand    │  │              │  │  • ONNX Runtime  │  │ │
│  │  │  • React Query│  │              │  │  • TensorFlow    │  │ │
│  │  └───────────────┘  │              │  │  • scikit-learn  │  │ │
│  │                     │              │  └──────────────────┘  │ │
│  │  ┌───────────────┐  │              │                        │ │
│  │  │  i18n:        │  │              │  ┌──────────────────┐  │ │
│  │  │  EN, HI, KN,  │  │              │  │  Database:      │  │ │
│  │  │  MR, TA, TE   │  │              │  │  • PostgreSQL   │  │ │
│  │  └───────────────┘  │              │  │  • SQLite (dev) │  │ │
│  └─────────────────────┘              │  └──────────────────┘  │ │
│                                        └────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

- **Frontend** serves the React SPA and proxies `/api/*` requests to the backend
- **Backend** handles business logic, ML inference, database operations, and external API integrations
- **Database** stores user profiles, chat history, and query logs (PostgreSQL in production, SQLite locally)
- **ML Models** are stored as ONNX (lightweight, ~30 MB) and pickle files for fast serverless inference

## Challenges Overcome

- **Serverless ONNX inference**: Converted large TensorFlow models (~1.3 GB) into lightweight ONNX formats (~30 MB) for Vercel's serverless environment compatibility
- **Multilingual support**: Integrated Google Translate with language detection for 10+ languages, with frontend i18n using i18next for 6 Indian languages
- **Environment parity**: Maintained SQLite for local development and PostgreSQL for production with zero code changes using SQLAlchemy's abstraction layer
- **Graceful degradation**: All external services (weather, prices, DeepSeek) have synthetic fallback data generators, ensuring the app works fully offline
- **Intent routing**: Built a custom NLP intent detection system that routes queries to specialized ML models (crop, weather, price, disease, FAQ) with DeepSeek as a smart fallback
- **Frontend modernization**: Migrated from vanilla HTML/CSS/JS to a full React 19 SPA with TypeScript, state management, and modern tooling
- **Dual backend architecture**: Maintained both a modern FastAPI backend (`src/`) and a legacy backend (`backend/`) for Android app compatibility

## Running Locally

### Prerequisites

- [Python](https://www.python.org/downloads/) 3.10+
- [Node.js](https://nodejs.org/) 18+
- [Git](https://git-scm.com/downloads/)

### Quick Start (Full Stack)

```bash
# 1. Clone the repository
git clone https://github.com/dipeshkumar123/FarmGenius.git
cd FarmGenius

# 2. Set up Python virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Copy environment file and configure
cp .env.example .env
# Edit .env with your API keys (optional — app works with synthetic defaults)

# 5. Start the backend
python -m src.app
# Backend runs at http://localhost:8000

# 6. In a new terminal, start the frontend
cd frontend
npm install
npm run dev
# Frontend runs at http://localhost:5173
```

### Quick Start (PowerShell — Windows)

```powershell
# Run both backend and frontend with a single command
.\run_all.ps1
```

### Access the Application

| Service | URL |
|---------|-----|
| Frontend (React SPA) | http://localhost:5173 |
| Backend API (FastAPI) | http://localhost:8000 |
| API Documentation (Swagger) | http://localhost:8000/docs |
| API Documentation (ReDoc) | http://localhost:8000/redoc |

### Environment Variables

Create a `.env` file in the project root:

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | No | PostgreSQL connection string (defaults to SQLite) |
| `DEEPSEEK_API_KEY` | No | OpenRouter API key for DeepSeek LLM |
| `WEATHER_API_KEY` | No | WeatherAPI.com API key for live weather |
| `PRICE_API_KEY` | No | data.gov.in API key for live commodity prices |
| `USE_LIVE_WEATHER_DATA` | No | Set to `True` to enable live weather API |
| `USE_LIVE_PRICE_DATA` | No | Set to `True` to enable live price API |
| `API_HOST` | No | Backend host (default: `0.0.0.0`) |
| `API_PORT` | No | Backend port (default: `8000`) |
| `DEBUG` | No | Set to `True` for hot reload |

## Project Structure

```
FarmGenius/
├── backend/                    # Legacy backend (Android app API)
│   ├── app/
│   │   ├── api/routes/         # FastAPI route modules
│   │   ├── core/               # Config, security, middleware
│   │   ├── models/             # Pydantic schemas
│   │   ├── services/           # Business logic services
│   │   └── ml_models/          # ONNX model files
│   ├── main.py                 # Legacy backend entry point
│   ├── Dockerfile              # Docker build for backend
│   └── vercel.json             # Vercel serverless config
├── frontend/                   # React SPA
│   ├── src/
│   │   ├── api/                # API client (axios)
│   │   ├── components/         # Reusable UI components
│   │   │   ├── layout/         # App shell, navigation
│   │   │   └── ui/             # Cards, banners, loaders
│   │   ├── pages/              # Route pages (10 pages)
│   │   ├── store/              # Zustand state management
│   │   ├── locales/            # i18n translations (6 languages)
│   │   └── utils/              # Localization helpers
│   ├── package.json
│   └── vite.config.ts
├── src/                        # Main backend (FastAPI)
│   ├── app.py                  # FastAPI application entry point
│   ├── routes.py               # API route definitions
│   ├── services/               # ML model services
│   │   ├── chatbot.py          # Main chatbot orchestrator
│   │   ├── crop_model.py       # Crop recommendation (RandomForest)
│   │   ├── disease_model.py    # Disease detection (ONNX/TF)
│   │   ├── weather_model.py    # Weather data & forecasts
│   │   ├── price_model.py      # Commodity prices & trends
│   │   ├── faq_model.py        # FAQ semantic search
│   │   ├── language_model.py   # Translation & detection
│   │   ├── voice_model.py      # Speech-to-text & TTS
│   │   ├── deepseek_model.py   # DeepSeek LLM integration
│   │   └── user_model.py       # User management
│   ├── db/                     # Database layer
│   │   ├── connection.py       # SQLAlchemy engine & session
│   │   ├── models.py           # ORM models (User, ChatHistory, QueryLog)
│   │   └── crud.py             # CRUD operations
│   └── utils/                  # Utilities
├── models/                     # Trained ML model files
│   ├── crop_recommendation_model.pkl
│   ├── disease_model.onnx      # ONNX-optimized (30 MB)
│   ├── disease_model.h5        # TensorFlow (1.3 GB)
│   ├── disease_class_map.json
│   └── faq_embeddings.pkl
├── data/                       # Data files
│   ├── disease_database.json   # Disease info database
│   ├── history/                # User history storage
│   └── languages/              # Translation caches
├── scripts/                    # Utility scripts
│   ├── train_disease_model.py
│   ├── train_chatbot_models.py
│   ├── convert_to_onnx.py
│   └── verify_models.py
├── alembic/                    # Database migrations
├── public/                     # Built frontend assets
├── requirements.txt            # Python dependencies
├── vercel.json                 # Vercel deployment config
├── pyproject.toml              # Project metadata
└── run_all.ps1                 # PowerShell launcher
```

## API Overview

All API endpoints are prefixed with `/api`. The API is fully documented at `/docs` (Swagger) and `/redoc` (ReDoc) when the server is running.

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Register a new user account |
| `/api/auth/login` | POST | Login with email & password |
| `/api/auth/me` | GET | Get current user profile (JWT required) |
| `/api/auth/profile` | PUT | Update user profile (JWT required) |
| `/api/my/history` | GET | Get authenticated user's chat history |
| `/api/my/history` | DELETE | Clear authenticated user's chat history |
| `/api/my/stats` | GET | Get authenticated user's activity statistics |

### Chat

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Process a natural language query with intent routing |

### Crops

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/crops/recommend` | POST | Get crop recommendations from soil parameters (N, P, K, pH, EC, S, Cu, Fe, Mn, Zn, B) |

### Weather

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/weather/current` | POST | Get current weather for a location |
| `/api/weather/forecast` | POST | Get weather forecast (1-7 days) |
| `/api/weather/crop-advice` | POST | Get crop-specific weather advice |

### Prices

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/prices/get` | POST | Get commodity price with optional trends |
| `/api/prices/commodities` | GET | List all available commodities |
| `/api/prices/trends` | POST | Get price trends with analysis |

### Diseases

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/diseases/identify-image` | POST | Identify plant disease from uploaded image |

### Language

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/languages` | GET | List supported languages |
| `/api/languages/detect` | POST | Detect language of input text |
| `/api/translate` | POST | Translate text between languages |

### Users

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/history` | POST | Get conversation history for a user |
| `/api/users/{id}/preferences` | GET/PUT | Get/update user preferences |
| `/api/users/{id}/statistics` | GET | Get user usage statistics |
| `/api/users/{id}/price-alerts` | POST | Add a price alert |
| `/api/users/{id}/weather-alerts` | POST | Add a weather alert |

### System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check endpoint |
| `/` | GET | Serve frontend SPA |

## Deployment

### Vercel (Recommended)

The project is configured for automatic deployment to Vercel via the `vercel.json` at the project root:

```json
{
  "builds": [
    { "src": "backend/main.py", "use": "@vercel/python" },
    { "src": "frontend/package.json", "use": "@vercel/static-build", "config": { "distDir": "dist" } }
  ],
  "routes": [
    { "src": "/api/(.*)", "dest": "/backend/main.py" },
    { "src": "/assets/(.*)", "dest": "/frontend/assets/$1" },
    { "src": "/(.*)", "dest": "/frontend/index.html" }
  ]
}
```

1. Push the repository to GitHub
2. In Vercel Dashboard, click **"Add New Project"**
3. Import your GitHub repository
4. Vercel automatically detects the monorepo configuration
5. Add environment variables in the Vercel dashboard:
   - `DATABASE_URL` — Your Neon PostgreSQL connection string
   - `DEEPSEEK_API_KEY` — OpenRouter API key (optional)
   - `WEATHER_API_KEY` — WeatherAPI.com key (optional)
   - `PRICE_API_KEY` — data.gov.in key (optional)
6. Deploy

### Docker

A Dockerfile is available in the `backend/` directory for containerized deployment:

```bash
cd backend
docker build -t farmgenius-backend .
docker run -p 8000:8000 farmgenius-backend
```

## License

MIT

