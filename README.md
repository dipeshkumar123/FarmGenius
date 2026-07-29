# FarmGenius 🌾

An AI-powered agricultural assistant platform providing crop recommendations, plant disease detection, weather analysis, commodity price tracking, government scheme recommendations, and an intelligent multilingual chatbot.

## Technology Stack

| Layer            | Technology                                   |
|------------------|-----------------------------------------------|
| **Frontend**     | React 19, Vite, TypeScript, Tailwind CSS      |
| **Backend**      | Python 3, FastAPI, SQLAlchemy                 |
| **ML Inference** | ONNX Runtime                                  |
| **Database**     | PostgreSQL, SQLite                            |
| **Deployment**   | Vercel                                        |

## Key Features

- **Smart Chat** — NLP chatbot with intent routing to specialized ML models
- **Crop Advisor** — Soil-parameter-based crop recommendations
- **Disease Detection** — Image-based plant disease identification using ONNX models
- **Weather** — Current conditions, forecasts, crop-specific advice
- **Prices** — Commodity price lookup, trends, and analytics
- **Schemes** — Government scheme recommendations
- **Multilingual** — Support for English, Hindi, Kannada, Marathi, Tamil, Telugu
- **History** — Per-user conversation and query history

## Architecture Overview

The system is composed of a React frontend and a FastAPI backend, deployed together on Vercel:

```
┌────────────────────────────────────────────────────────────┐
│                         Vercel                              │
│  ┌─────────────┐                           ┌─────────────┐ │
│  │  Frontend   │──────────────┐            │  FastAPI    │ │
│  │ React SPA   │              │            │  Serverless │ │
│  │ (Vite CDN)  │              │            │ (ONNX infer)│ │
│  └─────────────┘              │            └──────┬──────┘ │
│                                │                   │        │
│                                │           ┌───────▼───────┐ │
│                                │           │   Neon /      │ │
│                                │           │ SQLite DB     │ │
│                                │           └───────────────┘ │
│                                │                           │
│                     ┌──────────▼──────────┐               │
│                     │  Alembic Migrations │               │
│                     └─────────────────────┘               │
└────────────────────────────────────────────────────────────┘
```

- **Frontend** serves the UI and proxies API calls to the backend
- **Backend** handles routing, middleware, and module orchestration
- **Database** stores user history, conversation logs, and app state
- **ONNX Runtime** executes ML models for fast inference

## Challenges Overcome

- **Serverless ONNX inference**: Converted large TensorFlow models (~1.3 GB) into lightweight ONNX formats (~30 MB) for Vercel compatibility
- **Multilingual support**: Integrated translation workflows for 10+ languages using Google Translate
- **Environment parity**: Maintained SQLite for local development and PostgreSQL for production without code changes
- **Frontend modernization**: Migrated from vanilla HTML/CSS/JS to a React SPA while keeping Vercel edge compatibility

## Running Locally

### Prerequisites

- [Python](https://www.python.org/downloads/) 3.10+
- [Node.js](https://nodejs.org/) 18+
- [Git](https://git-scm.com/downloads/)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/dipeshkumar123/FarmGenius.git
cd FarmGenius

# 2. Set up backend
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
cp .env.example .env

# 3. Start backend
python -m src.api.app
# Backend runs at http://localhost:8000

# 4. In a new terminal, start frontend
cd frontend
npm install
npm run dev
# Frontend runs at http://localhost:5173
```

### Access the Application

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

## Project Structure

```
FarmGenius/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI routes
│   │   ├── models/           # ORM models
│   │   ├── core/             # Config, middleware
│   │   └── services/         # Business logic
│   ├── requirements.txt
│   └── vercel.json           # Serverless config
├── frontend/
│   ├── src/
│   │   ├── api/              # API client
│   │   ├── components/       # Reusable components
│   │   ├── pages/            # Route pages
│   │   ├── store/            # Zustand state
│   │   └── locales/          # i18n translations
│   ├── package.json
│   └── vite.config.ts
├── data/
│   └── disease_database.json
├── models/                   # ONNX / Pickle model files
├── public/                   # Built frontend assets
├── scripts/                  # Utility scripts
├── src/
│   └── api/                  # Local backend app entrypoint
├── alembic/                  # Database migrations
├── requirements.txt          # Python dependencies
├── vercel.json               # Vercel deployment config
└── pyproject.toml            # Project metadata
```

## Deployment

Push to main to auto-deploy via Vercel. Set environment variables in the Vercel dashboard for database connection, LLM API keys, and weather APIs. The frontend and backend are deployed together on Vercel.