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

- Smart chat with intent routing to specialized ML models
- Soil-parameter-based crop recommendations
- Image-based plant disease identification
- Weather forecasting with crop-specific advice
- Commodity price lookup and trend analysis
- Government schemes recommendation
- Multilingual support: English, Hindi, Kannada, Marathi, Tamil, Telugu
- Conversation and query history

## Architecture Overview

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

┌────────────────────────────────────────────────────────────┐
│                  Local Development                          │
│  Backend:  python -m src.api.app                           │
│  Frontend: npm run dev                                     │
└────────────────────────────────────────────────────────────┘
```

- **Frontend** serves a React UI that communicates with the backend API
- **Backend** handles routing, middleware, and module orchestration
- **Database** stores user history, conversation logs, and app state
- **ONNX Runtime** executes ML models for fast inference

## Challenges Overcome

- **Serverless ONNX inference**: Converted large TensorFlow models (~1.3 GB) into lightweight ONNX formats (~30 MB) for Vercel compatibility
- **Multilingual support**: Integrated translation workflows for 10+ languages using Google Translate
- **Environment parity**: Maintained SQLite for local development and PostgreSQL for production without code changes
- **Frontend bundling**: Switched from a vanilla HTML/CSS/JS frontend to a modern React SPA while keeping Vercel edge compatibility

## Running Locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### Backend

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
cp .env.example .env

python -m src.api.app
```

Backend runs at http://localhost:8000

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at http://localhost:5173

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