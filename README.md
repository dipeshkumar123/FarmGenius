# FarmGenius 🌾

AI-powered agricultural assistant providing crop recommendations, plant disease detection, weather analysis, commodity price tracking, and an intelligent chatbot — all in one platform.

## Features

| Feature | Description |
|---------|-------------|
| **Smart Chat** | NLP chatbot with intent routing to specialized ML models |
| **Crop Advisor** | Soil-parameter-based crop recommendations (scikit-learn) |
| **Disease Detection** | Image-based plant disease ID (MobileNetV2 / ONNX Runtime) |
| **Weather** | Current conditions, forecasts, crop-specific advice |
| **Prices** | Commodity price lookup, trends, and analytics |
| **Multilingual** | 10 languages via Google Translate |
| **History** | Per-user conversation & query history (PostgreSQL) |

## Architecture

```
┌──────────────────────────────────────────────┐
│                   Vercel                      │
│  ┌─────────────┐   ┌──────────────────────┐  │
│  │  public/     │   │  FastAPI Serverless  │  │
│  │  (CDN)       │   │  (ONNX inference)    │  │
│  │  HTML/CSS/JS │   │  src/api/app.py      │  │
│  └─────────────┘   └──────────┬───────────┘  │
│                                │              │
│                     ┌──────────▼───────────┐  │
│                     │  Neon PostgreSQL      │  │
│                     │  (Vercel Marketplace) │  │
│                     └──────────────────────┘  │
└──────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│               Local Development               │
│  TensorFlow training → ONNX conversion        │
│  → git push → Vercel auto-deploy              │
└──────────────────────────────────────────────┘
```

## Quick Start (Local Development)

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USER/FarmGenius.git
cd FarmGenius
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Environment Variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Initialize Database & Run

```bash
# Uses SQLite locally by default (no DATABASE_URL needed)
python -m src.api.app
# → http://localhost:8000
```

## Deploy to Vercel

### Prerequisites
- [Vercel account](https://vercel.com)
- [GitHub repo](https://github.com) connected to Vercel
- Neon PostgreSQL (add via Vercel → Storage → Create → Neon)

### Steps

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USER/FarmGenius.git
   git push -u origin main
   ```

2. **Connect to Vercel:**
   - Go to [vercel.com/new](https://vercel.com/new)
   - Import your GitHub repo
   - Vercel auto-detects `vercel.json` config

3. **Add Neon PostgreSQL:**
   - Vercel Dashboard → Storage → Create → Neon Postgres
   - This auto-sets `DATABASE_URL` in your project env vars

4. **Set Environment Variables** in Vercel Dashboard → Settings → Environment Variables:
   - `DEEPSEEK_API_KEY` — for LLM fallback
   - `WEATHER_API_KEY` — for OpenWeatherMap

5. **Deploy:**
   Every push to `main` triggers automatic deployment.

### GitHub Actions CI/CD

Add these secrets to your GitHub repo (Settings → Secrets → Actions):
- `VERCEL_TOKEN` — from [vercel.com/account/tokens](https://vercel.com/account/tokens)
- `VERCEL_ORG_ID` — from `.vercel/project.json` after first deploy
- `VERCEL_PROJECT_ID` — from `.vercel/project.json` after first deploy

The CI pipeline (`.github/workflows/ci.yml`) runs lint + tests on every PR, and deploys to production on push to `main`.

## ML Model Training (Local Only)

Training uses TensorFlow (~1.3 GB) and runs locally. After training, models are converted to ONNX format (~30 MB) for lightweight cloud deployment.

### Train Disease Model
```bash
python train_disease_model.py
```

### Convert to ONNX
```bash
python scripts/convert_to_onnx.py
```

### Retrain Chatbot Models
```bash
python train_chatbot_models.py
```

After conversion, commit the `models/` directory and push to deploy the updated models.

## Database Migrations

```bash
# Generate a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```

## Project Structure

```
FarmGenius/
├── .github/workflows/ci.yml   # CI/CD pipeline
├── alembic/                    # Database migrations
├── data/                       # Training data, disease DB
├── models/                     # Trained model files (.onnx, .pkl)
├── public/                     # Frontend (Vercel CDN)
│   ├── index.html
│   ├── css/
│   └── js/
├── scripts/                    # Utility scripts
│   └── convert_to_onnx.py
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI application
│   │   └── routes.py           # API endpoints
│   ├── database/
│   │   ├── connection.py       # SQLAlchemy engine
│   │   ├── models.py           # ORM models
│   │   └── crud.py             # CRUD operations
│   ├── models/
│   │   ├── chatbot.py          # Main chatbot orchestrator
│   │   ├── crop_model.py       # Crop recommendation
│   │   ├── disease_model.py    # Disease detection (ONNX/TF)
│   │   ├── price_model.py      # Price analytics
│   │   ├── weather_model.py    # Weather forecasting
│   │   └── user_model.py       # User management
│   └── utils/
├── tests/                      # Test suite
├── vercel.json                 # Vercel configuration
├── requirements.txt            # Full dependencies (local)
├── requirements-deploy.txt     # Lightweight deps (deploy)
└── pyproject.toml              # Project metadata
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/query` | Process natural language query |
| POST | `/api/crops/recommend` | Get crop recommendations |
| POST | `/api/prices/get` | Get commodity prices |
| POST | `/api/prices/trends` | Get price trends |
| GET | `/api/prices/commodities` | List commodities |
| POST | `/api/weather/current` | Current weather |
| POST | `/api/weather/forecast` | Weather forecast |
| POST | `/api/weather/crop-advice` | Crop weather advice |
| POST | `/api/diseases/identify-image` | Disease detection from image |
| POST | `/api/history` | Get conversation history |
| GET | `/api/health` | Health check |
| POST | `/api/translate` | Translate text |
| GET | `/api/languages` | Supported languages |

## Tech Stack

- **Backend:** Python 3.13, FastAPI, SQLAlchemy
- **ML Inference:** ONNX Runtime (deploy), TensorFlow (local training)
- **Database:** PostgreSQL (Neon) / SQLite (local dev)
- **Frontend:** Vanilla HTML/CSS/JS SPA
- **Deployment:** Vercel (serverless + CDN)
- **CI/CD:** GitHub Actions

## License

MIT
