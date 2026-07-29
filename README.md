# FarmGenius

AI-powered agricultural assistant platform providing crop recommendations, plant disease detection, weather analysis, commodity price tracking, government scheme recommendations, and an intelligent multilingual chatbot.

## Features

- Smart chat with intent routing to specialized ML models
- Crop recommendations based on soil parameters
- Image-based plant disease identification
- Weather forecasting with crop-specific advice
- Commodity price lookup and trend analysis
- Government schemes recommendation
- Multilingual support
- Conversation and query history

## Tech Stack

- **Backend:** Python 3, FastAPI, SQLAlchemy
- **ML:** ONNX Runtime
- **Database:** PostgreSQL / SQLite
- **Frontend:** React 19, Vite, TypeScript, Tailwind CSS
- **Deployment:** Vercel

## Process

The system uses a backend-focused architecture where user queries are parsed and routed to specialized handlers. A FastAPI app exposes endpoints for crop, weather, price, disease, and scheme modules. The frontend is a React SPA that communicates with these endpoints. ONNX models keep inference lightweight and compatible with serverless deployment.

## Running Locally

### Backend

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
cp .env.example .env

python -m src.api.app
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Deployment

Push to main to auto-deploy. Set environment variables in the deployment dashboard for database connection, LLM API keys, and weather APIs.