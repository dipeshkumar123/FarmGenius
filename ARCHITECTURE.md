# FarmGenius System Architecture Specification

## 1. FLUTTER APP ARCHITECTURE

### Constraints & Principles
- **Android Only:** Developed using Flutter targeting Android.
- **Offline-First:** Critical features like disease detection must work without an internet connection.
- **Low Connectivity:** Must handle 2G network timeouts (30s) gracefully and cache aggressively.
- **Voice-First UX:** Minimal typing required from the farmer.

### Folder Structure (Feature-First)
```
app/lib/
├── main.dart
├── core/                  # App-wide configurations, themes, error handling
│   ├── network/           # API client, interceptors
│   ├── storage/           # Hive initialization and abstract storage classes
│   └── utils/             # Constants, helpers
├── features/
│   ├── auth/              # Phone OTP login flow
│   ├── chat/              # Voice/Text chat interface
│   ├── disease/           # Camera, TFLite inference, results
│   ├── market/            # Mandi prices display
│   └── weather/           # 7-day forecast
└── shared/                # Shared widgets, models
```

### State Management: Riverpod
We chose **Riverpod** over Bloc for the following reasons:
- **Less Boilerplate:** Single developer maintainability is crucial. Riverpod reduces the boilerplate needed for state management.
- **Built-in Caching:** Riverpod's `AsyncValue` makes it exceptionally easy to handle loading, error, and cached states, which is vital for our 2G connectivity constraints.
- **Compile-Time Safety:** Prevents runtime errors associated with context lookups.

### Navigation: GoRouter
We use **GoRouter** for declarative routing. It simplifies deep linking (useful for WhatsApp sharing) and navigation state handling.

### Offline-First Data Flow (Hive)
**Hive** (a lightweight NoSQL local database) is used for offline caching. 
- **Read Path:** The UI always observes the Riverpod provider. The provider first reads from the local Hive box. If stale or empty, it triggers a network request.
- **Write Path:** Network responses are immediately written to the Hive box, triggering a UI update.

### TFLite Integration
- The quantized `.tflite` model and `labels.txt` are bundled in `assets/models/`.
- The `tflite_flutter` package is used to load the model.
- When an image is captured, it is resized to the model's input dimension (e.g., 224x224), converted to a tensor, and passed to the local interpreter.

### Voice-to-Response Pipeline
1. **Input:** `speech_to_text` listens to the farmer in their chosen regional language.
2. **Processing:** The recognized text string + language ID is sent to the backend `POST /chat` endpoint.
3. **LLM Translation & Generation:** The Groq Llama 3.1 model natively handles translation and generates a response directly in the target language.
4. **Output:** The response text is returned to the app and read aloud using `flutter_tts` in the farmer's language.

### Offline Degradation
- **No Internet:** 
  - Disease detection works 100% offline.
  - Chat gracefully degraded to an offline TTS message: "Internet not available. Please contact local KVK."
  - Weather and Prices display the last cached data with a "Last updated X hours ago" banner.

---

## 2. FASTAPI BACKEND ARCHITECTURE

### Folder Structure
```
backend/
├── main.py                  # FastAPI app entry point
├── requirements-api.txt     # Pinned versions
├── .env.example             # Documented env vars
├── render.yaml              # Render deployment config
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── chat.py      # POST /chat
│   │   │   ├── disease.py   # POST /disease/detect
│   │   │   ├── prices.py    # GET /prices
│   │   │   ├── weather.py   # GET /weather
│   │   │   └── schemes.py   # GET /schemes
│   │   └── middleware.py    # Rate limiting, CORS
│   ├── core/
│   │   ├── config.py        # pydantic-settings
│   │   └── security.py      # JWT validation
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response schemas
│   └── services/
│       ├── llm_service.py   # Groq integration
│       ├── price_service.py # data.gov.in integration
│       └── weather_service.py # Open-Meteo integration
```

### Endpoints
- **POST `/chat`**
  - *Req:* `{ "query": "str", "language": "str", "farmer_id": "str" }`
  - *Res:* `{ "response": "str", "source": "str", "confidence": float }`
- **POST `/disease/detect`** (Online fallback/analytics)
  - *Req:* `multipart/form-data` image upload
  - *Res:* `{ "disease_name": "str", "confidence": float, "disease_name_hi": "str", "organic_treatment": "str", "chemical_treatment": "str", "dosage": "str", "source_url": "str", "source_name": "str" }`
- **GET `/prices`**
  - *Req Query:* `commodity`, `district`, `state`
  - *Res:* `{ "commodity": "str", "district": "str", "min_price": float, "max_price": float, "modal_price": float, "date": "str", "unit": "str" }`
- **GET `/weather`**
  - *Req Query:* `district`, `state`
  - *Res:* `[{ "date": "str", "max_temp": float, "min_temp": float, "rainfall_mm": float, "wind_kmh": float, "farming_advisory": "str" }]`
- **GET `/schemes`**
  - *Req Query:* `crop`, `state`
  - *Res:* `[{ "scheme_name": "str", "description": "str", "eligibility": "str", "link": "str" }]`

### Groq API & LLM Service
Instead of a dedicated translation service (which proved unviable for all regional languages on the free tier), the `llm_service.py` directly calls Groq (Llama 3.1 70B/8B).
- **System Prompt:** "You are an agricultural advisor for Indian smallholder farmers. The farmer is asking in {language}. Answer in {language}. Keep answers under 3 sentences. Be specific — name exact quantities, timings, and product names. If you are unsure, say so clearly and recommend the farmer contact their local KVK. Do not make up treatments or dosages. Append source: KVK / state agriculture dept."

### Mandi Prices Caching
- **Redis Alternative:** To maintain zero monthly costs, we use an in-memory dictionary cache in FastAPI backed by the Supabase `prices_cache` table.
- A background worker or lazy-evaluation checks if data for a `commodity` + `district` is > 6 hours old. If so, it fetches from the `data.gov.in` AGMARKNET API, updates Supabase, and updates the in-memory cache.

### Authentication
- **Supabase Auth:** Phone number OTP login. 
- The Flutter app requests an OTP, verifies it, and receives a JWT.
- The FastAPI backend validates the JWT using the Supabase `jwt` secret in `core/security.py` before serving protected endpoints.

---

## 3. SUPABASE DATABASE SCHEMA

```sql
-- Farmers profile data
CREATE TABLE farmers (
    id UUID PRIMARY KEY REFERENCES auth.users(id),
    phone VARCHAR(20) UNIQUE NOT NULL,
    district VARCHAR(100),
    crops_grown TEXT[],
    language_pref VARCHAR(10) DEFAULT 'hi'
);

-- Audit log of LLM chat queries
CREATE TABLE queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    farmer_id UUID REFERENCES farmers(id),
    query_text TEXT NOT NULL,
    language VARCHAR(10),
    response TEXT NOT NULL,
    category VARCHAR(50),
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Farmer feedback on LLM answers
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID REFERENCES queries(id),
    was_helpful BOOLEAN NOT NULL,
    follow_up_action TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Mandi prices fallback cache
CREATE TABLE prices_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    commodity VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    min_price NUMERIC,
    max_price NUMERIC,
    modal_price NUMERIC,
    date DATE NOT NULL,
    UNIQUE(commodity, district, date)
);

-- Static disease reference dictionary
CREATE TABLE diseases (
    disease_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    crop VARCHAR(100) NOT NULL,
    disease_name_en VARCHAR(100) NOT NULL,
    disease_name_hi VARCHAR(100) NOT NULL,
    symptoms_farmer_language TEXT NOT NULL,
    organic_treatment TEXT,
    chemical_treatment TEXT,
    dosage TEXT,
    source_url TEXT,
    source_name VARCHAR(100)
);

-- Government schemes
CREATE TABLE schemes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scheme_name VARCHAR(200) NOT NULL,
    description TEXT,
    eligibility TEXT,
    link TEXT,
    crop VARCHAR(100),
    state VARCHAR(100)
);

-- KVK Directory
CREATE TABLE kvk_directory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    state VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    kvk_name VARCHAR(200) NOT NULL,
    contact_number VARCHAR(20),
    email VARCHAR(100)
);
```

---

## 4. TFLITE DISEASE MODEL PIPELINE

### Pipeline Details
- **Dataset:** Kaggle PlantVillage (filtered to Indian crops).
- **Training Environment:** Google Colab (Free Tier, T4 GPU).
- **Model Architecture:** Transfer learning using MobileNetV2.
- **Quantization:** INT8 post-training quantization to reduce weights from 32-bit floats to 8-bit integers, shrinking model size from ~16MB to <4MB.

### Colab Notebook Steps (Pseudocode)
1. **Setup:** Install `tensorflow`, `kaggle`, setup API keys.
2. **Download:** Fetch PlantVillage via Kaggle API.
3. **Filter:** Discard non-Indian crops (apples, cherries, etc.). Keep Tomato, Potato, Rice, Wheat, Cotton, etc.
4. **Augment:** Apply random rotation, flips, brightness adjustments to simulate field conditions.
5. **Base Model:** Load MobileNetV2 (weights='imagenet', include_top=False). Freeze base layers.
6. **Train Head:** Add global average pooling and dense classification layer. Train for 5-10 epochs.
7. **Fine-tune:** Unfreeze the top 20-30 layers of the base model, train with a very low learning rate for 5 epochs.
8. **Quantize & Export:** Use `tf.lite.TFLiteConverter` with `optimizations = [tf.lite.Optimize.DEFAULT]`. Export as `model_quant.tflite`.
9. **Validate:** Check model size (<15MB) and run sample inferences.

### Model Updates in App
- **v1:** The model and label file are bundled directly into the APK (`assets/models/`).
- **v2 (Future):** Implement OTA updates. The app checks Supabase on startup. If a newer model hash is found, it downloads the new `.tflite` file to the device's application documents directory and loads it dynamically.

---

## 5. OFFLINE SYNC STRATEGY

### Locally Cached (Hive)
- The last 50 queries and their responses.
- The latest fetched mandi prices (retained indefinitely until overwritten).
- The 7-day weather forecast.
- The disease classification model (TFLite).

### Requires Internet
- Making a new Groq-powered chat request.
- Fetching fresh prices or weather data.
- Submitting telemetry/feedback.

### Conflict Resolution
- Data flows mostly one-way (Cloud → Device). We use a "Last write wins" approach for caching.
- For feedback submissions generated offline, they are queued in a specialized Hive box. Upon detecting an active connection, a background sync service iterates over the queue, POSTs them to the backend, and deletes the local entries upon 200 OK.

---

## 6. DEPLOYMENT PIPELINE

### GitHub Monorepo Structure
```
FarmGenius/
├── app/                  # Flutter application
├── backend/              # FastAPI backend
└── model/                # Jupyter notebooks & training scripts
```

### GitHub Actions Workflows
- **Android APK Build (On Pull Request):**
  - Triggers on PRs modifying `app/`.
  - Sets up Flutter, gets dependencies.
  - Runs `flutter build apk --release`.
  - Uploads the resulting `.apk` as a GitHub Action Artifact.
- **Backend Deploy (On Main Merge):**
  - Triggers on merge to `main` modifying `backend/`.
  - Uses Render's Deploy Hook URL. A `curl` command hits the hook, instructing Render to pull the latest code, build the Dockerfile, and redeploy the FastAPI service.
- **Keep-Alive Cron:**
  - Pings the backend `/health` endpoint every 10 minutes to prevent Render free-tier cold starts.

### Distribution
- Because the Google Play Store requires 14 days of closed testing for new accounts, the initial pre-launch tests will be conducted by downloading the APK artifact generated by GitHub Actions and distributing it directly to farmers via WhatsApp.
