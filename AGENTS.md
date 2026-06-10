# FarmGenius — Multi-Agent Build System
# Antigravity AGENTS.md — Production Android App for Indian Farmers
# Each agent has a single responsibility. No agent starts until its dependencies complete.
# Run via Antigravity Manager Surface: Cmd+Shift+M (macOS) / Ctrl+Shift+M (Windows/Linux)

---

agents:

  # ─────────────────────────────────────────────
  # AGENT 1 — RESEARCH AGENT
  # Runs first. All other agents depend on its output.
  # ─────────────────────────────────────────────
  - name: "research-agent"
    role: "Feasibility & Technology Researcher"
    model: "gemini-2.5-pro"
    description: |
      Validates the complete free technology stack before any code is written.
      Searches the internet for current free tier limits, breaking changes,
      and alternatives. Produces a verified STACK.md that all other agents read.
    instructions: |
      You are a senior technology researcher. Your job is to verify every tool
      in the FarmGenius free stack is currently available and within free limits.

      Search the internet and verify the following, producing a STACK.md report:

      1. FLUTTER
         - Confirm flutter.dev latest stable version
         - Confirm tflite_flutter plugin works with current Flutter version
         - Confirm speech_to_text and flutter_tts plugins are maintained
         - Search: "tflite_flutter Flutter <current_version> compatibility 2025"

      2. DISEASE MODEL
         - Confirm PlantVillage dataset is still freely available on Kaggle
         - Confirm Google Colab free GPU tier still allows model training
         - Confirm MobileNetV2 TFLite quantization produces <20MB model
         - Search: "PlantVillage dataset Kaggle 2025 free download"
         - Search: "Google Colab free GPU limits 2025"

      3. BACKEND HOSTING (Render)
         - Confirm Render free tier still offers 750 hrs/month
         - Confirm FastAPI deploys successfully on Render free tier
         - Check if cold start issue persists and note current workaround
         - Search: "Render free tier 2025 FastAPI cold start"

      4. DATABASE (Supabase)
         - Confirm free tier limits: 500MB storage, 50k MAU, 1GB files
         - Confirm free projects still pause after 1 week inactivity
         - Confirm Row Level Security works on free tier
         - Search: "Supabase free tier limits 2025"

      5. LLM API (Groq)
         - Confirm Groq free tier daily request limits
         - Confirm Llama 3.1 70B or equivalent is available free
         - Note current rate limits (requests/min, tokens/day)
         - Search: "Groq free tier limits 2025 requests per day"

      6. MANDI PRICE DATA
         - Confirm data.gov.in AGMARKNET API is still free and active
         - Test the endpoint: https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070
         - Note if API key is required and how to get one free
         - Search: "data.gov.in agmarknet API 2025 free access"

      7. WEATHER API
         - Confirm Open-Meteo is still free, no API key required
         - Confirm district-level India coverage exists
         - Search: "open-meteo free API 2025 India coverage"

      8. GOOGLE PLAY
         - Confirm one-time $25 developer account fee unchanged
         - Confirm APK sideloading works for pre-launch testing

      9. TRANSLATION
         - Confirm LibreTranslate supports Hindi, Kannada, Telugu, Tamil, Marathi
         - Confirm it can self-host on Render free tier
         - Search: "LibreTranslate supported languages 2025"

      10. MONITORING
          - Confirm Sentry free tier: 5000 errors/month
          - Confirm UptimeRobot free tier: 50 monitors, 5 min intervals

      OUTPUT FORMAT — write STACK.md with this structure:
      - Tool name
      - Status: CONFIRMED FREE / CHANGED / BROKEN / NEEDS_KEY
      - Current free limit
      - Any breaking changes since 2024
      - Alternative if status is not CONFIRMED FREE

      Flag any tool marked CHANGED or BROKEN as a BLOCKER before agents proceed.
    tools:
      - web_search
      - web_fetch
    output_artifact: "STACK.md"
    timeout_seconds: 600


  # ─────────────────────────────────────────────
  # AGENT 2 — FARMER EMPATHY AGENT
  # Runs in parallel with Agent 3 after Agent 1 completes.
  # ─────────────────────────────────────────────
  - name: "farmer-empathy-agent"
    role: "Farmer Research & Real Query Dataset Builder"
    model: "gemini-2.5-pro"
    description: |
      Builds the real farmer query dataset and dialect vocabulary that the
      ML agent and chatbot will be trained on. Searches publicly available
      farmer helpline data, KVK records, and agricultural forums.
    dependencies:
      - "research-agent"
    instructions: |
      You are an agricultural anthropologist who speaks Hindi, Kannada, Telugu,
      Tamil, and Marathi. Your job is to build a corpus of how real Indian
      farmers actually ask questions — not how agronomists think they ask.

      TASK 1 — QUERY CORPUS
      Search the internet and collect 200+ real farmer queries in the following
      categories. Queries must be in natural spoken language, NOT formal English:

      Categories:
        - crop disease identification ("patti peeli ho rahi hai" not "leaf yellowing")
        - pest identification
        - weather and sowing timing
        - mandi prices and when to sell
        - fertilizer dosage
        - irrigation timing
        - government scheme eligibility (PM-KISAN, crop insurance)

      Sources to search:
        - Search: "Kisan Call Centre common farmer questions India"
        - Search: "farmer WhatsApp group questions agriculture India"
        - Search: "KVK farmer queries crop disease India regional language"
        - Search: "IFFCO Kisan helpline common questions"
        - Search: "India farmer forum questions pesticide fertilizer"

      For each query, record:
        - Original language (Hindi / Kannada / Telugu / Tamil / Marathi / English)
        - Query text as a farmer would say it
        - English translation
        - Category
        - Crop mentioned (if any)
        - State/region (if identifiable)

      TASK 2 — DIALECT VOCABULARY
      Build a glossary of regional crop and disease names that differ from
      standard agricultural terminology:

      Examples of what to find:
        - "tamaku roga" (Kannada) = tobacco mosaic virus
        - "jhulsa rog" (Hindi) = leaf blight
        - "pili patti" (Hindi) = yellow leaf disease
        - Local names for: wheat, rice, cotton, soybean, tomato, onion, potato

      Search: "regional language crop disease names India agriculture glossary"
      Search: "Hindi Kannada Telugu crop disease local names farmers"

      TASK 3 — FAILURE MODES OF EXISTING APPS
      Search for farmer reviews of existing AgriTech apps to understand
      what currently fails:
        - Search: "Plantix app review farmers India problems"
        - Search: "KisanVaani app review India farmers"
        - Search: "AgriApp India farmer complaints"

      Document: top 10 complaints farmers have about existing AgriTech apps.

      TASK 4 — TRUST SIGNALS
      Research what makes farmers trust an agricultural recommendation:
        - Search: "KVK Krishi Vigyan Kendra farmer trust India"
        - Search: "why Indian farmers trust local agronomist not app"

      OUTPUT — Write FARMER_CORPUS.md with:
        1. Query corpus (200+ entries in table format)
        2. Dialect glossary (100+ terms)
        3. Existing app failure modes (top 10)
        4. Trust signal findings
    tools:
      - web_search
      - web_fetch
    output_artifact: "FARMER_CORPUS.md"
    timeout_seconds: 900


  # ─────────────────────────────────────────────
  # AGENT 3 — ARCHITECTURE AGENT
  # Runs in parallel with Agent 2 after Agent 1 completes.
  # ─────────────────────────────────────────────
  - name: "architecture-agent"
    role: "System Architect"
    model: "gemini-2.5-pro"
    description: |
      Designs the complete technical architecture of the FarmGenius Android app
      using only the free tools confirmed by the research agent.
      Produces implementation-ready specs for all downstream agents.
    dependencies:
      - "research-agent"
    instructions: |
      You are a senior mobile + backend architect. Read STACK.md first.
      Design the complete FarmGenius system using only CONFIRMED FREE tools.

      ARCHITECTURE CONSTRAINTS:
        - Android only (Flutter)
        - Works offline for disease detection (TFLite on-device)
        - Works on 2G connectivity (all API calls must have 30s timeout + retry)
        - Voice-first UX (no mandatory typing)
        - Zero monthly cost until 500+ active farmers
        - Single developer maintainable

      DESIGN THESE COMPONENTS:

      1. FLUTTER APP ARCHITECTURE
         - Folder structure (feature-first, not layer-first)
         - State management choice (Riverpod recommended — explain why vs Bloc)
         - Navigation pattern (GoRouter)
         - Offline-first data flow using Hive for local storage
         - How TFLite model is bundled and invoked
         - How speech_to_text → Groq API → flutter_tts pipeline works
         - How the app behaves with no internet (what works, what degrades)

      2. FASTAPI BACKEND ARCHITECTURE
         - Folder structure
         - Endpoints list with request/response schemas:
             POST /chat — send farmer query, get response
             POST /disease/detect — upload image, get diagnosis (if online)
             GET  /prices — get mandi prices for a commodity + district
             GET  /weather — get 7-day forecast for a district
             GET  /schemes — list government schemes relevant to a crop
         - How Groq API is called with agricultural system prompt
         - How mandi prices are cached from data.gov.in (daily cron, Redis-free alternative)
         - Authentication: phone number OTP via Supabase Auth (no passwords)

      3. SUPABASE DATABASE SCHEMA
         - farmers table (id, phone, district, crops_grown, language_pref)
         - queries table (id, farmer_id, query_text, language, response, category, timestamp)
         - feedback table (id, query_id, was_helpful, follow_up_action, timestamp)
         - prices_cache table (commodity, district, min_price, max_price, modal_price, date)
         - diseases table (name, crop, symptoms_keywords, treatment, image_url, source_kvk)

      4. TFLITE DISEASE MODEL PIPELINE
         - Dataset: PlantVillage on Kaggle
         - Training: Google Colab (free GPU, MobileNetV2 transfer learning)
         - Quantization: INT8 post-training quantization → target <15MB
         - Colab notebook steps (pseudocode outline)
         - How model is updated in app (bundled in APK v1, OTA update path for v2)

      5. OFFLINE SYNC STRATEGY
         - What is cached locally on device (Hive): last 50 queries + responses,
           current mandi prices, 7-day weather, disease model
         - What requires internet: Groq fallback, fresh mandi prices, feedback submission
         - Conflict resolution when online sync happens

      6. DEPLOYMENT PIPELINE
         - GitHub repo structure (monorepo: /app, /backend, /model)
         - GitHub Actions: Flutter build → APK artifact on every PR
         - GitHub Actions: FastAPI → auto-deploy to Render on main merge
         - How to distribute APK via WhatsApp for testing (no Play Store yet)

      OUTPUT — Write ARCHITECTURE.md with full specs, folder trees, schema SQL,
      endpoint definitions, and deployment instructions.
    tools:
      - web_search
      - web_fetch
    output_artifact: "ARCHITECTURE.md"
    timeout_seconds: 900


  # ─────────────────────────────────────────────
  # AGENT 4 — ML AGENT
  # Depends on both farmer corpus and architecture.
  # ─────────────────────────────────────────────
  - name: "ml-agent"
    role: "Machine Learning & NLP Engineer"
    model: "gemini-2.5-pro"
    description: |
      Replaces the existing FarmGenius ML stack with a farmer-validated,
      production-ready pipeline. Fixes the googletrans dependency, retrains
      the chatbot on real farmer queries, and builds the TFLite disease model.
    dependencies:
      - "research-agent"
      - "farmer-empathy-agent"
      - "architecture-agent"
    instructions: |
      You are an ML engineer specialising in on-device inference and NLP
      for low-resource languages. Read STACK.md, FARMER_CORPUS.md, and
      ARCHITECTURE.md before producing any output.

      TASK 1 — FIX THE EXISTING CHATBOT (src/ folder in FarmGenius repo)

      The current codebase uses googletrans==3.1.0a0 which is broken.
      Replace it with LibreTranslate self-hosted on Render.

      Write the replacement translation service:
      File: src/services/translation_service.py
      - Class: TranslationService
      - Method: translate(text, source_lang, target_lang) -> str
      - Uses LibreTranslate REST API (configurable endpoint via env var)
      - Falls back to returning original text if API unreachable
      - Supported lang codes: hi, kn, te, ta, mr, en

      TASK 2 — RETRAIN CHATBOT ON REAL FARMER QUERIES

      The current Naïve Bayes + TF-IDF model was trained on "curated agricultural
      FAQs." Using FARMER_CORPUS.md, produce:

      File: src/models/train_chatbot_farmer.py
      - Load queries from FARMER_CORPUS.md categories
      - Preprocess: lowercase, remove punctuation, do NOT remove stopwords
        (farmers' stop words carry meaning: "kyun", "kab", "kaise")
      - Augment: for each Hindi query, add Hinglish variant
        (e.g., "meri wheat ki leaves yellow ho rahi hain")
      - Train MultinomialNB with TF-IDF (ngram_range=(1,3) to catch phrases)
      - Evaluate: print classification report per category
      - Save model to models/chatbot_farmer_v1.pkl
      - Confidence threshold: 0.72 (higher than current 0.65 — farmer stakes are high)

      File: src/services/chatbot_service.py
      - Replaces existing chatbot logic
      - Method: get_response(query, language, farmer_context) -> dict
      - If confidence >= 0.72: use local model response
      - If confidence < 0.72: call Groq API with agricultural system prompt
      - System prompt for Groq (write this carefully):
          "You are an agricultural advisor for Indian smallholder farmers.
           The farmer is asking in {language}. Answer in {language}.
           Keep answers under 3 sentences. Be specific — name exact quantities,
           timings, and product names. If you are unsure, say so clearly and
           recommend the farmer contact their local KVK.
           Do not make up treatments or dosages."
      - Always append source: KVK / state agriculture dept / your own knowledge
      - Log every query + response to Supabase queries table

      TASK 3 — TFLITE DISEASE DETECTION MODEL

      Write a complete Google Colab notebook (as Python cells in markdown):
      File: model/train_disease_model.ipynb (pseudocode + real code)

      Steps:
        Cell 1: Install deps (tensorflow, kaggle, Pillow)
        Cell 2: Download PlantVillage from Kaggle API
        Cell 3: Filter to Indian crops only:
                 tomato, potato, rice, wheat, cotton, chickpea, maize
                 (removes apple, blueberry, cherry etc. not grown by target farmers)
        Cell 4: Data augmentation (rotation, flip, brightness — simulate field photos)
        Cell 5: MobileNetV2 transfer learning (freeze base, train top layers)
        Cell 6: Fine-tune (unfreeze last 30 layers, lower LR)
        Cell 7: Evaluate — print per-class accuracy, flag any class < 80%
        Cell 8: Export to TFLite with INT8 quantization
        Cell 9: Verify model size < 15MB, test inference speed on CPU

      Also write: model/labels_indian_crops.txt
      - One class per line, format: "Tomato___Early_blight"
      - Include "Healthy" class for each crop
      - Include "Unknown — please retake photo" catch-all class

      TASK 4 — GROQ INTEGRATION (replaces DeepSeek)

      File: src/services/llm_service.py
      - Class: LLMService
      - Uses Groq API (model: llama-3.1-70b-versatile or current best free model)
      - API key from env var GROQ_API_KEY
      - Rate limit handler: if 429 received, wait 60s and retry once
      - If Groq fails: return graceful message in farmer's language
        Hindi: "माफ करें, अभी सेवा उपलब्ध नहीं है। कृपया अपने KVK से संपर्क करें।"
        Kannada: "ಕ್ಷಮಿಸಿ, ಸೇವೆ ಲಭ್ಯವಿಲ್ಲ. ನಿಮ್ಮ KVK ಗೆ ಸಂಪರ್ಕಿಸಿ."
        English: "Sorry, service is unavailable. Please contact your local KVK."

      OUTPUT — Write ML_IMPLEMENTATION.md with:
        - All file paths and complete code for each task
        - Requirements additions to requirements-api.txt and requirements-ml.txt
        - How to run training locally and on Colab
        - Model accuracy targets (minimum acceptable per class)
        - How model gets updated in the Flutter app (asset bundling steps)
    tools:
      - web_search
      - web_fetch
    output_artifact: "ML_IMPLEMENTATION.md"
    timeout_seconds: 1200


  # ─────────────────────────────────────────────
  # AGENT 5 — BACKEND AGENT
  # Depends on architecture spec.
  # ─────────────────────────────────────────────
  - name: "backend-agent"
    role: "Backend & API Engineer"
    model: "gemini-2.5-pro"
    description: |
      Builds the complete production FastAPI backend using the architecture spec.
      Handles auth, all endpoints, mandi price caching, Supabase integration,
      and Render deployment configuration.
    dependencies:
      - "architecture-agent"
      - "ml-agent"
    instructions: |
      You are a senior Python backend engineer. Read ARCHITECTURE.md and
      ML_IMPLEMENTATION.md before writing any code.

      TASK 1 — PROJECT STRUCTURE
      Create the complete FastAPI backend folder structure:

      backend/
      ├── main.py                  # FastAPI app entry point
      ├── requirements-api.txt     # Pinned versions only
      ├── .env.example             # All env vars documented
      ├── Dockerfile               # Multi-stage, slim Python image
      ├── render.yaml              # Render deployment config
      ├── .github/
      │   └── workflows/
      │       └── deploy.yml       # GitHub Actions: test + deploy on main merge
      ├── app/
      │   ├── api/
      │   │   ├── routes/
      │   │   │   ├── chat.py      # POST /chat
      │   │   │   ├── disease.py   # POST /disease/detect
      │   │   │   ├── prices.py    # GET /prices
      │   │   │   ├── weather.py   # GET /weather
      │   │   │   └── schemes.py   # GET /schemes
      │   │   └── middleware.py    # Rate limiting, CORS, logging
      │   ├── core/
      │   │   ├── config.py        # Settings from env vars (pydantic-settings)
      │   │   └── security.py      # JWT validation via Supabase
      │   ├── models/
      │   │   └── schemas.py       # All Pydantic request/response models
      │   └── services/
      │       ├── chatbot_service.py
      │       ├── llm_service.py
      │       ├── translation_service.py
      │       ├── price_service.py
      │       └── weather_service.py
      └── tests/
          ├── test_chat.py
          ├── test_prices.py
          └── test_disease.py

      TASK 2 — WRITE ALL ENDPOINT CODE

      Write complete, working Python code for:

      a) POST /chat
         Request: { query: str, language: str, farmer_id: str }
         - Translate query to English if not English (TranslationService)
         - Run through ChatbotService (local model → Groq fallback)
         - Translate response back to farmer's language
         - Log to Supabase queries table
         - Return: { response: str, source: str, confidence: float }

      b) GET /prices?commodity=wheat&district=dharwad&state=karnataka
         - Check Supabase prices_cache for today's data
         - If cache miss or data > 6 hours old: fetch from data.gov.in API
         - Store in cache, return to farmer
         - Return: { commodity, district, min_price, max_price, modal_price, date, unit }

      c) GET /weather?district=dharwad&state=karnataka
         - Fetch from Open-Meteo API (no key needed)
         - Use district centroid coordinates (maintain a districts_coords.json)
         - Return 7-day forecast relevant to farming:
           { date, max_temp, min_temp, rainfall_mm, wind_kmh, farming_advisory }
         - Farming advisory: simple rule-based text
           e.g. if rainfall_mm > 20: "Heavy rain expected — avoid spraying pesticides"

      d) POST /disease/detect (online fallback only — primary detection is on-device)
         - Accept: multipart/form-data image upload
         - Validate: file type (jpg/png only), size (<5MB)
         - Run inference using the TFLite model loaded at startup
         - Return: { disease_name, confidence, treatment, source_kvk }

      e) GET /schemes?crop=wheat&state=maharashtra
         - Query Supabase schemes table (pre-seeded from govt sources)
         - Return list of relevant government schemes with eligibility and links

      TASK 3 — SECURITY & MIDDLEWARE

      Write app/api/middleware.py:
        - CORS: allow only your Flutter app's domain + localhost for dev
        - Rate limiting: 30 requests/minute per farmer_id using in-memory dict
          (no Redis needed at this scale — upgrade later)
        - Request logging: log method, path, farmer_id, response_time to stdout
          (Render captures stdout as logs)
        - Auth: validate Supabase JWT on all routes except /health

      Write app/core/security.py:
        - validate_token(token: str) -> farmer_id: str
        - Calls Supabase auth.getUser() with the token
        - Caches result for 5 minutes to avoid hammering Supabase

      TASK 4 — RENDER DEPLOYMENT

      Write render.yaml:
        - Service type: web
        - Runtime: Python
        - Build command: pip install -r requirements-api.txt
        - Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
        - Health check path: /health
        - Instance type: free

      Write .env.example (with descriptions for each variable):
        SUPABASE_URL=
        SUPABASE_SERVICE_KEY=
        GROQ_API_KEY=
        LIBRE_TRANSLATE_URL=
        DATA_GOV_IN_API_KEY=
        ENVIRONMENT=production

      TASK 5 — KEEP-ALIVE CRON (prevents Render cold starts)
      Write a GitHub Actions cron job (.github/workflows/keepalive.yml):
        - Runs every 10 minutes
        - Hits GET /health on the Render URL
        - Keeps the free instance warm

      TASK 6 — TESTS
      Write pytest tests for at minimum:
        - /chat returns 200 with valid query
        - /prices returns cached data on second call
        - /disease/detect rejects non-image files
        - Rate limiter blocks request 31 on same farmer_id

      OUTPUT — Write BACKEND_CODE.md with all complete file contents,
      ready to copy-paste into the repository. Include pinned requirements.
    tools:
      - web_search
      - web_fetch
    output_artifact: "BACKEND_CODE.md"
    timeout_seconds: 1500


  # ─────────────────────────────────────────────
  # AGENT 6 — FLUTTER AGENT
  # Depends on architecture and backend specs.
  # ─────────────────────────────────────────────
  - name: "flutter-agent"
    role: "Flutter Android Developer"
    model: "gemini-2.5-pro"
    description: |
      Builds the complete Flutter Android app — voice-first, offline-capable,
      works on 2G. Primary interface is speak → get spoken answer.
      No mandatory typing anywhere.
    dependencies:
      - "architecture-agent"
      - "backend-agent"
    instructions: |
      You are a senior Flutter developer specialising in rural India use cases.
      Read ARCHITECTURE.md and BACKEND_CODE.md before writing any code.

      DESIGN PRINCIPLES (never violate these):
        - A farmer who cannot read must be able to use this app
        - Every action has a voice alternative
        - Every screen works offline (shows cached data, not error screens)
        - App size must be < 50MB (farmers have limited storage)
        - Target: Android 6.0+ (API 23+) — rural India device profile
        - Text size minimum 18sp — farmers often use phones at arm's length

      TASK 1 — PROJECT SETUP
      Write pubspec.yaml with pinned dependencies:
        - flutter_riverpod: ^2.x (state management)
        - go_router: ^13.x (navigation)
        - tflite_flutter: ^0.10.x (on-device inference)
        - speech_to_text: ^6.x (mic input)
        - flutter_tts: ^4.x (voice output)
        - hive_flutter: ^1.x (local offline storage)
        - dio: ^5.x (HTTP client with retry)
        - supabase_flutter: ^2.x (auth + realtime)
        - image_picker: ^1.x (camera for disease detection)
        - permission_handler: ^11.x (mic, camera permissions)

      TASK 2 — FOLDER STRUCTURE
      app/
      ├── lib/
      │   ├── main.dart
      │   ├── app.dart                   # GoRouter setup, theme
      │   ├── core/
      │   │   ├── constants.dart         # API URLs, Supabase keys
      │   │   ├── theme.dart             # Large text, high contrast, green palette
      │   │   └── offline_manager.dart   # Hive init, cache helpers
      │   ├── features/
      │   │   ├── voice_chat/            # Main feature: speak → answer
      │   │   │   ├── screen.dart
      │   │   │   ├── provider.dart
      │   │   │   └── widgets/
      │   │   ├── disease_detect/        # Camera → TFLite → result
      │   │   │   ├── screen.dart
      │   │   │   ├── provider.dart
      │   │   │   └── tflite_service.dart
      │   │   ├── mandi_prices/          # Price lookup by crop + district
      │   │   │   ├── screen.dart
      │   │   │   └── provider.dart
      │   │   ├── weather/               # 7-day forecast + farming advisory
      │   │   │   ├── screen.dart
      │   │   │   └── provider.dart
      │   │   ├── schemes/               # Govt scheme discovery
      │   │   │   ├── screen.dart
      │   │   │   └── provider.dart
      │   │   └── onboarding/            # Language select, district, crops grown
      │   │       ├── screen.dart
      │   │       └── provider.dart
      │   └── shared/
      │       ├── widgets/
      │       │   ├── mic_button.dart    # Big circular mic button — core UI
      │       │   ├── speak_card.dart    # Spoken response card with replay button
      │       │   └── offline_banner.dart
      │       └── services/
      │           ├── api_service.dart   # Dio wrapper for all backend calls
      │           └── tts_service.dart   # flutter_tts wrapper
      ├── assets/
      │   ├── models/
      │   │   └── disease_model.tflite  # Bundled on-device model
      │   │   └── labels_indian_crops.txt
      │   └── translations/             # ARB files for i18n
      │       ├── app_en.arb
      │       ├── app_hi.arb
      │       ├── app_kn.arb
      │       ├── app_te.arb
      │       └── app_ta.arb

      TASK 3 — WRITE CORE SCREENS

      a) ONBOARDING SCREEN (first launch only)
         - Step 1: "Aapki bhasha chuniye" — language selector (big flag buttons)
           Languages: Hindi, Kannada, Telugu, Tamil, Marathi, English
         - Step 2: "Aapka district?" — searchable district dropdown
         - Step 3: "Aap kya ugaate hain?" — crop multiselect with icons
         - Step 4: Phone number OTP via Supabase Auth
         - Save all to Hive + Supabase farmers table
         - No typing required except phone number

      b) VOICE CHAT SCREEN (home screen — first thing farmer sees)
         - Huge circular mic button (min 80dp, green)
         - Press and hold → recording → release → processing → spoken answer
         - Show waveform while recording
         - Show response text + play button (for farmer to replay)
         - Show category tag: "Rog" / "Mausam" / "Bhav" / "Yojana"
         - Offline: use cached response if similar query found in Hive
         - Write complete Dart code for VoiceChatScreen and VoiceChatProvider

      c) DISEASE DETECT SCREEN
         - Single large camera button: "Patti ki photo lo" (Take leaf photo)
         - On capture: run tflite_service.dart inference immediately (offline)
         - Show: disease name (in farmer's language), confidence %, treatment steps
         - Show: "Source: KVK [district name]" trust badge
         - If confidence < 60%: "Photo dubara lo — aur pass se" (Retake, closer)
         - Write complete Dart code for DiseaseDetectScreen and TFLiteService

      d) MANDI PRICES SCREEN
         - Default: show prices for farmer's crops in their district
         - Search bar to look up any commodity
         - Show: min / max / modal price, date updated, market name
         - Offline: show last cached price with "Last updated: X hours ago" warning
         - Write complete Dart code

      e) WEATHER SCREEN
         - Show 7-day forecast as horizontal scrollable cards
         - Each card: date, icon (sun/rain/cloud), max/min temp, rainfall
         - Below forecast: farming advisory in farmer's language
           e.g. "Kal barish ho sakti hai — aaj spray mat karein"
         - Write complete Dart code

      TASK 4 — TFLITE SERVICE
      Write lib/features/disease_detect/tflite_service.dart:
        - Load model from assets at app startup (not on demand — avoid latency)
        - Method: diagnose(XFile imageFile) -> DiagnosisResult
        - Preprocess: resize to 224x224, normalize to [0,1]
        - Run inference: return top-3 predictions with confidence scores
        - Map label to: disease name (localised), treatment text, KVK source
        - Under 500ms on a mid-range Android device

      TASK 5 — OFFLINE STRATEGY
      Write lib/core/offline_manager.dart:
        - Hive boxes: 'queries', 'prices', 'weather', 'farmer_profile'
        - Cache last 50 query-response pairs
        - Cache mandi prices for farmer's crops (refreshed when online)
        - Cache weather for farmer's district (refreshed when online)
        - Method: findSimilarQuery(query) — simple keyword overlap, returns
          cached response if similarity > 0.6

      TASK 6 — API SERVICE WITH 2G RESILIENCE
      Write lib/shared/services/api_service.dart:
        - Dio base options: connectTimeout 15s, receiveTimeout 30s
        - Retry interceptor: 3 retries with exponential backoff
        - Error interceptor: on any error, check Hive cache for fallback
        - Auth interceptor: attach Supabase JWT to every request

      TASK 7 — BUILD & DISTRIBUTION
      Write a GitHub Actions workflow (.github/workflows/build_apk.yml):
        - Trigger: push to main
        - Steps: flutter pub get → flutter test → flutter build apk --release
        - Upload APK as GitHub Actions artifact
        - This APK link can be shared via WhatsApp for farmer testing
          (no Play Store needed during testing phase)

      OUTPUT — Write FLUTTER_CODE.md with all complete Dart file contents,
      pubspec.yaml, and build instructions.
    tools:
      - web_search
      - web_fetch
    output_artifact: "FLUTTER_CODE.md"
    timeout_seconds: 1500


  # ─────────────────────────────────────────────
  # AGENT 7 — TRUST AGENT
  # Runs in parallel with Agents 4, 5, 6.
  # ─────────────────────────────────────────────
  - name: "trust-agent"
    role: "Farmer Trust & Adoption Specialist"
    model: "gemini-2.5-pro"
    description: |
      Builds the content, data, and partnerships that make farmers trust
      the app's recommendations. Sources all agricultural content from
      verified government and KVK sources.
    dependencies:
      - "farmer-empathy-agent"
      - "architecture-agent"
    instructions: |
      You are an agricultural extension worker turned technologist.
      You understand that a wrong recommendation can destroy a farmer's
      livelihood. Every piece of content this agent produces must be
      traceable to a government or KVK source.

      TASK 1 — DISEASE DATABASE
      Search the internet and build a verified disease database for
      the 7 Indian crops in the TFLite model:
      Crops: tomato, potato, rice, wheat, cotton, chickpea, maize

      For each disease in PlantVillage that matches these crops, find:
        - Official disease name
        - Local name in Hindi (and Kannada where available)
        - Symptoms in plain farmer language (not scientific)
        - Organic treatment option (search: "<disease> organic treatment India")
        - Chemical treatment: active ingredient, dosage, frequency
        - Source: ICAR / state agriculture dept / KVK advisory

      Search sources:
        - https://icar.org.in publications
        - Search: "ICAR crop disease management <crop>"
        - Search: "KVK advisory <crop> disease treatment India"
        - Search: "state agriculture department <crop> disease India"

      Output format for each disease:
        disease_id, crop, disease_name_en, disease_name_hi,
        symptoms_farmer_language, organic_treatment, chemical_treatment,
        dosage, source_url, source_name

      Minimum 40 disease entries covering all 7 crops.

      TASK 2 — GOVERNMENT SCHEMES DATABASE
      Search and compile currently active government schemes relevant
      to small and marginal farmers in India (< 5 acres):

      Mandatory schemes to include (search current status of each):
        - PM-KISAN (₹6000/year direct transfer) — search eligibility 2025
        - PM Fasal Bima Yojana (crop insurance) — search current premium rates
        - Kisan Credit Card — search current interest rate and limit
        - Soil Health Card scheme — search how to apply
        - PM Krishi Sinchai Yojana (irrigation subsidy)
        - eNAM — search how farmers register to sell online

      For each scheme record:
        scheme_name, benefit_description, eligibility, how_to_apply,
        documents_needed, helpline_number, website_url, applicable_states

      TASK 3 — KVK DIRECTORY
      Search and compile KVK (Krishi Vigyan Kendra) contact details
      for the following states (start with these, expand later):
        - Karnataka, Maharashtra, Madhya Pradesh, Uttar Pradesh, Andhra Pradesh

      For each KVK: district, state, phone_number, email, website
      Search: "KVK Krishi Vigyan Kendra contact directory <state>"
      Source: icar.org.in/en/kvk

      TASK 4 — TRUST UX COPY
      Write all UI strings that build farmer trust.
      For each string, provide: English + Hindi + Kannada

      Strings needed:
        - Disease result trust badge: "As per ICAR advisory"
        - Low confidence message: "I'm not sure — please consult your KVK"
        - KVK contact prompt: "Talk to an expert at [KVK name]: [phone]"
        - Data source label for mandi prices: "Data from Government of India"
        - Disclaimer for treatments: "Always confirm dosage with your local dealer"
        - Scheme eligibility caveat: "Eligibility may vary. Contact your bank/CSC"

      TASK 5 — COMMUNITY VERIFICATION PLAN
      Write a simple plan (not code) for how experienced farmers can
      verify AI answers in a future version:
        - How a "verified farmer" badge is earned
        - How they up-vote or correct an AI response
        - How verified corrections are fed back to improve the model
        - What incentive they receive (airtime credit, recognition)

      OUTPUT — Write TRUST_CONTENT.md with:
        1. Disease database (SQL INSERT statements for Supabase)
        2. Government schemes database (SQL INSERT statements)
        3. KVK directory (SQL INSERT statements)
        4. Trust UX copy (table: key | English | Hindi | Kannada)
        5. Community verification plan (prose)
    tools:
      - web_search
      - web_fetch
    output_artifact: "TRUST_CONTENT.md"
    timeout_seconds: 1200


  # ─────────────────────────────────────────────
  # AGENT 8 — FEEDBACK AGENT
  # Runs last, after all build agents complete.
  # ─────────────────────────────────────────────
  - name: "feedback-agent"
    role: "Farmer Testing & Feedback System Builder"
    model: "gemini-2.5-pro"
    description: |
      Builds the complete farmer testing protocol and in-app feedback system.
      Ensures every sprint ends with structured farmer input that directly
      drives the next sprint's priorities.
    dependencies:
      - "flutter-agent"
      - "backend-agent"
      - "trust-agent"
    instructions: |
      You are a UX researcher who has conducted field studies with farmers
      across rural India. You know that farmers won't fill forms, won't rate
      apps, but will answer 2–3 spoken questions.

      TASK 1 — IN-APP VOICE FEEDBACK SYSTEM

      After every chatbot response, show a simple feedback widget.
      Write the Flutter widget: lib/shared/widgets/feedback_widget.dart

      The widget shows two large buttons:
        - Green thumbs up (with sound: "Helpful!")
        - Red thumbs up down (with sound: "Not helpful")

      On thumbs down, ask ONE voice question (spoken by TTS):
        Hindi: "Aapko aur kya chahiye tha?" (What more did you need?)
        The farmer speaks — speech_to_text captures it.
        This is sent to: POST /feedback

      Backend endpoint POST /feedback:
        Request: { query_id, was_helpful: bool, follow_up_text: str, language: str }
        Store in Supabase feedback table
        If was_helpful == false: trigger Slack webhook notification to developer

      TASK 2 — FIELD TESTING PROTOCOL

      Write FIELD_TEST_PROTOCOL.md — a guide for conducting farmer testing sessions.
      Format it as a step-by-step script that a non-technical person can follow.

      Include:

      a) PRE-SESSION (30 minutes before)
         - Install APK via WhatsApp on farmer's own phone
         - Confirm mic and camera permissions are granted
         - Set language to farmer's language in onboarding
         - Do NOT explain what the app does — let them explore

      b) SESSION SCRIPT (45 minutes with one farmer)
         - Give farmer 3 tasks (do not help them, just observe):
             Task 1: "Ask the app something about your crop right now"
             Task 2: "Check today's price for [their main crop]"
             Task 3: "Take a photo of a leaf and find out if it's sick"
         - Record with phone (with permission): what they tap, where they hesitate
         - Note: did they use voice or try to type?
         - Note: did they understand the response?
         - Note: did they trust the response or want to verify it?

      c) POST-SESSION (10 minutes — 3 spoken questions only)
         Question 1 (spoken in their language):
           "App ne jo bataya — aapne kuch alag kiya apne khet mein?"
           (Did you do anything different in your field because of what the app said?)
           → This is the only question that matters for impact.

         Question 2:
           "Koi cheez samajh nahi aayi?" (Anything you didn't understand?)

         Question 3:
           "Kya aap kisi doosre kisan ko dikhayenge?" (Would you show another farmer?)
           → Net promoter score proxy. If yes: adoption is real.

      d) WHAT TO RECORD
         Create a simple spreadsheet template (CSV format) with columns:
         farmer_id, district, crop, task1_completed, task2_completed,
         task3_completed, used_voice (yes/no), understood_response (yes/no),
         trusted_response (yes/no), would_show_neighbour (yes/no),
         key_observation, priority_fix

      e) AFTER 5 SESSIONS — TRIAGE MEETING
         Score each issue by: (frequency × farmer_impact)
         Top 3 issues go to next sprint. Everything else waits.

      TASK 3 — SPRINT TRACKER
      Write SPRINT_TRACKER.md — a simple table tracking each sprint:

      | Sprint | Built | Tested with | Top issue found | Fixed in | Farmers reached |
      |--------|-------|-------------|-----------------|----------|-----------------|
      | 1      | Disease detect + voice (Kannada) | 5 farmers Dharwad | Photo too blurry | Sprint 2 | 5 |

      Pre-fill Sprint 1 based on what this agent system is building.
      Leave Sprint 2–5 blank for the team to fill in.

      TASK 4 — WHATSAPP DISTRIBUTION GUIDE
      Write a guide for distributing APK to farmers via WhatsApp:
        - How to generate download link from GitHub Actions artifact
        - WhatsApp message template (in Hindi):
          "नमस्ते! FarmGenius app try करें — बिल्कुल free है।
           Link: [APK link]
           Install करने के लिए: Settings → Security → Unknown sources ON करें
           कोई problem हो तो बताएं।"
        - How to handle "Unknown sources" fear (farmers are wary of this)
        - Sideloading instructions with screenshots (describe what to write)

      OUTPUT — Write FEEDBACK_SYSTEM.md with:
        1. Complete Flutter feedback widget code
        2. Backend /feedback endpoint code
        3. FIELD_TEST_PROTOCOL.md content
        4. SPRINT_TRACKER.md template
        5. WhatsApp APK distribution guide
    tools:
      - web_search
      - web_fetch
    output_artifact: "FEEDBACK_SYSTEM.md"
    timeout_seconds: 900


  # ─────────────────────────────────────────────
  # AGENT 9 — ORCHESTRATOR AGENT
  # Runs last. Reviews all outputs, finds conflicts, produces final plan.
  # ─────────────────────────────────────────────
  - name: "orchestrator-agent"
    role: "Project Orchestrator & Integration Reviewer"
    model: "gemini-2.5-pro"
    description: |
      Reviews all agent outputs for conflicts, gaps, and integration issues.
      Produces the final BUILD_PLAN.md that a developer can follow start-to-finish
      to ship the FarmGenius Android app.
    dependencies:
      - "research-agent"
      - "farmer-empathy-agent"
      - "architecture-agent"
      - "ml-agent"
      - "backend-agent"
      - "flutter-agent"
      - "trust-agent"
      - "feedback-agent"
    instructions: |
      You are the technical lead reviewing all agent outputs before a developer
      picks up this project. Read ALL output artifacts:
        STACK.md, FARMER_CORPUS.md, ARCHITECTURE.md, ML_IMPLEMENTATION.md,
        BACKEND_CODE.md, FLUTTER_CODE.md, TRUST_CONTENT.md, FEEDBACK_SYSTEM.md

      YOUR JOB:

      1. CONFLICT DETECTION
         Find any inconsistencies across agent outputs. Examples to look for:
         - ML agent uses a Python library not in backend requirements
         - Flutter agent calls an endpoint that backend agent didn't implement
         - Architecture schema columns don't match what backend code queries
         - Trust content uses a language code not supported by translation service
         Flag each conflict with: CONFLICT #N — [agent A] vs [agent B] — [description]

      2. GAP DETECTION
         Find anything missing that would block a developer from shipping:
         - Env vars referenced in code but not in .env.example
         - Supabase tables referenced in code but not in schema
         - Flutter screens routed to but not implemented
         - API endpoints called in Flutter but not defined in backend
         Flag each gap: GAP #N — [location] — [what's missing]

      3. PRIORITY ORDER
         Produce a numbered developer task list in the order a single developer
         should tackle them to get to first farmer test as fast as possible.
         Group into:
           Week 1 (get something on a farmer's phone)
           Week 2–3 (make it useful)
           Week 4+ (make it trustworthy and scalable)

      4. SPRINT 1 DEFINITION
         Write the exact scope of Sprint 1 — what ONE thing the app does
         end-to-end that a farmer can test. Must be achievable in 2 weeks.
         Recommended: Voice → disease detection (Kannada) → spoken diagnosis
         Confirm this is fully buildable with the outputs the agents produced.

      5. SUCCESS METRICS
         Define how the team knows Sprint 1 worked:
         - Technical: app installs on Android 6+ without crash
         - UX: farmer completes disease detect task without help
         - Trust: farmer says they would retake photo based on app result
         - Impact: at least 1 farmer says they would act differently

      6. RISK REGISTER
         List the top 5 risks and mitigations:
         e.g. "Groq API changes free tier → mitigation: Gemini Flash free tier backup"

      OUTPUT — Write BUILD_PLAN.md:
        - Conflicts list (numbered)
        - Gaps list (numbered)
        - Developer task list (week-by-week)
        - Sprint 1 exact scope
        - Success metrics
        - Risk register
        - One-paragraph summary a non-technical founder can read

      This is the document the developer opens on Day 1.
    tools:
      - web_search
    output_artifact: "BUILD_PLAN.md"
    timeout_seconds: 600


# ─────────────────────────────────────────────────────────────────────────────
# COMMUNICATION PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

communication:

  - from: "research-agent"
    to: ["farmer-empathy-agent", "architecture-agent"]
    protocol: "async"
    format: "markdown"
    artifact: "STACK.md"
    description: "Verified free stack — all downstream agents must read before starting"

  - from: "farmer-empathy-agent"
    to: ["ml-agent", "trust-agent", "feedback-agent"]
    protocol: "async"
    format: "markdown"
    artifact: "FARMER_CORPUS.md"
    description: "Real farmer query corpus and dialect vocabulary"

  - from: "architecture-agent"
    to: ["ml-agent", "backend-agent", "flutter-agent", "trust-agent"]
    protocol: "async"
    format: "markdown"
    artifact: "ARCHITECTURE.md"
    description: "Full system architecture — all agents must align to this spec"

  - from: "ml-agent"
    to: ["backend-agent", "flutter-agent"]
    protocol: "async"
    format: "markdown"
    artifact: "ML_IMPLEMENTATION.md"
    description: "ML code, model training steps, and service interfaces"

  - from: "backend-agent"
    to: ["flutter-agent", "feedback-agent"]
    protocol: "async"
    format: "markdown"
    artifact: "BACKEND_CODE.md"
    description: "Complete backend code and endpoint contracts"

  - from: ["flutter-agent", "backend-agent", "trust-agent"]
    to: "feedback-agent"
    protocol: "async"
    format: "markdown"
    description: "Feedback agent needs all three to build the testing system"

  - from: "*"
    to: "orchestrator-agent"
    protocol: "async"
    format: "markdown"
    description: "Orchestrator reads all artifacts only after all agents complete"


# ─────────────────────────────────────────────────────────────────────────────
# MANAGER SURFACE — ORCHESTRATION RULES
# ─────────────────────────────────────────────────────────────────────────────

manager_surface:
  enabled: true
  orchestrator: "orchestrator-agent"

  execution_plan:
    - phase: "research"
      agents: ["research-agent"]
      mode: "sequential"
      description: "Must complete before anything else starts"

    - phase: "discovery"
      agents: ["farmer-empathy-agent", "architecture-agent"]
      mode: "parallel"
      description: "Run simultaneously — independent of each other"
      depends_on: ["research"]

    - phase: "build"
      agents: ["ml-agent", "backend-agent", "flutter-agent", "trust-agent"]
      mode: "parallel"
      description: "Run simultaneously — all depend on discovery outputs"
      depends_on: ["discovery"]

    - phase: "testing"
      agents: ["feedback-agent"]
      mode: "sequential"
      depends_on: ["build"]

    - phase: "integration"
      agents: ["orchestrator-agent"]
      mode: "sequential"
      description: "Reviews all outputs, produces final BUILD_PLAN.md"
      depends_on: ["testing"]

  quality_gates:
    - after_phase: "research"
      check: "STACK.md must contain no BROKEN status tools"
      on_fail: "halt — research agent must find alternatives before proceeding"

    - after_phase: "discovery"
      check: "ARCHITECTURE.md must define all endpoints called in FLUTTER_CODE scope"
      on_fail: "architecture-agent reruns endpoint definitions section"

    - after_phase: "build"
      check: "All output artifacts exist: ML_IMPLEMENTATION.md, BACKEND_CODE.md, FLUTTER_CODE.md, TRUST_CONTENT.md"
      on_fail: "flag missing artifact to human reviewer"

  decision_rules:
    - condition: "tool_status_BROKEN in STACK.md"
      action: "halt_all_agents — research agent must resolve before proceeding"

    - condition: "conflict_detected by orchestrator"
      action: "flag to human — do not auto-resolve architecture conflicts"

    - condition: "agent_timeout"
      action: "retry once, then output partial artifact and flag"

  failover_rules:
    - primary: "gemini-2.5-pro"
      fallback: "gemini-2.0-flash"
      trigger: "rate_limit_exceeded"

  conflict_resolution:
    - between: "ml-agent"
      and: "backend-agent"
      rule: "backend-agent owns API contracts — ml-agent must match"

    - between: "flutter-agent"
      and: "backend-agent"
      rule: "backend-agent endpoint definitions are authoritative"

    - between: "architecture-agent"
      and: "any"
      rule: "architecture-agent schema is authoritative for database"


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECKS & TIMEOUTS
# ─────────────────────────────────────────────────────────────────────────────

health_check:
  interval_seconds: 120
  on_failure:
    action: "retry_agent"
    max_retries: 2
    then: "output_partial_and_flag"

logging:
  enabled: true
  level: "INFO"
  log_agent_start: true
  log_agent_complete: true
  log_artifacts_produced: true
  log_conflicts_detected: true


# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED OUTPUT ARTIFACTS (in order of production)
# ─────────────────────────────────────────────────────────────────────────────

# Phase 1:  STACK.md              — verified free tech stack
# Phase 2:  FARMER_CORPUS.md      — real farmer queries and dialect vocabulary
#           ARCHITECTURE.md       — full system design and database schema
# Phase 3:  ML_IMPLEMENTATION.md  — chatbot retraining, TFLite model, Groq service
#           BACKEND_CODE.md       — complete FastAPI code, ready to deploy
#           FLUTTER_CODE.md       — complete Flutter app code
#           TRUST_CONTENT.md      — disease DB, schemes, KVK contacts, UI copy
# Phase 4:  FEEDBACK_SYSTEM.md    — field testing protocol, in-app feedback
# Phase 5:  BUILD_PLAN.md         — final integration review, day-1 developer guide


# ─────────────────────────────────────────────────────────────────────────────
# HOW TO RUN THIS IN ANTIGRAVITY
# ─────────────────────────────────────────────────────────────────────────────

# 1. Open the FarmGenius repository in Antigravity IDE
# 2. Place this file at the project root as: AGENTS.md
# 3. Open Manager Surface: Cmd+Shift+M (macOS) / Ctrl+Shift+M (Windows/Linux)
# 4. Click "Load from AGENTS.md"
# 5. Review the execution plan — 5 phases will be shown
# 6. Click "Run All" — agents will execute in dependency order
# 7. Monitor progress in Manager Surface dashboard
# 8. When complete, open BUILD_PLAN.md — that is your Day 1 developer guide
#
# ESTIMATED RUNTIME: 2–4 hours total (agents run in parallel where possible)
# COST: Depends on your Antigravity plan and model tier selected
# HUMAN REVIEW REQUIRED: After orchestrator-agent flags any conflicts
