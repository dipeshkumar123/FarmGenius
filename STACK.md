# FarmGenius Technology Stack Verification Report (2026)

**STATUS SUMMARY:** 
⚠️ **BLOCKER IDENTIFIED:** The translation tool (LibreTranslate) does not support all required regional languages. An alternative must be used.

---

### 1. FLUTTER
- **Status:** CONFIRMED FREE
- **Current free limit:** Open source and completely free.
- **Breaking changes since 2024:** Transition to Swift Package Manager (SPM) for iOS/macOS. Null safety is mandatory (Dart 3). `tflite_flutter` backend is transitioning to LiteRT branding, but the package remains functional.
- **Alternative:** N/A

### 2. DISEASE MODEL (PlantVillage & Google Colab)
- **Status:** CONFIRMED FREE
- **Current free limit:** Kaggle dataset is free. Colab provides 15-30 hours/week of free NVIDIA T4 GPU compute (sessions up to 12 hours).
- **Breaking changes since 2024:** Colab usage is more strictly throttled on the free tier, with dynamic cooldowns based on demand.
- **Alternative:** N/A

### 3. BACKEND HOSTING (Render)
- **Status:** CONFIRMED FREE
- **Current free limit:** 750 hours/month compute. 0.1 vCPU and 512 MB RAM.
- **Breaking changes since 2024:** Free web services spin down after 15 minutes of inactivity. Cold starts take 30-60 seconds. Postgres database expires after 90 days.
- **Alternative:** For cold starts: Use a cron job or uptime tool to ping the service every 10-14 minutes.

### 4. DATABASE (Supabase)
- **Status:** CONFIRMED FREE
- **Current free limit:** 50,000 MAU, 500MB Database storage, 1GB File storage.
- **Breaking changes since 2024:** Free projects automatically pause after 7 days of inactivity. Account limited to 2 active free projects.
- **Alternative:** N/A

### 5. LLM API (Groq)
- **Status:** CONFIRMED FREE
- **Current free limit:** 14,400 Requests Per Day, 30 RPM, 6,000 TPM (for Llama 3.1 8B Instant).
- **Breaking changes since 2024:** Limits are enforced strictly per organization.
- **Alternative:** N/A

### 6. MANDI PRICE DATA (data.gov.in AGMARKNET API)
- **Status:** NEEDS_KEY
- **Current free limit:** Free to access, subject to standard data.gov.in rate limits.
- **Breaking changes since 2024:** Requires registration on data.gov.in to generate a free API Key via the "My Account" dashboard.
- **Alternative:** N/A

### 7. WEATHER API (Open-Meteo)
- **Status:** CONFIRMED FREE
- **Current free limit:** 10,000 API calls per day for non-commercial use.
- **Breaking changes since 2024:** None. No API key required. Query relies on latitude/longitude coordinates rather than district names.
- **Alternative:** N/A

### 8. GOOGLE PLAY
- **Status:** CHANGED
- **Current free limit:** $25 one-time registration fee remains.
- **Breaking changes since 2024:** Google now requires new individual developer accounts to complete a closed testing period with at least 12 testers for 14 consecutive days before production access is granted.
- **Alternative:** Use APK sideloading or third-party distribution (like WhatsApp) for initial pre-launch farmer testing.

### 9. TRANSLATION (LibreTranslate)
- **Status:** BROKEN (BLOCKER)
- **Current free limit:** Self-hosted is free.
- **Breaking changes since 2024:** LibreTranslate natively supports Hindi, but it **DOES NOT** officially support Kannada, Telugu, Tamil, or Marathi out of the box.
- **Alternative:** **Bhashini API** (Indian Govt initiative, free for low-volume/prototyping) OR bypass dedicated translation services and use **Groq (Llama 3.1 70B/8B)** directly for both translation and response generation, as the model inherently supports these languages.

### 10. MONITORING (Sentry & UptimeRobot)
- **Status:** CONFIRMED FREE
- **Current free limit:** Sentry: 5,000 errors/month. UptimeRobot: 50 monitors at 5-minute intervals.
- **Breaking changes since 2024:** UptimeRobot has introduced stricter restrictions regarding commercial use on their free tier.
- **Alternative:** N/A
