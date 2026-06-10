# FarmGenius Build Plan

## 1. Conflicts Detected
**CONFLICT 1 — [Architecture Agent] vs [Trust Content Agent] — `diseases` Table Schema Mismatch:**
The `ARCHITECTURE.md` schema defines the `diseases` table with columns `(id, name, crop, symptoms_keywords, treatment, image_url, source_kvk)`. However, `TRUST_CONTENT.md` inserts data using completely different columns: `(disease_id, crop, disease_name_en, disease_name_hi, symptoms_farmer_language, organic_treatment, chemical_treatment, dosage, source_url, source_name)`.

**CONFLICT 2 — [Backend Agent] vs [Trust Content Agent] — `/disease/detect` Endpoint Response Mismatch:**
In `BACKEND_CODE.md`, the `/disease/detect` endpoint returns a `DiseaseResponse` containing `treatment` and `source_kvk`. This does not map correctly to the granular treatment fields (`organic_treatment`, `chemical_treatment`, `dosage`) defined by the Trust Content Agent.

## 2. Gaps Detected
**GAP 1 — `ARCHITECTURE.md` — Missing `schemes` and `kvk_directory` Tables:**
Both `BACKEND_CODE.md` and `TRUST_CONTENT.md` rely on `schemes` and `kvk_directory` tables, but the Architecture Agent failed to include them in the initial Supabase SQL schema design.

**GAP 2 — `BACKEND_CODE.md` — Missing `/feedback` Route Registration:**
The Feedback System Agent defined the `/feedback` route code, but `backend/main.py` does not import `feedback.py` or register its router (`app.include_router(feedback.router)`).

**GAP 3 — `BACKEND_CODE.md` — Missing `SLACK_WEBHOOK_URL` in `.env.example`:**
The `/feedback` endpoint triggers a Slack notification using the `SLACK_WEBHOOK_URL` environment variable, but this is missing from the backend `.env.example` file.

**GAP 4 — `BACKEND_CODE.md` — Missing `districts_coords.json`:**
`ARCHITECTURE.md` explicitly specifies maintaining a `districts_coords.json` file for weather queries, but the Backend Agent hardcoded a small dictionary in `weather_service.py` instead of creating the JSON file.

## 3. Developer Task List (Priority Order)

**Week 1 (Get something on a farmer's phone)**
1. **Resolve Database Schemas:** Unify the Supabase table definitions to accommodate all granular fields required by the `TRUST_CONTENT.md` disease, schemes, and KVK directory data.
2. **Setup Backend Foundation:** Initialize the FastAPI project, add `SLACK_WEBHOOK_URL` to `.env.example`, and register the `/feedback` router in `main.py`.
3. **Core API Implementation:** Implement the `/chat` endpoint using Groq Llama-3.1 to enable voice-to-response workflows.
4. **Deploy Backend:** Push FastAPI to Render free tier and configure the GitHub Actions cron keep-alive to prevent cold starts.
5. **Setup Flutter App:** Initialize the Flutter project with Riverpod, GoRouter, `speech_to_text`, and `flutter_tts`. 
6. **Sprint 1 Feature Build:** Build the Voice Chat UI and wire it to the `/chat` endpoint.
7. **First Distribution:** Build a release APK and distribute it via WhatsApp to the first set of 5 test farmers.

**Week 2–3 (Make it useful)**
8. **Train Disease Model:** Run the Colab notebook to train and quantize the MobileNetV2 TFLite model using the PlantVillage dataset.
9. **Implement Crop Doctor UI:** Integrate `tflite_flutter` into the app and build the offline camera disease detection screen.
10. **Populate Supabase:** Run the SQL insert scripts from `TRUST_CONTENT.md` to seed diseases, KVKs, and schemes.
11. **Implement Remaining APIs:** Build the `/prices`, `/weather`, and `/schemes` endpoints with the in-memory caching logic.
12. **Build Feedback Loop:** Integrate the Voice Feedback Widget into the Flutter app and connect it to the `/feedback` endpoint.

**Week 4+ (Make it trustworthy and scalable)**
13. **Train NLP Fallback:** Retrain the local `MultinomialNB` chatbot using the dialect-rich queries in `FARMER_CORPUS.md`.
14. **Offline Sync & Caching:** Implement Hive local storage to cache weather, prices, and chat history for full offline degradation.
15. **CI/CD Pipeline:** Finalize GitHub Actions for automated APK builds on Pull Requests and automatic backend deployments on merge.
16. **OTA Updates Strategy:** Implement dynamic downloading of new `.tflite` model versions from Supabase to prevent forcing large APK updates.

## 4. Sprint 1 Definition
**Scope:** "Voice → disease detection (Kannada) → spoken diagnosis"
- **Goal:** Deliver an end-to-end voice-driven interaction that diagnoses a crop issue entirely in a regional language without typing.
- **Workflow:**
  1. The farmer opens the app and presses the microphone button.
  2. The farmer describes a crop symptom in Kannada (e.g., "Tomato leaves are turning yellow").
  3. `speech_to_text` captures the Kannada speech and sends it to the backend `/chat` endpoint.
  4. The Groq API analyzes the symptom, matches it against known agricultural advice, and generates a treatment response directly in Kannada.
  5. The Flutter app receives the response and reads it aloud using `flutter_tts` in Kannada.
- **Timeframe:** 2 weeks.

## 5. Success Metrics
- **Technical:** The app installs and runs on an Android 6+ device without crashing.
- **UX:** The farmer successfully completes the voice-based disease detect task without requiring manual assistance or typing.
- **Trust:** The farmer explicitly states they would retake a photo or rely on the app's diagnosis for a real crop issue.
- **Impact:** At least 1 out of 5 test farmers states they would act differently (e.g., use a different pesticide or dosage) based on the app's advice.

## 6. Risk Register
1. **Risk:** 2G network timeouts cause API requests (like Groq generation) to fail.
   - *Mitigation:* Set generous 30s connection timeouts in `dio`, implement retry interceptors, and default gracefully to the offline TTS error message.
2. **Risk:** The `tflite` model size causes device storage strain or memory crashes on low-end phones.
   - *Mitigation:* Aggressively use INT8 post-training quantization to force the model under 15MB.
3. **Risk:** Groq Llama-3.1 fails to comprehend heavy rural dialects (e.g., confusing "jhulsa" for something else).
   - *Mitigation:* Pass the dialect glossary from `FARMER_CORPUS.md` directly into the system prompt context.
4. **Risk:** Farmers mistrust the AI's chemical dosage recommendations.
   - *Mitigation:* Always append the source KVK/ICAR authority to every generated response and include a "Verify with local dealer" disclaimer.
5. **Risk:** Render's free tier spins down, causing 60-second cold starts which break the voice UX flow.
   - *Mitigation:* Maintain the GitHub Actions cron job to ping the `/health` endpoint every 10 minutes.

## 7. Founder Summary
FarmGenius is architected to survive the realities of rural Indian connectivity. We’re building an Android app that completely bypasses the digital literacy barrier by using a Voice-First interface (listening and speaking in regional dialects) and an Offline-First disease scanner that runs directly on the phone without internet. By stitching together a completely free tech stack—combining Llama 3.1 for intelligent dialect translation, quantized machine learning for plant health, and data.gov.in for real-time market prices—we have a production-ready blueprint. Our immediate goal is Sprint 1: proving a Kannada-speaking farmer can press one button, describe a sick plant, and hear an accurate, ICAR-backed treatment within seconds.
