# FarmGenius Feedback System & Field Testing

This document contains the implementation details for the in-app voice feedback system, the field testing protocol for rural farmers, sprint tracking, and APK distribution guidelines.

## 1. Flutter Feedback Widget (`lib/shared/widgets/feedback_widget.dart`)

```dart
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'package:dio/dio.dart';

class FeedbackWidget extends StatefulWidget {
  final String queryId;
  final String language;

  const FeedbackWidget({Key? key, required this.queryId, required this.language}) : super(key: key);

  @override
  _FeedbackWidgetState createState() => _FeedbackWidgetState();
}

class _FeedbackWidgetState extends State<FeedbackWidget> {
  final FlutterTts _flutterTts = FlutterTts();
  final SpeechToText _speechToText = SpeechToText();
  bool _isListening = false;
  bool _submitted = false;
  String _followUpText = "";

  @override
  void initState() {
    super.initState();
    _speechToText.initialize();
  }

  Future<void> _submitFeedback(bool wasHelpful) async {
    if (wasHelpful) {
      await _flutterTts.setLanguage("en-US");
      await _flutterTts.speak("Helpful!");
      await _sendToBackend(true, "");
      setState(() => _submitted = true);
    } else {
      await _flutterTts.setLanguage("en-US");
      await _flutterTts.speak("Not helpful");
      await Future.delayed(const Duration(seconds: 1));
      
      await _flutterTts.setLanguage("hi-IN");
      await _flutterTts.speak("Aapko aur kya chahiye tha?");
      
      await Future.delayed(const Duration(seconds: 2));

      bool available = await _speechToText.initialize();
      if (available) {
        setState(() => _isListening = true);
        _speechToText.listen(
          onResult: (result) async {
            if (result.finalResult) {
              setState(() {
                _isListening = false;
                _followUpText = result.recognizedWords;
              });
              await _sendToBackend(false, _followUpText);
              setState(() => _submitted = true);
            }
          },
          localeId: 'hi_IN',
          listenFor: const Duration(seconds: 10),
        );
      } else {
        await _sendToBackend(false, "");
        setState(() => _submitted = true);
      }
    }
  }

  Future<void> _sendToBackend(bool wasHelpful, String followUpText) async {
    try {
      final dio = Dio();
      // Replace with configured base URL
      await dio.post('https://farmgenius-backend.onrender.com/feedback/', data: {
        "query_id": widget.queryId,
        "was_helpful": wasHelpful,
        "follow_up_text": followUpText,
        "language": widget.language,
      });
    } catch (e) {
      // Gracefully ignore network errors for feedback
      print("Failed to send feedback: \$e");
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_submitted) {
      return const Padding(
        padding: EdgeInsets.all(16.0),
        child: Text("Thank you for your feedback! / धन्यवाद!", 
          style: TextStyle(fontSize: 18, color: Colors.green, fontWeight: FontWeight.bold),
          textAlign: TextAlign.center,
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 24.0),
      child: Column(
        children: [
          const Text("Was this helpful?", style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  minimumSize: const Size(120, 100),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                ),
                onPressed: _isListening ? null : () => _submitFeedback(true),
                child: const Icon(Icons.thumb_up, size: 50, color: Colors.white),
              ),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.red,
                  minimumSize: const Size(120, 100),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                ),
                onPressed: _isListening ? null : () => _submitFeedback(false),
                child: const Icon(Icons.thumb_down, size: 50, color: Colors.white),
              ),
            ],
          ),
          if (_isListening)
            const Padding(
              padding: EdgeInsets.only(top: 20.0),
              child: Text("Listening... / सुन रहा हूँ...", 
                style: TextStyle(fontSize: 20, color: Colors.red, fontWeight: FontWeight.bold)),
            ),
        ],
      ),
    );
  }
}
```

## 2. Backend `/feedback` Endpoint Code (`backend/app/api/routes/feedback.py`)

```python
from fastapi import APIRouter, Depends, BackgroundTasks
from pydantic import BaseModel
import httpx
import os
from app.core.security import get_current_user, supabase

router = APIRouter()

class FeedbackRequest(BaseModel):
    query_id: str
    was_helpful: bool
    follow_up_text: str = ""
    language: str

def notify_slack(feedback: FeedbackRequest):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return
    message = {
        "text": (
            f"🚨 *Negative Feedback Alert* 🚨\n"
            f"*Query ID:* {feedback.query_id}\n"
            f"*Language:* {feedback.language}\n"
            f"*User Follow-up:* {feedback.follow_up_text}"
        )
    }
    try:
        httpx.post(webhook_url, json=message, timeout=5.0)
    except Exception as e:
        print(f"Failed to send Slack notification: {e}")

@router.post("/")
async def submit_feedback(req: FeedbackRequest, background_tasks: BackgroundTasks, user_id: str = Depends(get_current_user)):
    # Store in Supabase feedback table
    try:
        supabase.table("feedback").insert({
            "query_id": req.query_id,
            "was_helpful": req.was_helpful,
            "follow_up_action": req.follow_up_text,
            "timestamp": "now()"
        }).execute()
    except Exception as e:
        print(f"Failed to save feedback to DB: {e}")
        
    # Trigger Slack webhook notification to developer if not helpful
    if not req.was_helpful:
        background_tasks.add_task(notify_slack, req)

    return {"status": "success"}
```

*(Note: Ensure this router is included in `backend/main.py` like the other routes: `app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])`)*

## 3. FIELD_TEST_PROTOCOL.md

This protocol is designed for non-technical facilitators conducting tests with rural Indian farmers.

### a) PRE-SESSION (30 minutes before)
- Fully charge the Android test device.
- Ensure the app has the downloaded TFLite model and latest UI via APK install.
- Disable Wi-Fi and set the phone network to 2G/3G to simulate actual village internet conditions.
- Clear previous cached data (`App Info -> Storage -> Clear Data`) to ensure a fresh session.
- Turn media volume to MAX so voice responses are clearly audible.

### b) SESSION SCRIPT (45 minutes with one farmer)
- **Introduction**: "Namaste! We are building an app that talks instead of typing. I want to see if it actually works for you. I will not tell you how to use it, just try it like a new tool."
- **Task 1: Voice Question**: "Imagine your tomato leaves are turning yellow. Use this app to ask what to do." *(Observe: Do they press and hold? Do they wait for the beep? Do they speak naturally?)*
- **Task 2: Crop Disease Camera**: Hand them a leaf (sick or healthy). "Show this leaf to the app and see what it says."
- **Task 3: Mandi Prices**: "Check the price of wheat in your nearest mandi." *(Observe if they can navigate to the market section and understand the offline cached message).*
- **Silent Observation**: Do not interrupt. Only intervene if the app crashes. Note any hesitation or confusing UI elements.

### c) POST-SESSION (10 minutes — 3 spoken questions only)
Ask these questions and note the answers directly (do not ask them to fill out a form):
1. "Did the app understand your accent/dialect?"
2. "Would you trust the medicine/chemical this app told you to buy?"
3. "What is one thing that annoyed you?"

### d) WHAT TO RECORD
- Record the screen using Android's built-in Screen Recorder (with mic enabled to hear their query and the app's TTS).
- Write down exact quotes of what they said during the post-session.
- Note any Network Timeouts or crashes.
- Record whether the "thumbs down -> voice feedback" loop successfully triggered and recorded their complaint.

### e) AFTER 5 SESSIONS — TRIAGE MEETING
- Gather all the feedback, screen recordings, and Slack notifications.
- Categorize issues into: "Critical UX blockers", "Model inaccuracy", "Network timeout issues".
- Decide what to fix in the next 2-week sprint before testing with the next 5 farmers.

## 4. SPRINT_TRACKER.md

| Sprint | Goal | Deliverables | Status | Notes |
|--------|------|--------------|--------|-------|
| **Sprint 1** | **Core App & Feedback System** | Voice Chat, Local ML Model (TFLite), 2G Resilient APIs, Voice Feedback Widget | **Done** | Initial prototype for field testing, focusing on offline-first capabilities and voice UX. Backend & Flutter integrated. |
| Sprint 2 | Field Testing & Iteration | Fix Top 3 UX issues from 5 farmer tests, Improve Hindi/regional model accuracy | To Do | Dependent on Sprint 1 field test results. |
| Sprint 3 | Expansion & Caching | Offline Market Data, Weather Advisory Push Notifications | To Do | Focus on data consistency and aggressive caching strategies. |
| Sprint 4 | Soft Launch | 100 Farmers WhatsApp Distribution, Slack Monitoring | To Do | Full pilot with analytics tracking. |

## 5. WhatsApp APK Distribution Guide

Since Play Store reviews take time and many farmers sideload apps, we distribute the initial beta via WhatsApp.

1. **Build the Release APK**: 
   Run `flutter build apk --release` (or download it from the GitHub Actions artifact). Ensure it is a 'fat APK' (contains both `arm64` and `armeabi-v7a` architectures) so it works on any older Android phone.
2. **Rename the APK**: 
   Rename `app-release.apk` to something friendly, e.g., `FarmGenius_v1.apk`.
3. **Draft the WhatsApp Message**:
   Keep it short, use local language, and use emojis. Example (Hindi):
   *Namaste 🙏! FarmGenius app ab taiyaar hai. Is app se aap bolkar kheti ke sawal pooch sakte hain aur bimari wali patti ka photo khinch kar ilaj jaan sakte hain.*
   *App install karne ke liye is file par click karein. Agar 'Install unknown apps' ka message aaye, toh 'Settings' mein jakar 'Allow' kar dein.*
4. **Attach and Send**: 
   Send the `.apk` file directly as a "Document" on WhatsApp to the farmer or the farmer group. WhatsApp allows document sharing up to 100MB, and our APK is lightweight (~15MB).
5. **Video Tutorial (Crucial)**: 
   Send a 30-second screen-recording video right after the APK. Show them *exactly* how to click the file, click "Install", bypass the Google Play Protect warning ("Install anyway"), and open the app. Farmers will likely ignore text instructions, so the visual guide is necessary.
