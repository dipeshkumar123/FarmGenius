# FarmGenius Flutter Application Code

This document contains the complete Dart codebase, `pubspec.yaml`, and build instructions for the FarmGenius Android app, tailored for rural Indian farmers with offline capabilities and voice-first UX.

## TASK 1 & 2 — PROJECT SETUP & PUB SPEC

### `app/pubspec.yaml`
```yaml
name: farmgenius
description: An agricultural advisor app for Indian smallholder farmers.
publish_to: 'none'
version: 1.0.0+1

environment:
  sdk: '>=3.2.0 <4.0.0'

dependencies:
  flutter:
    sdk: flutter
  flutter_riverpod: ^2.4.9
  go_router: ^12.1.1
  hive: ^2.2.3
  hive_flutter: ^1.1.0
  dio: ^5.4.0
  tflite_flutter: ^0.10.4
  speech_to_text: ^6.5.1
  flutter_tts: ^3.8.3
  image_picker: ^1.0.7
  path_provider: ^2.1.2
  permission_handler: ^11.2.0
  image: ^4.1.3

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.1

flutter:
  uses-material-design: true
  assets:
    - assets/models/model_quant.tflite
    - assets/models/labels_indian_crops.txt
```

## TASK 3 — CORE SCREENS & ENTRY POINT

### `app/lib/main.dart`
```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'core/storage/offline_manager.dart';
import 'features/auth/onboarding_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize Offline Storage (Hive)
  await Hive.initFlutter();
  await OfflineManager.init();
  
  runApp(const ProviderScope(child: FarmGeniusApp()));
}

class FarmGeniusApp extends StatelessWidget {
  const FarmGeniusApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FarmGenius',
      theme: ThemeData(
        primarySwatch: Colors.green,
        // Enforcing minimum text size 18sp for readability
        textTheme: const TextTheme(
          bodyLarge: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
          bodyMedium: TextStyle(fontSize: 18),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            minimumSize: const Size(double.infinity, 60),
            textStyle: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
        ),
      ),
      home: const OnboardingScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}
```

### `app/lib/features/auth/onboarding_screen.dart`
```dart
import 'package:flutter/material.dart';
import '../chat/voice_chat_screen.dart';
import '../disease/disease_detect_screen.dart';
import '../market/mandi_prices_screen.dart';
import '../weather/weather_screen.dart';

class OnboardingScreen extends StatelessWidget {
  const OnboardingScreen({Key? key}) : super(key: key);

  void _navigateTo(BuildContext context, Widget screen) {
    Navigator.push(context, MaterialPageRoute(builder: (_) => screen));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text('FarmGenius / फार्म जीनियस'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.agriculture, size: 120, color: Colors.green),
            const SizedBox(height: 30),
            ElevatedButton.icon(
              onPressed: () => _navigateTo(context, const VoiceChatScreen()),
              icon: const Icon(Icons.mic, size: 30),
              label: const Text('Ask a Question (Voice)'),
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: () => _navigateTo(context, const DiseaseDetectScreen()),
              icon: const Icon(Icons.camera_alt, size: 30),
              label: const Text('Scan Crop Disease'),
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: () => _navigateTo(context, const MandiPricesScreen()),
              icon: const Icon(Icons.store, size: 30),
              label: const Text('Check Mandi Prices'),
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: () => _navigateTo(context, const WeatherScreen()),
              icon: const Icon(Icons.cloud, size: 30),
              label: const Text('Weather & Advisory'),
            ),
          ],
        ),
      ),
    );
  }
}
```

### `app/lib/features/chat/voice_chat_screen.dart`
```dart
import 'package:flutter/material.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'package:flutter_tts/flutter_tts.dart';
import '../../core/network/api_service.dart';

class VoiceChatScreen extends StatefulWidget {
  const VoiceChatScreen({Key? key}) : super(key: key);

  @override
  _VoiceChatScreenState createState() => _VoiceChatScreenState();
}

class _VoiceChatScreenState extends State<VoiceChatScreen> {
  final SpeechToText _speechToText = SpeechToText();
  final FlutterTts _flutterTts = FlutterTts();
  final ApiService _apiService = ApiService();
  
  bool _isListening = false;
  String _userText = 'Press the mic to ask a question.';
  String _assistantResponse = '';
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _initSpeech();
  }

  void _initSpeech() async {
    await _speechToText.initialize();
    setState(() {});
  }

  void _startListening() async {
    await _speechToText.listen(
      onResult: (result) {
        setState(() {
          _userText = result.recognizedWords;
        });
      },
      localeId: 'hi_IN', // Default to Hindi, can be dynamic
    );
    setState(() => _isListening = true);
  }

  void _stopListening() async {
    await _speechToText.stop();
    setState(() {
      _isListening = false;
      _isLoading = true;
    });
    
    // Call Backend
    try {
      final response = await _apiService.sendChatQuery(_userText, 'hi', 'farmer_123');
      setState(() {
        _assistantResponse = response['response'];
        _isLoading = false;
      });
      _speak(_assistantResponse);
    } catch (e) {
      setState(() {
        _assistantResponse = "Offline mode. Please contact KVK.";
        _isLoading = false;
      });
      _speak("Internet is down. Please contact your local KVK.");
    }
  }

  void _speak(String text) async {
    await _flutterTts.setLanguage("hi-IN");
    await _flutterTts.speak(text);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Voice Advisor')),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.grey[200],
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(_userText, style: const TextStyle(fontSize: 20)),
                    ),
                    const SizedBox(height: 20),
                    if (_isLoading) const Center(child: CircularProgressIndicator()),
                    if (_assistantResponse.isNotEmpty)
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: Colors.green[100],
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(_assistantResponse, style: const TextStyle(fontSize: 20)),
                      ),
                  ],
                ),
              ),
            ),
            GestureDetector(
              onTapDown: (_) => _startListening(),
              onTapUp: (_) => _stopListening(),
              child: CircleAvatar(
                radius: 60,
                backgroundColor: _isListening ? Colors.red : Colors.green,
                child: const Icon(Icons.mic, size: 60, color: Colors.white),
              ),
            ),
            const SizedBox(height: 20),
            const Text('Hold to Speak', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
          ],
        ),
      ),
    );
  }
}
```

### `app/lib/features/disease/disease_detect_screen.dart`
```dart
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'tflite_service.dart';

class DiseaseDetectScreen extends StatefulWidget {
  const DiseaseDetectScreen({Key? key}) : super(key: key);

  @override
  _DiseaseDetectScreenState createState() => _DiseaseDetectScreenState();
}

class _DiseaseDetectScreenState extends State<DiseaseDetectScreen> {
  File? _image;
  String _result = 'Capture a photo of the sick leaf';
  final TFLiteService _tfliteService = TFLiteService();
  bool _isModelLoaded = false;

  @override
  void initState() {
    super.initState();
    _initModel();
  }
  
  Future<void> _initModel() async {
    await _tfliteService.initModel();
    setState(() => _isModelLoaded = true);
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.camera, imageQuality: 80);
    
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _result = 'Analyzing locally...';
      });
      _analyzeImage(_image!);
    }
  }

  Future<void> _analyzeImage(File image) async {
    final prediction = await _tfliteService.predict(image);
    setState(() {
      _result = prediction;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Crop Doctor')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _image == null 
                  ? const Icon(Icons.yard, size: 120, color: Colors.grey)
                  : Image.file(_image!, height: 300, fit: BoxFit.cover),
              const SizedBox(height: 30),
              Text(
                _result, 
                style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold), 
                textAlign: TextAlign.center
              ),
              const Spacer(),
              ElevatedButton.icon(
                onPressed: _isModelLoaded ? _pickImage : null,
                icon: const Icon(Icons.camera),
                label: const Text('Take Photo (Works Offline)'),
              ),
              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }
}
```

### `app/lib/features/market/mandi_prices_screen.dart`
```dart
import 'package:flutter/material.dart';
import '../../core/storage/offline_manager.dart';
import '../../core/network/api_service.dart';

class MandiPricesScreen extends StatefulWidget {
  const MandiPricesScreen({Key? key}) : super(key: key);

  @override
  _MandiPricesScreenState createState() => _MandiPricesScreenState();
}

class _MandiPricesScreenState extends State<MandiPricesScreen> {
  Map<String, dynamic>? _prices;
  final ApiService _apiService = ApiService();
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadPrices();
  }

  Future<void> _loadPrices() async {
    // 1. Try loading from offline cache first
    final cached = OfflineManager.getPrices();
    if (cached != null) {
      setState(() {
        _prices = Map<String, dynamic>.from(cached);
        _isLoading = false;
      });
    }

    // 2. Fetch fresh data from network in background
    try {
      final freshData = await _apiService.fetchPrices('wheat', 'dharwad', 'karnataka');
      await OfflineManager.savePrices(freshData);
      setState(() {
        _prices = freshData;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
      // Fails silently, uses cached data
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Mandi Prices')),
      body: Center(
        child: _isLoading && _prices == null
            ? const CircularProgressIndicator()
            : _prices == null
                ? const Text('No offline data available.', style: TextStyle(fontSize: 20))
                : Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.store, size: 100, color: Colors.orange),
                      const SizedBox(height: 20),
                      Text('${_prices!['commodity']} in ${_prices!['district']}', style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 20),
                      Text('₹${_prices!['modal_price']} / Quintal', style: const TextStyle(fontSize: 40, fontWeight: FontWeight.bold, color: Colors.green)),
                      const SizedBox(height: 10),
                      Text('Date: ${_prices!['date']}', style: const TextStyle(fontSize: 18, color: Colors.grey)),
                      const SizedBox(height: 40),
                      const Text('Showing latest available offline data.', style: TextStyle(fontSize: 16, color: Colors.grey)),
                    ],
                  ),
      ),
    );
  }
}
```

### `app/lib/features/weather/weather_screen.dart`
```dart
import 'package:flutter/material.dart';
import '../../core/network/api_service.dart';
import '../../core/storage/offline_manager.dart';

class WeatherScreen extends StatefulWidget {
  const WeatherScreen({Key? key}) : super(key: key);

  @override
  _WeatherScreenState createState() => _WeatherScreenState();
}

class _WeatherScreenState extends State<WeatherScreen> {
  final ApiService _apiService = ApiService();
  List<dynamic> _forecast = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadWeather();
  }

  Future<void> _loadWeather() async {
    final cached = OfflineManager.getWeather();
    if (cached != null && cached.isNotEmpty) {
      setState(() {
        _forecast = List.from(cached);
        _isLoading = false;
      });
    }

    try {
      final freshData = await _apiService.fetchWeather('dharwad', 'karnataka');
      await OfflineManager.saveWeather(freshData);
      setState(() {
        _forecast = freshData;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Weather Advisory')),
      body: _isLoading && _forecast.isEmpty
          ? const Center(child: CircularProgressIndicator())
          : _forecast.isEmpty
              ? const Center(child: Text('No offline weather data.', style: TextStyle(fontSize: 20)))
              : ListView.builder(
                  padding: const EdgeInsets.all(16.0),
                  itemCount: _forecast.length,
                  itemBuilder: (context, index) {
                    final day = _forecast[index];
                    return Card(
                      margin: const EdgeInsets.only(bottom: 16),
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text('Date: ${day['date']}', style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                                Text('${day['max_temp']}°C', style: const TextStyle(fontSize: 24, color: Colors.red)),
                              ],
                            ),
                            const SizedBox(height: 10),
                            Text('Rainfall: ${day['rainfall_mm']} mm', style: const TextStyle(fontSize: 18)),
                            const SizedBox(height: 10),
                            Text('Advisory: ${day['farming_advisory']}', style: const TextStyle(fontSize: 18, color: Colors.green, fontWeight: FontWeight.bold)),
                          ],
                        ),
                      ),
                    );
                  },
                ),
    );
  }
}
```

## TASK 4 — TFLITE SERVICE

### `app/lib/features/disease/tflite_service.dart`
```dart
import 'dart:io';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class TFLiteService {
  Interpreter? _interpreter;
  List<String>? _labels;

  Future<void> initModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/models/model_quant.tflite');
      // In production, load actual labels.txt here.
      _labels = ["Healthy Crop", "Tomato Early Blight", "Potato Late Blight", "Unknown"];
    } catch (e) {
      print('Error loading offline model: $e');
    }
  }

  Future<String> predict(File imageFile) async {
    if (_interpreter == null) return "Model not loaded. Try restarting app.";

    // Pre-processing
    var imageBytes = await imageFile.readAsBytes();
    img.Image? originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) return "Failed to process image.";
    
    // Resize to match MobileNetV2 input shape (224, 224, 3)
    img.Image resizedImage = img.copyResize(originalImage, width: 224, height: 224);

    // Prepare float32 tensor
    var input = List.generate(1, (i) => List.generate(224, (j) => List.generate(224, (k) => List.generate(3, (l) => 0.0))));
    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        var pixel = resizedImage.getPixel(x, y);
        input[0][y][x][0] = pixel.r / 255.0;
        input[0][y][x][1] = pixel.g / 255.0;
        input[0][y][x][2] = pixel.b / 255.0;
      }
    }

    // Output shape (1, NUM_CLASSES)
    var output = List.filled(1 * 4, 0.0).reshape([1, 4]);
    _interpreter!.run(input, output);

    // Argmax
    int maxIdx = 0;
    double maxVal = output[0][0];
    for (int i = 1; i < output[0].length; i++) {
      if (output[0][i] > maxVal) {
        maxVal = output[0][i];
        maxIdx = i;
      }
    }

    String label = _labels != null && maxIdx < _labels!.length ? _labels![maxIdx] : "Unknown";
    return "$label\nConfidence: ${(maxVal * 100).toStringAsFixed(1)}%";
  }
}
```

## TASK 5 — OFFLINE STRATEGY

### `app/lib/core/storage/offline_manager.dart`
```dart
import 'package:hive_flutter/hive_flutter.dart';

class OfflineManager {
  static const String queriesBox = 'queries';
  static const String pricesBox = 'prices';
  static const String weatherBox = 'weather';

  static Future<void> init() async {
    await Hive.openBox(queriesBox);
    await Hive.openBox(pricesBox);
    await Hive.openBox(weatherBox);
  }

  // Prices Cache
  static Future<void> savePrices(Map<String, dynamic> prices) async {
    final box = Hive.box(pricesBox);
    await box.put('latest_prices', prices);
  }

  static Map<dynamic, dynamic>? getPrices() {
    return Hive.box(pricesBox).get('latest_prices');
  }

  // Weather Cache
  static Future<void> saveWeather(List<dynamic> forecast) async {
    final box = Hive.box(weatherBox);
    await box.put('latest_weather', forecast);
  }

  static List<dynamic>? getWeather() {
    return Hive.box(weatherBox).get('latest_weather');
  }
}
```

## TASK 6 — API SERVICE WITH 2G RESILIENCE

### `app/lib/core/network/api_service.dart`
```dart
import 'package:dio/dio.dart';

class ApiService {
  late final Dio _dio;
  
  ApiService() {
    _dio = Dio(BaseOptions(
      baseUrl: 'https://farmgenius-backend.onrender.com', // Replace with dynamic env config later
      connectTimeout: const Duration(seconds: 30), // Extended for 2G
      receiveTimeout: const Duration(seconds: 30),
    ));

    // Custom Interceptor for 2G retries
    _dio.interceptors.add(InterceptorsWrapper(
      onError: (DioException err, ErrorInterceptorHandler handler) async {
        if (_shouldRetry(err) && (err.requestOptions.extra['retryCount'] ?? 0) < 3) {
          err.requestOptions.extra['retryCount'] = (err.requestOptions.extra['retryCount'] ?? 0) + 1;
          try {
            // Wait before retry
            await Future.delayed(const Duration(seconds: 3));
            final response = await _dio.fetch(err.requestOptions);
            return handler.resolve(response);
          } catch (e) {
            return handler.next(err);
          }
        }
        return handler.next(err);
      },
    ));
  }

  bool _shouldRetry(DioException err) {
    return err.type == DioExceptionType.connectionTimeout || 
           err.type == DioExceptionType.receiveTimeout || 
           err.type == DioExceptionType.unknown;
  }

  Future<Map<String, dynamic>> sendChatQuery(String query, String language, String farmerId) async {
    final response = await _dio.post('/chat/', data: {
      "query": query,
      "language": language,
      "farmer_id": farmerId
    });
    return response.data;
  }

  Future<Map<String, dynamic>> fetchPrices(String commodity, String district, String state) async {
    final response = await _dio.get('/prices/', queryParameters: {
      "commodity": commodity,
      "district": district,
      "state": state
    });
    return response.data;
  }

  Future<List<dynamic>> fetchWeather(String district, String state) async {
    final response = await _dio.get('/weather/', queryParameters: {
      "district": district,
      "state": state
    });
    return response.data;
  }
}
```

## TASK 7 — BUILD & DISTRIBUTION

### `.github/workflows/build_apk.yml`
```yaml
name: Build Android APK for Distribution

on:
  push:
    branches:
      - main
    paths:
      - 'app/**'
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v3

      - name: Setup Java Environment
        uses: actions/setup-java@v3
        with:
          distribution: 'zulu'
          java-version: '17'

      - name: Setup Flutter
        uses: subosito/flutter-action@v2
        with:
          channel: 'stable'
          flutter-version: '3.19.0' # Pinned version as per STACK.md principles

      - name: Install Dependencies
        run: flutter pub get
        working-directory: ./app

      - name: Build APK (Release mode)
        # Builds a fat APK containing ARM and ARM64 suitable for WhatsApp distribution
        run: flutter build apk --release
        working-directory: ./app

      - name: Upload APK Artifact
        uses: actions/upload-artifact@v3
        with:
          name: farmgenius-app-release
          path: app/build/app/outputs/flutter-apk/app-release.apk
          retention-days: 14
```
