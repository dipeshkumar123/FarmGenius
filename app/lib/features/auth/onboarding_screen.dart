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
