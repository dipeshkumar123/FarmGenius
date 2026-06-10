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
