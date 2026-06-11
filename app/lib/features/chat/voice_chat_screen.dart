import 'package:flutter/material.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:flutter_animate/flutter_animate.dart';
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
  String _currentSpeech = '';
  List<Map<String, dynamic>> _messages = [
    {'text': 'Namaste! How can I help you today?', 'isUser': false},
  ];
  bool _isLoading = false;
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _initSpeech();
  }

  void _initSpeech() async {
    await _speechToText.initialize();
    setState(() {});
  }

  void _scrollToBottom() {
    if (_scrollController.hasClients) {
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    }
  }

  void _startListening() async {
    await _flutterTts.stop(); // Stop any ongoing speech
    setState(() {
      _isListening = true;
      _currentSpeech = '';
    });
    
    await _speechToText.listen(
      onResult: (result) {
        setState(() {
          _currentSpeech = result.recognizedWords;
        });
      },
      localeId: 'hi_IN',
    );
  }

  void _stopListening() async {
    await _speechToText.stop();
    setState(() {
      _isListening = false;
    });

    if (_currentSpeech.trim().isEmpty) return;

    final userMessage = _currentSpeech;
    setState(() {
      _messages.add({'text': userMessage, 'isUser': true});
      _currentSpeech = '';
      _isLoading = true;
    });
    
    Future.delayed(const Duration(milliseconds: 100), _scrollToBottom);

    try {
      final response = await _apiService.sendChatQuery(userMessage, 'hi', 'farmer_123');
      final assistantText = response['response'] ?? 'Sorry, I could not understand.';
      
      setState(() {
        _messages.add({'text': assistantText, 'isUser': false});
        _isLoading = false;
      });
      _speak(assistantText);
    } catch (e) {
      final errorText = "Internet is down. Please contact your local KVK.";
      setState(() {
        _messages.add({'text': "Offline mode. Please contact KVK.", 'isUser': false});
        _isLoading = false;
      });
      _speak(errorText);
    }
    Future.delayed(const Duration(milliseconds: 100), _scrollToBottom);
  }

  void _speak(String text) async {
    await _flutterTts.setLanguage("hi-IN");
    await _flutterTts.speak(text);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAF8),
      appBar: AppBar(
        title: const Text('Voice Advisor', style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.white,
        foregroundColor: const Color(0xFF2E7D32),
        elevation: 0,
        centerTitle: true,
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length + (_currentSpeech.isNotEmpty ? 1 : 0),
              itemBuilder: (context, index) {
                if (index == _messages.length) {
                  return _buildChatBubble(text: _currentSpeech, isUser: true, isTranscribing: true);
                }
                final msg = _messages[index];
                return _buildChatBubble(
                  text: msg['text'],
                  isUser: msg['isUser'],
                ).animate().fade().slideY(begin: 0.1, duration: 300.ms);
              },
            ),
          ),
          if (_isLoading)
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: const CircularProgressIndicator(color: Color(0xFF2E7D32))
                  .animate()
                  .fade(),
            ),
          // Voice Input Area
          Container(
            padding: const EdgeInsets.only(top: 16, bottom: 32, left: 24, right: 24),
            decoration: BoxDecoration(
              color: Colors.white,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.05),
                  blurRadius: 10,
                  offset: const Offset(0, -5),
                )
              ],
              borderRadius: const BorderRadius.vertical(top: Radius.circular(32)),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  _isListening ? 'Listening...' : 'Hold to Speak',
                  style: TextStyle(
                    fontSize: 16,
                    color: _isListening ? Colors.red : Colors.grey[600],
                    fontWeight: FontWeight.bold,
                  ),
                ).animate(target: _isListening ? 1 : 0).fade(),
                const SizedBox(height: 16),
                GestureDetector(
                  onTapDown: (_) => _startListening(),
                  onTapUp: (_) => _stopListening(),
                  onTapCancel: () => _stopListening(),
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      if (_isListening)
                        Container(
                          width: 100,
                          height: 100,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: Colors.green.withOpacity(0.3),
                          ),
                        ).animate(onPlay: (controller) => controller.repeat())
                         .scale(begin: const Offset(1, 1), end: const Offset(1.5, 1.5), duration: 1.seconds)
                         .fade(begin: 0.5, end: 0, duration: 1.seconds),
                      Container(
                        width: 80,
                        height: 80,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          gradient: LinearGradient(
                            colors: _isListening 
                                ? [Colors.redAccent, Colors.red] 
                                : [const Color(0xFF43A047), const Color(0xFF2E7D32)],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: (_isListening ? Colors.red : Colors.green).withOpacity(0.4),
                              blurRadius: 20,
                              offset: const Offset(0, 10),
                            ),
                          ],
                        ),
                        child: const Icon(Icons.mic, color: Colors.white, size: 40),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildChatBubble({required String text, required bool isUser, bool isTranscribing = false}) {
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 8),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.75),
        decoration: BoxDecoration(
          color: isUser ? const Color(0xFF2E7D32) : Colors.white,
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(20),
            topRight: const Radius.circular(20),
            bottomLeft: Radius.circular(isUser ? 20 : 0),
            bottomRight: Radius.circular(isUser ? 0 : 20),
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.05),
              blurRadius: 5,
              offset: const Offset(0, 2),
            )
          ],
        ),
        child: Text(
          text + (isTranscribing ? '...' : ''),
          style: TextStyle(
            color: isUser ? Colors.white : Colors.black87,
            fontSize: 16,
            height: 1.4,
          ),
        ),
      ),
    );
  }
}
