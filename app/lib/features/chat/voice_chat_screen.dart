import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:uuid/uuid.dart';
import '../../core/network/api_service.dart';

// ─── Model ──────────────────────────────────────────────────────────────────

class ChatMessage {
  final String id;
  final String text;
  final bool isUser;
  final DateTime timestamp;

  ChatMessage({
    required this.id,
    required this.text,
    required this.isUser,
    required this.timestamp,
  });
}

// ─── Screen ─────────────────────────────────────────────────────────────────

class VoiceChatScreen extends StatefulWidget {
  const VoiceChatScreen({Key? key}) : super(key: key);

  @override
  State<VoiceChatScreen> createState() => _VoiceChatScreenState();
}

class _VoiceChatScreenState extends State<VoiceChatScreen>
    with TickerProviderStateMixin {
  // ── Services ──
  final stt.SpeechToText _speech = stt.SpeechToText();
  final FlutterTts _tts = FlutterTts();
  final ApiService _apiService = ApiService();
  final _uuid = const Uuid();

  // ── State ──
  bool _speechAvailable = false;
  bool _isListening = false;
  bool _isTyping = false; // bot typing indicator
  String _liveTranscript = '';
  String _selectedLanguage = 'हिंदी'; // tracks active UI language for the menu

  final List<ChatMessage> _messages = [];
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final FocusNode _inputFocus = FocusNode();

  // ── Quick reply suggestions ──
  static const _chips = [
    'Crop advice?',
    'Disease help?',
    'Mandi price?',
    'Weather today?',
  ];

  // ── Colours ──
  static const _green = Color(0xFF2E7D32);
  static const _greenLight = Color(0xFF43A047); // used in bot avatar LinearGradient
  static const _bg = Color(0xFFF1F8E9);

  // ─── Init ───────────────────────────────────────────────────────────────
  @override
  void initState() {
    super.initState();
    _initSpeech();
    _initTts();
    _addBotMessage(
      'नमस्ते! मैं FarmGenius AI हूँ।\n'
      'आज आपकी खेती में क्या मदद करूँ? 🌾\n\n'
      'Hello! I am FarmGenius AI.\n'
      'How can I help your farm today?',
    );
  }

  Future<void> _initSpeech() async {
    _speechAvailable = await _speech.initialize(
      onError: (e) => setState(() => _isListening = false),
      onStatus: (status) {
        if (status == stt.SpeechToText.doneStatus ||
            status == stt.SpeechToText.notListeningStatus) {
          if (_isListening) _handleStopListening();
        }
      },
    );
    setState(() {});
  }

  Future<void> _initTts() async {
    await _tts.setLanguage('hi-IN');
    await _tts.setSpeechRate(0.5);
    await _tts.setVolume(1.0);
  }

  @override
  void dispose() {
    _textController.dispose();
    _scrollController.dispose();
    _inputFocus.dispose();
    _tts.stop();
    super.dispose();
  }

  // ─── Scroll ──────────────────────────────────────────────────────────────
  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent + 100,
          duration: const Duration(milliseconds: 350),
          curve: Curves.easeOut,
        );
      }
    });
  }

  // ─── Messaging ───────────────────────────────────────────────────────────
  void _addBotMessage(String text) {
    setState(() {
      _messages.add(ChatMessage(
        id: _uuid.v4(),
        text: text,
        isUser: false,
        timestamp: DateTime.now(),
      ));
    });
    _scrollToBottom();
  }

  void _addUserMessage(String text) {
    setState(() {
      _messages.add(ChatMessage(
        id: _uuid.v4(),
        text: text,
        isUser: true,
        timestamp: DateTime.now(),
      ));
    });
    _scrollToBottom();
  }

  Future<void> _sendMessage(String text) async {
    final query = text.trim();
    if (query.isEmpty) return;

    _textController.clear();
    _addUserMessage(query);

    // Show typing indicator
    setState(() => _isTyping = true);
    _scrollToBottom();

    // Simulate / call API
    await Future.delayed(const Duration(milliseconds: 1500));

    try {
      final response =
          await _apiService.sendChatQuery(query, 'hi', 'farmer_001');
      final botText = (response?['response'] as String?) ??
          'मुझे जवाब नहीं मिला। कृपया फिर से पूछें।';
      setState(() => _isTyping = false);
      _addBotMessage(botText);
      _speak(botText);
    } catch (_) {
      setState(() => _isTyping = false);
      const errorMsg =
          'माफ करें, अभी सेवा उपलब्ध नहीं है। कृपया अपने KVK से संपर्क करें।\n\n'
          'Sorry, service unavailable. Please contact your local KVK.';
      _addBotMessage(errorMsg);
    }
  }

  Future<void> _speak(String text) async {
    await _tts.stop();
    await _tts.speak(text);
  }

  // ─── Speech ──────────────────────────────────────────────────────────────
  void _handleMicPress() {
    if (!_speechAvailable) return;
    HapticFeedback.mediumImpact();
    if (_isListening) {
      _handleStopListening();
    } else {
      _handleStartListening();
    }
  }

  Future<void> _handleStartListening() async {
    await _tts.stop();
    setState(() {
      _isListening = true;
      _liveTranscript = '';
    });
    await _speech.listen(
      onResult: (result) {
        setState(() {
          _liveTranscript = result.recognizedWords;
        });
        if (result.finalResult) {
          _handleStopListening();
        }
      },
      localeId: 'hi_IN',
      listenOptions: stt.SpeechListenOptions(
        listenMode: stt.ListenMode.confirmation,
      ),
    );
  }

  Future<void> _handleStopListening() async {
    await _speech.stop();
    final text = _liveTranscript.trim();
    setState(() {
      _isListening = false;
      _liveTranscript = '';
    });
    if (text.isNotEmpty) _sendMessage(text);
  }

  // ─── Build ───────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg,
      appBar: _buildAppBar(),
      body: Column(
        children: [
          Expanded(child: _buildMessageList()),
          if (_isListening) _buildLiveTranscriptBanner(),
          _buildQuickReplies(),
          _buildInputBar(),
        ],
      ),
    );
  }

  PreferredSizeWidget _buildAppBar() {
    return AppBar(
      backgroundColor: Colors.white,
      foregroundColor: _green,
      elevation: 0,
      shadowColor: Colors.black12,
      surfaceTintColor: Colors.white,
      leading: IconButton(
        icon: const Icon(Icons.arrow_back_ios_rounded),
        onPressed: () => Navigator.of(context).maybePop(),
      ),
      title: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'FarmGenius AI',
            style: GoogleFonts.poppins(
              fontSize: 17,
              fontWeight: FontWeight.w700,
              color: _green,
            ),
          ),
          Row(
            children: [
              Container(
                width: 7,
                height: 7,
                decoration: const BoxDecoration(
                  color: Color(0xFF66BB6A),
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 4),
              Text(
                '● Online',
                style: GoogleFonts.poppins(
                  fontSize: 11,
                  color: const Color(0xFF66BB6A),
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ],
      ),
      actions: [
        PopupMenuButton<String>(
          onSelected: (lang) => setState(() => _selectedLanguage = lang),
          itemBuilder: (_) => const [
            PopupMenuItem(value: 'हिंदी', child: Text('हिंदी')),
            PopupMenuItem(value: 'English', child: Text('English')),
            PopupMenuItem(value: 'ಕನ್ನಡ', child: Text('ಕನ್ನಡ')),
            PopupMenuItem(value: 'తెలుగు', child: Text('తెలుగు')),
            PopupMenuItem(value: 'मराठी', child: Text('मराठी')),
          ],
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: const Color(0xFFE8F5E9),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.translate_rounded, color: _green, size: 16),
                const SizedBox(width: 4),
                Text(
                  _selectedLanguage,
                  style: const TextStyle(
                    color: _green,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(width: 4),
      ],
    );
  }

  Widget _buildMessageList() {
    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.fromLTRB(12, 16, 12, 8),
      itemCount: _messages.length + (_isTyping ? 1 : 0),
      itemBuilder: (context, index) {
        if (_isTyping && index == _messages.length) {
          return _TypingIndicator().animate().fadeIn();
        }
        final msg = _messages[index];
        return _ChatBubble(message: msg)
            .animate()
            .fadeIn(duration: 300.ms)
            .slideY(begin: 0.3, duration: 300.ms, curve: Curves.easeOut);
      },
    );
  }

  Widget _buildLiveTranscriptBanner() {
    return Container(
      width: double.infinity,
      color: _green.withOpacity(0.06),
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
      child: Row(
        children: [
          const Icon(Icons.graphic_eq_rounded, color: _green, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              _liveTranscript.isEmpty ? 'Listening...' : _liveTranscript,
              style: GoogleFonts.poppins(
                fontSize: 13,
                color: _green,
                fontStyle: _liveTranscript.isEmpty
                    ? FontStyle.italic
                    : FontStyle.normal,
              ),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQuickReplies() {
    return SizedBox(
      height: 44,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 12),
        itemCount: _chips.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (context, index) {
          return ActionChip(
            label: Text(
              _chips[index],
              style: GoogleFonts.poppins(
                fontSize: 13,
                color: _green,
                fontWeight: FontWeight.w500,
              ),
            ),
            backgroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(20),
              side: const BorderSide(color: _green, width: 1.2),
            ),
            onPressed: () => _sendMessage(_chips[index]),
            elevation: 0,
            padding:
                const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
          );
        },
      ),
    );
  }

  Widget _buildInputBar() {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.07),
            blurRadius: 12,
            offset: const Offset(0, -3),
          ),
        ],
      ),
      child: SafeArea(
        top: false,
        child: Row(
          children: [
            // Text field
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceVariant,
                  borderRadius: BorderRadius.circular(24),
                ),
                child: TextField(
                  controller: _textController,
                  focusNode: _inputFocus,
                  textInputAction: TextInputAction.send,
                  onSubmitted: _sendMessage,
                  style: GoogleFonts.poppins(fontSize: 14),
                  decoration: InputDecoration(
                    hintText: 'Ask anything... / कुछ भी पूछें...',
                    hintStyle: GoogleFonts.poppins(
                      fontSize: 13,
                      color: Colors.grey[500],
                    ),
                    border: InputBorder.none,
                    contentPadding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 12),
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            // Mic button
            GestureDetector(
              onTap: _handleMicPress,
              child: Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: _isListening
                      ? Colors.red.shade50
                      : _green.withOpacity(0.1),
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  Icons.mic_rounded,
                  color: _isListening ? Colors.red : _green,
                  size: 22,
                ),
              )
                  .animate(
                    target: _isListening ? 1 : 0,
                    onPlay: (c) =>
                        _isListening ? c.repeat() : c.stop(),
                  )
                  .scaleXY(
                    begin: 1.0,
                    end: 1.15,
                    duration: 800.ms,
                    curve: Curves.easeInOut,
                  ),
            ),
            const SizedBox(width: 4),
            // Send button
            GestureDetector(
              onTap: () => _sendMessage(_textController.text),
              child: CircleAvatar(
                radius: 22,
                backgroundColor: _green,
                child: const Icon(
                  Icons.send_rounded,
                  color: Colors.white,
                  size: 20,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ─── Chat Bubble ─────────────────────────────────────────────────────────────

class _ChatBubble extends StatelessWidget {
  final ChatMessage message;
  const _ChatBubble({required this.message});

  static const _green = Color(0xFF2E7D32);
  static const _textDark = Color(0xFF1B2B1D);

  String _formatTime(DateTime dt) {
    final h = dt.hour.toString().padLeft(2, '0');
    final m = dt.minute.toString().padLeft(2, '0');
    return '$h:$m';
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(
        left: message.isUser ? 48 : 0,
        right: message.isUser ? 0 : 48,
        bottom: 12,
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        mainAxisAlignment:
            message.isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        children: [
          if (!message.isUser) ...[
            _BotAvatar(),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Column(
              crossAxisAlignment: message.isUser
                  ? CrossAxisAlignment.end
                  : CrossAxisAlignment.start,
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 16, vertical: 12),
                  decoration: BoxDecoration(
                    color: message.isUser ? _green : Colors.white,
                    borderRadius: BorderRadius.only(
                      topLeft: const Radius.circular(18),
                      topRight: const Radius.circular(18),
                      bottomLeft: Radius.circular(message.isUser ? 18 : 4),
                      bottomRight: Radius.circular(message.isUser ? 4 : 18),
                    ),
                    boxShadow: message.isUser
                        ? []
                        : [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.06),
                              blurRadius: 8,
                              offset: const Offset(0, 2),
                            ),
                          ],
                  ),
                  child: Text(
                    message.text,
                    style: GoogleFonts.poppins(
                      fontSize: 14,
                      height: 1.55,
                      color: message.isUser ? Colors.white : _textDark,
                      fontWeight: FontWeight.w400,
                    ),
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  _formatTime(message.timestamp),
                  style: GoogleFonts.poppins(
                    fontSize: 10,
                    color: Colors.grey[500],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _BotAvatar extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      width: 32,
      height: 32,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [_VoiceChatScreenState._green, _VoiceChatScreenState._greenLight],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        shape: BoxShape.circle,
      ),
      child: const Icon(
        Icons.smart_toy_rounded,
        color: Colors.white,
        size: 20,
      ),
    );
  }
}

// ─── Typing Indicator ────────────────────────────────────────────────────────

class _TypingIndicator extends StatefulWidget {
  @override
  State<_TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<_TypingIndicator>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat();
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          _BotAvatar(),
          const SizedBox(width: 8),
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(18),
                topRight: Radius.circular(18),
                bottomLeft: Radius.circular(4),
                bottomRight: Radius.circular(18),
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.06),
                  blurRadius: 8,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: List.generate(3, (i) {
                return AnimatedBuilder(
                  animation: _ctrl,
                  builder: (_, __) {
                    final offset =
                        ((_ctrl.value * 3 - i).clamp(0.0, 1.0));
                    final bounce = offset < 0.5
                        ? offset * 2
                        : 2 - offset * 2;
                    return Container(
                      margin: const EdgeInsets.symmetric(horizontal: 3),
                      width: 8,
                      height: 8,
                      transform: Matrix4.translationValues(
                          0, -4 * bounce, 0),
                      decoration: const BoxDecoration(
                        color: Color(0xFF2E7D32),
                        shape: BoxShape.circle,
                      ),
                    );
                  },
                );
              }),
            ),
          ),
        ],
      ),
    );
  }
}
