import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'tflite_service.dart';
import '../../core/network/api_service.dart';

// ─── Screen ─────────────────────────────────────────────────────────────────

class DiseaseDetectScreen extends StatefulWidget {
  const DiseaseDetectScreen({Key? key}) : super(key: key);

  @override
  State<DiseaseDetectScreen> createState() => _DiseaseDetectScreenState();
}

class _DiseaseDetectScreenState extends State<DiseaseDetectScreen> {
  // ── Services ──
  final TFLiteService _tfliteService = TFLiteService();
  final ApiService _apiService = ApiService();
  final ImagePicker _picker = ImagePicker();

  // ── State ──
  File? _image;
  bool _isModelLoaded = false;
  bool _isAnalyzing = false;

  // ── Result State ──
  String? _diseaseName;
  double? _confidence;
  bool _isHealthy = false;

  // ── Colours ──
  static const _green = Color(0xFF2E7D32);
  static const _bg = Color(0xFFF1F8E9);

  // ─── Init ────────────────────────────────────────────────────────────────
  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    await _tfliteService.init();
    if (mounted) setState(() => _isModelLoaded = true);
  }

  // ─── Picking ─────────────────────────────────────────────────────────────
  Future<void> _pickImage(ImageSource source) async {
    final picked = await _picker.pickImage(
      source: source,
      imageQuality: 85,
      maxWidth: 1024,
    );
    if (picked == null) return;
    setState(() {
      _image = File(picked.path);
      _diseaseName = null;
      _confidence = null;
      _isHealthy = false;
    });
  }

  // ─── Analysis ────────────────────────────────────────────────────────────
  Future<void> _analyzeImage() async {
    if (_image == null) return;
    setState(() => _isAnalyzing = true);

    // Try backend fallback
    try {
      final bytes = await _image!.readAsBytes();
      final response = await _apiService.detectDisease(bytes);
      
      if (response != null && response['disease_name'] != null) {
        setState(() {
          _diseaseName = response['disease_name'];
          _confidence = response['confidence']?.toDouble() ?? 0.85;
          _isHealthy = _diseaseName!.toLowerCase().contains('healthy');
          _isAnalyzing = false;
        });
        return;
      }
    } catch (_) {}

    // Fallback to local TFLite model
    final result = await _tfliteService.detectDisease(_image!);
    if (result != null) {
      setState(() {
        _diseaseName = result['disease_name'];
        _confidence = result['confidence']?.toDouble() ?? 0.85;
        _isHealthy = _diseaseName!.toLowerCase().contains('healthy');
      });
    }

    setState(() => _isAnalyzing = false);
  }

  // ─── Treatment Tips ───────────────────────────────────────────────────────
  List<String> _getTreatmentTips(String diseaseName) {
    final d = diseaseName.toLowerCase();
    if (d.contains('healthy')) {
      return [
        'Your plant looks healthy! Keep up the good care.',
        'Maintain proper irrigation and nutrient balance.',
        'Monitor regularly to catch any early signs of stress.',
      ];
    }
    if (d.contains('blight')) {
      return [
        'Remove and destroy affected leaves immediately.',
        'Apply copper-based fungicide every 7–10 days.',
        'Avoid overhead watering; water at soil level.',
      ];
    }
    if (d.contains('rust')) {
      return [
        'Apply sulfur or mancozeb fungicide at first sign.',
        'Ensure good air circulation between plants.',
        'Rotate crops to break the disease cycle next season.',
      ];
    }
    if (d.contains('mosaic') || d.contains('virus')) {
      return [
        'Remove and burn infected plants to prevent spread.',
        'Control aphid populations with neem oil spray.',
        'Use virus-resistant seed varieties next crop.',
      ];
    }
    if (d.contains('spot')) {
      return [
        'Apply chlorothalonil or mancozeb fungicide weekly.',
        'Collect and destroy fallen leaves from the field.',
        'Contact your local KVK for specific dosage advice.',
      ];
    }
    // Generic fallback
    return [
      'Isolate affected plants to prevent spread.',
      'Consult a local KVK agronomist for certified treatment.',
      'Document the symptoms and contact your state agriculture department.',
    ];
  }

  // ─── Build ───────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg,
      appBar: AppBar(
        backgroundColor: Colors.white,
        foregroundColor: _green,
        elevation: 0,
        surfaceTintColor: Colors.white,
        title: Text(
          '🌿 Crop Doctor',
          style: GoogleFonts.poppins(
            fontSize: 18,
            fontWeight: FontWeight.w700,
            color: _green,
          ),
        ),
        centerTitle: false,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_rounded),
          onPressed: () => Navigator.of(context).maybePop(),
        ),
      ),
      body: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        child: Column(
          children: [
            _buildImageCard(),
            _buildActionButtons(),
            _buildInstructionsCard(),
            if (_image != null && _diseaseName == null && !_isAnalyzing)
              _buildAnalyzeButton(),
            if (_diseaseName != null && !_isAnalyzing)
              _buildResultCard()
                  .animate()
                  .slideY(begin: 0.4, duration: 500.ms, curve: Curves.easeOut)
                  .fadeIn(duration: 400.ms),
            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }

  // ─── Image Preview Card ───────────────────────────────────────────────────
  Widget _buildImageCard() {
    return GestureDetector(
      onTap: _image == null
          ? () => _pickImage(ImageSource.camera)
          : null,
      child: Container(
        height: 280,
        margin: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: _image == null
                ? const Color(0xFF60AD5E).withOpacity(0.45)
                : Colors.transparent,
            width: 2,
            // Dashed border via custom paint on empty state
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.06),
              blurRadius: 14,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        clipBehavior: Clip.antiAlias,
        child: _image == null ? _buildEmptyImageState() : _buildImageState(),
      ),
    ).animate().fadeIn(delay: 100.ms).scale(begin: const Offset(0.96, 0.96));
  }

  Widget _buildEmptyImageState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.add_photo_alternate_outlined,
            size: 80,
            color: const Color(0xFF60AD5E),
          )
              .animate(onPlay: (c) => c.repeat(reverse: true))
              .scaleXY(
                begin: 1.0,
                end: 1.08,
                duration: 1800.ms,
                curve: Curves.easeInOut,
              ),
          const SizedBox(height: 16),
          Text(
            'Tap to scan a leaf',
            style: GoogleFonts.poppins(
              fontSize: 16,
              fontWeight: FontWeight.w600,
              color: const Color(0xFF546E7A),
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'Works offline • No internet needed',
            style: GoogleFonts.poppins(
              fontSize: 12,
              color: Colors.grey[500],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImageState() {
    return Stack(
      children: [
        Positioned.fill(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: Image.file(_image!, fit: BoxFit.cover),
          ),
        ),
        if (_isAnalyzing)
          Container(
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.55),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(
                    color: Colors.white,
                    strokeWidth: 3,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Analyzing...',
                    style: GoogleFonts.poppins(
                      color: Colors.white,
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                    ),
                  )
                      .animate(onPlay: (c) => c.repeat())
                      .shimmer(
                        duration: 1200.ms,
                        color: Colors.white.withOpacity(0.6),
                      ),
                ],
              ),
            ),
          ),
      ],
    );
  }

  // ─── Camera / Gallery Buttons ─────────────────────────────────────────────
  Widget _buildActionButtons() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
      child: Row(
        children: [
          Expanded(
            child: _ActionBtn(
              icon: Icons.camera_alt_rounded,
              label: 'Camera',
              onPressed: _isModelLoaded
                  ? () => _pickImage(ImageSource.camera)
                  : null,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: _ActionBtn(
              icon: Icons.photo_library_rounded,
              label: 'Gallery',
              onPressed: _isModelLoaded
                  ? () => _pickImage(ImageSource.gallery)
                  : null,
            ),
          ),
        ],
      ),
    );
  }

  // ─── Instructions Card ────────────────────────────────────────────────────
  Widget _buildInstructionsCard() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 0, 16, 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: const Border(
          left: BorderSide(color: Color(0xFF2E7D32), width: 3),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '📋 How to get best results',
            style: GoogleFonts.poppins(
              fontSize: 14,
              fontWeight: FontWeight.w700,
              color: const Color(0xFF1B2B1D),
            ),
          ),
          const SizedBox(height: 10),
          ...[
            'Use natural light',
            'Focus on one leaf',
            'Show visible symptoms',
          ].map(
            (tip) => Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Row(
                children: [
                  const Icon(Icons.check_circle_rounded,
                      color: Color(0xFF43A047), size: 18),
                  const SizedBox(width: 8),
                  Text(
                    tip,
                    style: GoogleFonts.poppins(
                      fontSize: 13,
                      color: Colors.grey[700],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    ).animate().fadeIn(delay: 200.ms);
  }

  // ─── Analyze Button ───────────────────────────────────────────────────────
  Widget _buildAnalyzeButton() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
      child: SizedBox(
        height: 52,
        width: double.infinity,
        child: ElevatedButton(
          onPressed: _analyzeImage,
          style: ElevatedButton.styleFrom(
            backgroundColor: _green,
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14)),
            elevation: 2,
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text('🔬', style: TextStyle(fontSize: 18)),
              const SizedBox(width: 8),
              Text(
                'Analyze Disease',
                style: GoogleFonts.poppins(
                  fontSize: 15,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
        ),
      )
          .animate()
          .fadeIn(duration: 350.ms)
          .slideY(begin: 0.2),
    );
  }

  // ─── Result Card ──────────────────────────────────────────────────────────
  Widget _buildResultCard() {
    final conf = _confidence ?? 0.85;
    final disease = _diseaseName ?? 'Unknown';
    final tips = _getTreatmentTips(disease);
    final confColor = conf >= 0.80
        ? const Color(0xFF43A047)
        : conf >= 0.60
            ? Colors.orange
            : Colors.red;

    return Container(
      margin: const EdgeInsets.fromLTRB(16, 0, 16, 0),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.08),
            blurRadius: 20,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Header ──
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
            child: Row(
              children: [
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: (_isHealthy
                            ? const Color(0xFF43A047)
                            : Colors.red.shade400)
                        .withOpacity(0.12),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    _isHealthy
                        ? Icons.check_circle_rounded
                        : Icons.coronavirus_rounded,
                    color: _isHealthy
                        ? const Color(0xFF43A047)
                        : Colors.red.shade400,
                    size: 28,
                  ),
                )
                    .animate()
                    .scale(
                      begin: const Offset(0.5, 0.5),
                      duration: 400.ms,
                      curve: Curves.easeOutBack,
                    ),
                const SizedBox(width: 14),
                Expanded(
                  child: Text(
                    disease,
                    style: GoogleFonts.poppins(
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                      color: const Color(0xFF1B2B1D),
                      height: 1.2,
                    ),
                  ),
                ),
              ],
            ),
          ),

          // ── Confidence bar ──
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Confidence',
                      style: GoogleFonts.poppins(
                        fontSize: 12,
                        color: Colors.grey[600],
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    Text(
                      '${(conf * 100).toInt()}% match',
                      style: GoogleFonts.poppins(
                        fontSize: 12,
                        color: confColor,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: LinearProgressIndicator(
                    value: conf,
                    minHeight: 8,
                    backgroundColor: Colors.grey[200],
                    valueColor:
                        AlwaysStoppedAnimation<Color>(confColor),
                  ),
                ),
              ],
            ),
          ),

          const Divider(height: 28, indent: 20, endIndent: 20),

          // ── Treatment ──
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 0),
            child: Text(
              'What to do:',
              style: GoogleFonts.poppins(
                fontSize: 14,
                fontWeight: FontWeight.w700,
                color: const Color(0xFF1B2B1D),
              ),
            ),
          ),
          const SizedBox(height: 8),
          ...tips.asMap().entries.map(
            (entry) => Padding(
              padding: const EdgeInsets.fromLTRB(12, 0, 20, 6),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.arrow_right_rounded,
                      color: _green, size: 22),
                  const SizedBox(width: 4),
                  Expanded(
                    child: Text(
                      entry.value,
                      style: GoogleFonts.poppins(
                        fontSize: 13,
                        color: Colors.grey[800],
                        height: 1.4,
                      ),
                    ),
                  ),
                ],
              ),
            )
                .animate(delay: (entry.key * 80).ms)
                .fadeIn()
                .slideX(begin: -0.1),
          ),

          // ── Actions ──
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
            child: Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: () {
                      Navigator.of(context).pop();
                      // Navigate to chat screen with disease query
                    },
                    style: OutlinedButton.styleFrom(
                      foregroundColor: _green,
                      side: const BorderSide(color: _green),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10)),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                    child: Text(
                      'Ask AI →',
                      style: GoogleFonts.poppins(
                          fontWeight: FontWeight.w600, fontSize: 13),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton(
                    onPressed: () {
                      // Could open a details/wiki page
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _green,
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10)),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      elevation: 0,
                    ),
                    child: Text(
                      'Learn More',
                      style: GoogleFonts.poppins(
                          fontWeight: FontWeight.w600, fontSize: 13),
                    ),
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

// ─── Action Button Widget ─────────────────────────────────────────────────────

class _ActionBtn extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback? onPressed;

  const _ActionBtn({
    required this.icon,
    required this.label,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 52,
      child: OutlinedButton.icon(
        onPressed: onPressed,
        icon: Icon(icon, size: 20),
        label: Text(
          label,
          style: GoogleFonts.poppins(
            fontWeight: FontWeight.w600,
            fontSize: 14,
          ),
        ),
        style: OutlinedButton.styleFrom(
          foregroundColor: const Color(0xFF2E7D32),
          side: const BorderSide(color: Color(0xFF2E7D32), width: 1.5),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          backgroundColor: Colors.white,
        ),
      ),
    );
  }
}
