import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:flutter_animate/flutter_animate.dart';
import 'tflite_service.dart';

class DiseaseDetectScreen extends StatefulWidget {
  const DiseaseDetectScreen({Key? key}) : super(key: key);

  @override
  _DiseaseDetectScreenState createState() => _DiseaseDetectScreenState();
}

class _DiseaseDetectScreenState extends State<DiseaseDetectScreen> {
  File? _image;
  String _result = '';
  final TFLiteService _tfliteService = TFLiteService();
  bool _isModelLoaded = false;
  bool _isAnalyzing = false;

  @override
  void initState() {
    super.initState();
    _initModel();
  }
  
  Future<void> _initModel() async {
    await _tfliteService.initModel();
    setState(() => _isModelLoaded = true);
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source, imageQuality: 80);
    
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _isAnalyzing = true;
        _result = '';
      });
      _analyzeImage(_image!);
    }
  }

  Future<void> _analyzeImage(File image) async {
    // Simulating delay for better UI perception if inference is too fast
    await Future.delayed(const Duration(milliseconds: 1500));
    final prediction = await _tfliteService.predict(image);
    setState(() {
      _result = prediction;
      _isAnalyzing = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAF8),
      appBar: AppBar(
        title: const Text('Crop Doctor', style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.white,
        foregroundColor: const Color(0xFF2E7D32),
        elevation: 0,
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Identify Crop Disease',
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: Colors.black87,
                ),
                textAlign: TextAlign.center,
              ).animate().fade().slideY(begin: -0.2),
              const SizedBox(height: 8),
              Text(
                'Upload or take a photo of the affected leaf',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: Colors.grey[600],
                ),
                textAlign: TextAlign.center,
              ).animate().fade().slideY(begin: -0.2, delay: 100.ms),
              const SizedBox(height: 32),

              // Image Card
              GestureDetector(
                onTap: () => _pickImage(ImageSource.camera),
                child: Container(
                  height: 300,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(32),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.05),
                        blurRadius: 20,
                        offset: const Offset(0, 10),
                      ),
                    ],
                    border: Border.all(
                      color: _image == null ? Colors.green.withOpacity(0.3) : Colors.transparent,
                      width: 2,
                    ),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      if (_image == null)
                        Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Container(
                              padding: const EdgeInsets.all(24),
                              decoration: BoxDecoration(
                                color: Colors.green.withOpacity(0.1),
                                shape: BoxShape.circle,
                              ),
                              child: const Icon(
                                Icons.add_a_photo_rounded,
                                size: 64,
                                color: Color(0xFF2E7D32),
                              ),
                            ),
                            const SizedBox(height: 16),
                            const Text(
                              'Tap to open camera',
                              style: TextStyle(
                                fontSize: 16,
                                color: Colors.green,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        )
                      else
                        Positioned.fill(
                          child: Image.file(_image!, fit: BoxFit.cover),
                        ),
                      if (_isAnalyzing)
                        Container(
                          color: Colors.black54,
                          child: Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                const CircularProgressIndicator(color: Colors.white),
                                const SizedBox(height: 16),
                                const Text(
                                  'Analyzing locally...',
                                  style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
                                ).animate(onPlay: (controller) => controller.repeat()).shimmer(duration: 1.seconds),
                              ],
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ).animate().fade().scale(delay: 200.ms),
              const SizedBox(height: 32),

              // Action Buttons
              if (!_isAnalyzing && _result.isEmpty)
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: _isModelLoaded ? () => _pickImage(ImageSource.camera) : null,
                        icon: const Icon(Icons.camera_alt),
                        label: const Text('Camera'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF2E7D32),
                          foregroundColor: Colors.white,
                        ),
                      ),
                    ).animate().fade().slideX(begin: -0.2, delay: 300.ms),
                    const SizedBox(width: 16),
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: _isModelLoaded ? () => _pickImage(ImageSource.gallery) : null,
                        icon: const Icon(Icons.photo_library),
                        label: const Text('Gallery'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white,
                          foregroundColor: const Color(0xFF2E7D32),
                          side: const BorderSide(color: Color(0xFF2E7D32)),
                        ),
                      ),
                    ).animate().fade().slideX(begin: 0.2, delay: 300.ms),
                  ],
                ),

              // Result Display
              if (_result.isNotEmpty && !_isAnalyzing)
                Container(
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [Color(0xFF43A047), Color(0xFF2E7D32)],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(24),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.green.withOpacity(0.3),
                        blurRadius: 15,
                        offset: const Offset(0, 8),
                      ),
                    ],
                  ),
                  child: Column(
                    children: [
                      const Icon(Icons.check_circle_outline, color: Colors.white, size: 48)
                          .animate()
                          .scale(duration: 400.ms, curve: Curves.easeOutBack),
                      const SizedBox(height: 16),
                      const Text(
                        'Diagnosis Complete',
                        style: TextStyle(
                          color: Colors.white70,
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 1.2,
                        ),
                      ).animate().fade(delay: 200.ms),
                      const SizedBox(height: 8),
                      Text(
                        _result,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.bold,
                        ),
                        textAlign: TextAlign.center,
                      ).animate().fade(delay: 300.ms).slideY(begin: 0.2),
                    ],
                  ),
                ).animate().fade().slideY(begin: 0.2),
            ],
          ),
        ),
      ),
    );
  }
}
