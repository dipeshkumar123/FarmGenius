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
