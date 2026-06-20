import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteService {
  Interpreter? _interpreter;
  List<String>? _labels;

  Future<void> init() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/models/disease_model_quant.tflite');
      final labelsData = await rootBundle.loadString('assets/models/labels_indian_crops.txt');
      _labels = labelsData.split('\n').where((s) => s.trim().isNotEmpty).toList();
      print('TFLite model loaded successfully.');
    } catch (e) {
      print('Failed to load TFLite model: $e');
    }
  }

  Future<Map<String, dynamic>?> detectDisease(File imageFile) async {
    if (_interpreter == null || _labels == null) {
      print('Model not initialized.');
      return null;
    }

    try {
      // 1. Read and decode image
      final bytes = await imageFile.readAsBytes();
      img.Image? originalImage = img.decodeImage(bytes);
      if (originalImage == null) return null;

      // 2. Resize to 224x224 (assuming standard MobileNetV2 input)
      img.Image resizedImage = img.copyResize(originalImage, width: 224, height: 224);

      // 3. Convert image to 1x224x224x3 Float32 tensor
      // Note: If using quantized INT8, you would use Uint8. We assume Float32 for simplicity or INT8 depending on model.
      // We will create a float list to be safe if the model expects floats.
      var input = List.generate(
          1,
          (i) => List.generate(
              224,
              (y) => List.generate(
                  224,
                  (x) {
                    final pixel = resizedImage.getPixel(x, y);
                    // Normalize 0-255 to 0-1 if Float32 model, or keep 0-255 if INT8
                    return [
                      pixel.r.toDouble() / 255.0,
                      pixel.g.toDouble() / 255.0,
                      pixel.b.toDouble() / 255.0
                    ];
                  })));

      // 4. Prepare output tensor: 1 x N_classes
      var output = List.filled(1 * _labels!.length, 0.0).reshape([1, _labels!.length]);

      // 5. Run inference
      _interpreter!.run(input, output);

      // 6. Find max confidence
      List<double> probabilities = (output[0] as List).cast<double>();
      double maxConfidence = 0;
      int maxIndex = -1;

      for (int i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxConfidence) {
          maxConfidence = probabilities[i];
          maxIndex = i;
        }
      }

      if (maxIndex != -1) {
        String label = _labels![maxIndex];
        return {
          "disease_name": label,
          "confidence": maxConfidence,
          "source_kvk": "On-Device TFLite (Offline)"
        };
      }
    } catch (e) {
      print('Inference failed: $e');
    }
    return null;
  }
}
