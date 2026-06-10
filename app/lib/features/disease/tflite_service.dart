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
