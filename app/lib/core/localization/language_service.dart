import 'package:flutter_riverpod/flutter_riverpod.dart';

class LanguageService extends StateNotifier<String> {
  LanguageService() : super('English');

  final Map<String, Map<String, String>> _dictionary = {
    'English': {
      'quick_actions': 'Quick Actions',
      'crop_rec': 'Crop Rec.',
      'scan_disease': 'Scan Disease',
      'ask_ai': 'Ask AI',
      'mandi_prices': 'Mandi Prices',
      'weather': 'Weather',
      'schemes': 'Schemes',
      'good_morning': 'Good Morning, Kisan Bhai! 🌾',
      'today_prices': "Today's Market Prices",
    },
    'हिंदी': {
      'quick_actions': 'त्वरित कार्रवाइयां',
      'crop_rec': 'फसल सलाह',
      'scan_disease': 'रोग स्कैन',
      'ask_ai': 'AI से पूछें',
      'mandi_prices': 'मंडी भाव',
      'weather': 'मौसम',
      'schemes': 'योजनाएं',
      'good_morning': 'सुप्रभात, किसान भाई! 🌾',
      'today_prices': "आज के मंडी भाव",
    },
    'ಕನ್ನಡ': {
      'quick_actions': 'ತ್ವರಿತ ಕ್ರಿಯೆಗಳು',
      'crop_rec': 'ಬೆಳೆ ಸಲಹೆ',
      'scan_disease': 'ರೋಗ ಸ್ಕ್ಯಾನ್',
      'ask_ai': 'AI ಕೇಳಿ',
      'mandi_prices': 'ಮಾರುಕಟ್ಟೆ ಬೆಲೆ',
      'weather': 'ಹವಾಮಾನ',
      'schemes': 'ಯೋಜನೆಗಳು',
      'good_morning': 'ಶುಭೋದಯ, ಕಿಸಾನ್ ಭಾಯ್! 🌾',
      'today_prices': "ಇಂದಿನ ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳು",
    },
    'తెలుగు': {
      'quick_actions': 'త్వరిత చర్యలు',
      'crop_rec': 'పంట సలహా',
      'scan_disease': 'వ్యాధి స్కాన్',
      'ask_ai': 'AI ని అడగండి',
      'mandi_prices': 'మండి ధరలు',
      'weather': 'వాతావరణం',
      'schemes': 'పథకాలు',
      'good_morning': 'శుభోదయం, కిసాన్ భాయ్! 🌾',
      'today_prices': "నేటి మార్కెట్ ధరలు",
    },
    'मराठी': {
      'quick_actions': 'त्वरित कृती',
      'crop_rec': 'पीक सल्ला',
      'scan_disease': 'रोग स्कॅन',
      'ask_ai': 'AI ला विचारा',
      'mandi_prices': 'बाजारभाव',
      'weather': 'हवामान',
      'schemes': 'योजना',
      'good_morning': 'शुभ प्रभात, किसान भाई! 🌾',
      'today_prices': "आजचे बाजारभाव",
    },
  };

  String get currentLanguage => state;

  String get apiLanguageCode {
    switch (state) {
      case 'हिंदी': return 'hi';
      case 'ಕನ್ನಡ': return 'kn';
      case 'తెలుగు': return 'te';
      case 'मराठी': return 'mr';
      case 'English':
      default:
        return 'en';
    }
  }

  void setLanguage(String lang) {
    if (_dictionary.containsKey(lang)) {
      state = lang;
    }
  }

  String translate(String key) {
    return _dictionary[state]?[key] ?? _dictionary['English']?[key] ?? key;
  }
}

final languageProvider = StateNotifierProvider<LanguageService, String>((ref) {
  return LanguageService();
});
