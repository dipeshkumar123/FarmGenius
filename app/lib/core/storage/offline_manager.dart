import 'package:hive_flutter/hive_flutter.dart';

class OfflineManager {
  static const String queriesBox = 'queries';
  static const String pricesBox = 'prices';
  static const String weatherBox = 'weather';

  static Future<void> init() async {
    await Hive.openBox(queriesBox);
    await Hive.openBox(pricesBox);
    await Hive.openBox(weatherBox);
  }

  // Prices Cache
  static Future<void> savePrices(Map<String, dynamic> prices) async {
    final box = Hive.box(pricesBox);
    await box.put('latest_prices', prices);
  }

  static Map<dynamic, dynamic>? getPrices() {
    return Hive.box(pricesBox).get('latest_prices');
  }

  // Weather Cache
  static Future<void> saveWeather(List<dynamic> forecast) async {
    final box = Hive.box(weatherBox);
    await box.put('latest_weather', forecast);
  }

  static List<dynamic>? getWeather() {
    return Hive.box(weatherBox).get('latest_weather');
  }
}
