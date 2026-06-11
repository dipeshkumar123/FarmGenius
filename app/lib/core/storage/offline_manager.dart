import 'package:hive_flutter/hive_flutter.dart';

/// Offline cache manager for FarmGenius.
///
/// Wraps Hive to provide:
///   1. Initialisation (call [init] once in `main()` before `runApp`).
///   2. A general-purpose key-value cache via the `farmgenius_cache` box.
///   3. Typed boxes for queries, prices, and weather that previously existed.
///   4. Static helpers [saveData], [getData], and [clearAll] for the generic box.
///
/// Existing code that references [queriesBox], [pricesBox], and [weatherBox]
/// continues to work unchanged — those boxes are opened in [init].
///
/// Usage in main():
/// ```dart
/// Future<void> main() async {
///   WidgetsFlutterBinding.ensureInitialized();
///   await OfflineManager.init();
///   runApp(const FarmGeniusApp());
/// }
/// ```
class OfflineManager {
  // ─── Private constructor — all members are static ─────────────────────────
  const OfflineManager._();

  // ─── Named Hive box keys ──────────────────────────────────────────────────
  /// Box that stores recent chat queries and responses (up to 50 entries).
  static const String queriesBox = 'queries';

  /// Box that stores the last-fetched mandi price data for the farmer's crops.
  static const String pricesBox = 'prices';

  /// Box that stores the last 7-day weather forecast for the farmer's district.
  static const String weatherBox = 'weather';

  /// General-purpose cache box used by [saveData] / [getData].
  static const String _cacheBox = 'farmgenius_cache';

  // ─── Initialisation ───────────────────────────────────────────────────────

  /// Initialises Hive and opens all required boxes.
  ///
  /// Must be called once, before [runApp], after
  /// [WidgetsFlutterBinding.ensureInitialized].
  static Future<void> init() async {
    // Initialises Hive in the app's documents directory on the device.
    await Hive.initFlutter();

    // Open all boxes concurrently to reduce startup latency.
    await Future.wait([
      Hive.openBox<dynamic>(queriesBox),
      Hive.openBox<dynamic>(pricesBox),
      Hive.openBox<dynamic>(weatherBox),
      Hive.openBox<dynamic>(_cacheBox),
    ]);
  }

  // ─── Generic key-value cache ──────────────────────────────────────────────

  /// Persists [value] under the given [key] in the general cache box.
  ///
  /// [value] must be a Hive-primitive type (String, int, double, bool, List,
  /// Map) or a class registered with a [TypeAdapter].
  static Future<void> saveData(String key, dynamic value) async {
    final box = Hive.box<dynamic>(_cacheBox);
    await box.put(key, value);
  }

  /// Retrieves the value stored under [key], or `null` if not found.
  ///
  /// Cast the result to the expected type at the call-site:
  /// ```dart
  /// final prefs = OfflineManager.getData('farmer_prefs') as Map<String, dynamic>?;
  /// ```
  static dynamic getData(String key) {
    return Hive.box<dynamic>(_cacheBox).get(key);
  }

  /// Clears **all** entries from the general cache box.
  ///
  /// Does not affect [queriesBox], [pricesBox], or [weatherBox].
  /// Call this when the farmer logs out or explicitly clears app data.
  static Future<void> clearAll() async {
    await Hive.box<dynamic>(_cacheBox).clear();
  }

  // ─── Typed helpers — Prices ───────────────────────────────────────────────

  /// Saves a prices map under the standard 'latest_prices' key.
  static Future<void> savePrices(Map<String, dynamic> prices) async {
    final box = Hive.box<dynamic>(pricesBox);
    await box.put('latest_prices', prices);
  }

  /// Returns the last cached prices, or `null` if no data has been saved yet.
  static Map<dynamic, dynamic>? getPrices() {
    return Hive.box<dynamic>(pricesBox).get('latest_prices')
        as Map<dynamic, dynamic>?;
  }

  // ─── Typed helpers — Weather ──────────────────────────────────────────────

  /// Saves the 7-day weather forecast list under 'latest_weather'.
  static Future<void> saveWeather(List<dynamic> forecast) async {
    final box = Hive.box<dynamic>(weatherBox);
    await box.put('latest_weather', forecast);
  }

  /// Returns the last cached weather forecast, or `null` if none exists.
  static List<dynamic>? getWeather() {
    return Hive.box<dynamic>(weatherBox).get('latest_weather') as List<dynamic>?;
  }

  // ─── Typed helpers — Queries ──────────────────────────────────────────────

  /// Appends a [query] map to the local query history (most recent first).
  ///
  /// Automatically prunes the list to the last 50 entries so the box does not
  /// grow unbounded on low-storage devices.
  static Future<void> saveQuery(Map<String, dynamic> query) async {
    final box = Hive.box<dynamic>(queriesBox);
    final List<dynamic> existing =
        (box.get('history') as List<dynamic>?) ?? [];

    // Prepend so index 0 is always the most recent query.
    existing.insert(0, query);

    // Keep only the 50 most recent queries to conserve storage.
    final List<dynamic> pruned =
        existing.length > 50 ? existing.sublist(0, 50) : existing;

    await box.put('history', pruned);
  }

  /// Returns the list of saved queries (newest first), or an empty list.
  static List<dynamic> getQueries() {
    return (Hive.box<dynamic>(queriesBox).get('history') as List<dynamic>?) ??
        [];
  }
}
