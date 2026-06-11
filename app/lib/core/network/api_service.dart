import 'dart:typed_data';
import 'package:dio/dio.dart';

/// Centralised HTTP client for all FarmGenius backend calls.
///
/// Design decisions:
///   - A single [Dio] instance is reused across the app (no repeated init cost).
///   - The ngrok skip-browser-warning header is added globally so every request
///     bypasses ngrok's interstitial HTML page during development/staging.
///   - Retry logic is handled by a custom [InterceptorsWrapper] that retries up
///     to 3 times with a 3-second delay on connection/receive timeouts and
///     unknown network errors — essential for India's rural 2G networks.
///   - All public methods return `null` on failure rather than propagating
///     exceptions, so the UI can render graceful offline states.
class ApiService {
  // ─── Base URL — can be overridden for testing or future production URL ─────
  static const String _defaultBaseUrl =
      'https://pagodalike-dannielle-stageably.ngrok-free.dev';

  late final Dio _dio;

  /// Creates an [ApiService]. Optionally provide a [baseUrl] to override the
  /// default ngrok development URL (e.g. in unit tests).
  ApiService({String? baseUrl}) {
    _dio = Dio(
      BaseOptions(
        baseUrl: baseUrl ?? _defaultBaseUrl,
        // 30 seconds — enough headroom for 2G connections without indefinite hangs.
        connectTimeout: const Duration(seconds: 30),
        receiveTimeout: const Duration(seconds: 30),
        sendTimeout: const Duration(seconds: 30),
        headers: {
          // Bypass ngrok's browser interstitial page in development/staging.
          'ngrok-skip-browser-warning': 'true',
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
      ),
    );

    // ── Retry interceptor ─────────────────────────────────────────────────
    // Retries up to 3 times with a 3-second back-off for transient errors.
    _dio.interceptors.add(
      InterceptorsWrapper(
        onError: (DioException err, ErrorInterceptorHandler handler) async {
          final int retryCount = err.requestOptions.extra['retryCount'] ?? 0;

          if (_shouldRetry(err) && retryCount < 3) {
            err.requestOptions.extra['retryCount'] = retryCount + 1;
            try {
              await Future.delayed(const Duration(seconds: 3));
              final response = await _dio.fetch(err.requestOptions);
              return handler.resolve(response);
            } catch (_) {
              // Fall through to the outer handler if retry also fails.
            }
          }
          return handler.next(err);
        },
      ),
    );
  }

  // ─── Retry eligibility ────────────────────────────────────────────────────
  bool _shouldRetry(DioException err) {
    return err.type == DioExceptionType.connectionTimeout ||
        err.type == DioExceptionType.receiveTimeout ||
        err.type == DioExceptionType.sendTimeout ||
        err.type == DioExceptionType.unknown;
  }

  // ─── Chat ─────────────────────────────────────────────────────────────────

  /// Sends a farmer query to the backend and returns the AI response.
  ///
  /// Returns `null` on network failure so callers can fall back to local
  /// responses or cached answers.
  ///
  /// Expected response shape:
  /// ```json
  /// { "response": "...", "source": "groq|local", "confidence": 0.91 }
  /// ```
  Future<Map<String, dynamic>?> sendChatQuery(
    String query,
    String language,
    String farmerId,
  ) async {
    try {
      final response = await _dio.post(
        '/chat',
        data: {
          'query': query,
          'language': language,
          'farmer_id': farmerId,
        },
      );
      return response.data as Map<String, dynamic>?;
    } on DioException {
      // Network / timeout errors already retried by the interceptor above.
      return null;
    } catch (_) {
      return null;
    }
  }

  // ─── Mandi prices ─────────────────────────────────────────────────────────

  /// Fetches today's mandi prices for a commodity in a given district/state.
  ///
  /// Returns `null` on failure; the UI should display cached Hive data instead.
  ///
  /// Expected response shape:
  /// ```json
  /// {
  ///   "commodity": "wheat",
  ///   "district": "dharwad",
  ///   "min_price": 2100,
  ///   "max_price": 2350,
  ///   "modal_price": 2250,
  ///   "date": "2025-06-01",
  ///   "unit": "quintal"
  /// }
  /// ```
  Future<Map<String, dynamic>?> fetchPrices(
    String commodity,
    String district,
    String state,
  ) async {
    try {
      final response = await _dio.get(
        '/prices',
        queryParameters: {
          'commodity': commodity,
          'district': district,
          'state': state,
        },
      );
      return response.data as Map<String, dynamic>?;
    } on DioException {
      return null;
    } catch (_) {
      return null;
    }
  }

  // ─── Weather ──────────────────────────────────────────────────────────────

  /// Returns a 7-day farming-optimised weather forecast for the given location.
  ///
  /// Each element in the list is a day's forecast:
  /// ```json
  /// {
  ///   "date": "2025-06-01",
  ///   "max_temp": 34,
  ///   "min_temp": 24,
  ///   "rainfall_mm": 5.2,
  ///   "wind_kmh": 12,
  ///   "farming_advisory": "Light rain — safe to spray fertiliser."
  /// }
  /// ```
  Future<List<dynamic>?> fetchWeather(
    String district,
    String state,
  ) async {
    try {
      final response = await _dio.get(
        '/weather',
        queryParameters: {
          'district': district,
          'state': state,
        },
      );
      return response.data as List<dynamic>?;
    } on DioException {
      return null;
    } catch (_) {
      return null;
    }
  }

  // ─── Disease detection ────────────────────────────────────────────────────

  /// Sends a crop image to the backend for server-side disease detection.
  ///
  /// This is the *online fallback*; primary detection runs on-device via TFLite.
  /// [imageBytes] should be the raw bytes of a JPEG or PNG image (≤ 5 MB).
  ///
  /// Expected response shape:
  /// ```json
  /// {
  ///   "disease_name": "Tomato___Early_blight",
  ///   "confidence": 0.94,
  ///   "treatment": "Apply Mancozeb @ 2g/L...",
  ///   "source_kvk": "ICAR-IIHR Bengaluru"
  /// }
  /// ```
  Future<Map<String, dynamic>?> detectDisease(Uint8List imageBytes) async {
    try {
      final formData = FormData.fromMap({
        'file': MultipartFile.fromBytes(
          imageBytes,
          filename: 'crop_photo.jpg',
          contentType: DioMediaType('image', 'jpeg'),
        ),
      });

      final response = await _dio.post(
        '/disease/detect',
        data: formData,
        options: Options(
          // Override Content-Type — Dio sets multipart/form-data automatically.
          headers: {
            'ngrok-skip-browser-warning': 'true',
          },
        ),
      );
      return response.data as Map<String, dynamic>?;
    } on DioException {
      return null;
    } catch (_) {
      return null;
    }
  }
}
