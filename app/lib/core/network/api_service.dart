import 'package:dio/dio.dart';

class ApiService {
  late final Dio _dio;
  
  ApiService() {
    _dio = Dio(BaseOptions(
      baseUrl: 'https://farmgenius-backend.onrender.com', // Replace with dynamic env config later
      connectTimeout: const Duration(seconds: 30), // Extended for 2G
      receiveTimeout: const Duration(seconds: 30),
    ));

    // Custom Interceptor for 2G retries
    _dio.interceptors.add(InterceptorsWrapper(
      onError: (DioException err, ErrorInterceptorHandler handler) async {
        if (_shouldRetry(err) && (err.requestOptions.extra['retryCount'] ?? 0) < 3) {
          err.requestOptions.extra['retryCount'] = (err.requestOptions.extra['retryCount'] ?? 0) + 1;
          try {
            // Wait before retry
            await Future.delayed(const Duration(seconds: 3));
            final response = await _dio.fetch(err.requestOptions);
            return handler.resolve(response);
          } catch (e) {
            return handler.next(err);
          }
        }
        return handler.next(err);
      },
    ));
  }

  bool _shouldRetry(DioException err) {
    return err.type == DioExceptionType.connectionTimeout || 
           err.type == DioExceptionType.receiveTimeout || 
           err.type == DioExceptionType.unknown;
  }

  Future<Map<String, dynamic>> sendChatQuery(String query, String language, String farmerId) async {
    final response = await _dio.post('/chat/', data: {
      "query": query,
      "language": language,
      "farmer_id": farmerId
    });
    return response.data;
  }

  Future<Map<String, dynamic>> fetchPrices(String commodity, String district, String state) async {
    final response = await _dio.get('/prices/', queryParameters: {
      "commodity": commodity,
      "district": district,
      "state": state
    });
    return response.data;
  }

  Future<List<dynamic>> fetchWeather(String district, String state) async {
    final response = await _dio.get('/weather/', queryParameters: {
      "district": district,
      "state": state
    });
    return response.data;
  }
}
