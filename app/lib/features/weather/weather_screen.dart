import 'package:flutter/material.dart';
import '../../core/network/api_service.dart';
import '../../core/storage/offline_manager.dart';

class WeatherScreen extends StatefulWidget {
  const WeatherScreen({Key? key}) : super(key: key);

  @override
  _WeatherScreenState createState() => _WeatherScreenState();
}

class _WeatherScreenState extends State<WeatherScreen> {
  final ApiService _apiService = ApiService();
  List<dynamic> _forecast = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadWeather();
  }

  Future<void> _loadWeather() async {
    final cached = OfflineManager.getWeather();
    if (cached != null && cached.isNotEmpty) {
      setState(() {
        _forecast = List.from(cached);
        _isLoading = false;
      });
    }

    try {
      final freshData = await _apiService.fetchWeather('dharwad', 'karnataka');
      await OfflineManager.saveWeather(freshData);
      setState(() {
        _forecast = freshData;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Weather Advisory')),
      body: _isLoading && _forecast.isEmpty
          ? const Center(child: CircularProgressIndicator())
          : _forecast.isEmpty
              ? const Center(child: Text('No offline weather data.', style: TextStyle(fontSize: 20)))
              : ListView.builder(
                  padding: const EdgeInsets.all(16.0),
                  itemCount: _forecast.length,
                  itemBuilder: (context, index) {
                    final day = _forecast[index];
                    return Card(
                      margin: const EdgeInsets.only(bottom: 16),
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text('Date: ${day['date']}', style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                                Text('${day['max_temp']}°C', style: const TextStyle(fontSize: 24, color: Colors.red)),
                              ],
                            ),
                            const SizedBox(height: 10),
                            Text('Rainfall: ${day['rainfall_mm']} mm', style: const TextStyle(fontSize: 18)),
                            const SizedBox(height: 10),
                            Text('Advisory: ${day['farming_advisory']}', style: const TextStyle(fontSize: 18, color: Colors.green, fontWeight: FontWeight.bold)),
                          ],
                        ),
                      ),
                    );
                  },
                ),
    );
  }
}
