import 'package:flutter/material.dart';
import '../../core/storage/offline_manager.dart';
import '../../core/network/api_service.dart';

class MandiPricesScreen extends StatefulWidget {
  const MandiPricesScreen({Key? key}) : super(key: key);

  @override
  _MandiPricesScreenState createState() => _MandiPricesScreenState();
}

class _MandiPricesScreenState extends State<MandiPricesScreen> {
  Map<String, dynamic>? _prices;
  final ApiService _apiService = ApiService();
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadPrices();
  }

  Future<void> _loadPrices() async {
    // 1. Try loading from offline cache first
    final cached = OfflineManager.getPrices();
    if (cached != null) {
      setState(() {
        _prices = Map<String, dynamic>.from(cached);
        _isLoading = false;
      });
    }

    // 2. Fetch fresh data from network in background
    try {
      final freshData = await _apiService.fetchPrices('wheat', 'dharwad', 'karnataka');
      await OfflineManager.savePrices(freshData);
      setState(() {
        _prices = freshData;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
      // Fails silently, uses cached data
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Mandi Prices')),
      body: Center(
        child: _isLoading && _prices == null
            ? const CircularProgressIndicator()
            : _prices == null
                ? const Text('No offline data available.', style: TextStyle(fontSize: 20))
                : Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.store, size: 100, color: Colors.orange),
                      const SizedBox(height: 20),
                      Text('${_prices!['commodity']} in ${_prices!['district']}', style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 20),
                      Text('₹${_prices!['modal_price']} / Quintal', style: const TextStyle(fontSize: 40, fontWeight: FontWeight.bold, color: Colors.green)),
                      const SizedBox(height: 10),
                      Text('Date: ${_prices!['date']}', style: const TextStyle(fontSize: 18, color: Colors.grey)),
                      const SizedBox(height: 40),
                      const Text('Showing latest available offline data.', style: TextStyle(fontSize: 16, color: Colors.grey)),
                    ],
                  ),
      ),
    );
  }
}
