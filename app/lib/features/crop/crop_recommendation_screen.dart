import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../core/network/api_service.dart';

class CropRecommendationScreen extends StatefulWidget {
  const CropRecommendationScreen({super.key});

  @override
  State<CropRecommendationScreen> createState() => _CropRecommendationScreenState();
}

class _CropRecommendationScreenState extends State<CropRecommendationScreen> {
  final ApiService _apiService = ApiService();
  bool _isLoading = false;
  List<dynamic>? _recommendations;
  
  // Form State
  final _locationController = TextEditingController(text: 'Dharwad, Karnataka');
  String _soilType = 'Black';
  String _waterAvailability = 'Rainfed';
  String _farmSize = '2 Acres';
  String _season = 'Kharif';

  final _soilTypes = ['Alluvial', 'Black', 'Red', 'Laterite', 'Desert', 'Mountain'];
  final _waterTypes = ['Rainfed', 'Irrigated', 'Tube Well', 'Canal'];
  final _farmSizes = ['<1 Acre', '1-2 Acres', '3-5 Acres', '>5 Acres'];
  final _seasons = ['Kharif', 'Rabi', 'Zaid', 'Year-round'];

  Future<void> _getRecommendations() async {
    setState(() {
      _isLoading = true;
      _recommendations = null;
    });

    final response = await _apiService.recommendCrop(
      location: _locationController.text,
      soilType: _soilType,
      waterAvailability: _waterAvailability,
      farmSize: _farmSize,
      season: _season,
    );

    if (mounted) {
      setState(() {
        _isLoading = false;
        if (response != null && response['recommendations'] != null) {
          _recommendations = response['recommendations'];
        } else {
          _recommendations = []; // Empty signifies error or no results
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF1F8E9),
      appBar: AppBar(
        backgroundColor: Colors.white,
        foregroundColor: const Color(0xFF2E7D32),
        title: Text(
          'Crop Recommendation',
          style: GoogleFonts.poppins(fontWeight: FontWeight.w600, fontSize: 18),
        ),
        elevation: 0,
      ),
      body: _recommendations != null
          ? _buildResults()
          : _buildForm(),
    );
  }

  Widget _buildForm() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      physics: const BouncingScrollPhysics(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Tell us about your farm',
            style: GoogleFonts.poppins(
              fontSize: 22,
              fontWeight: FontWeight.w700,
              color: const Color(0xFF1B2B1D),
            ),
          ).animate().fadeIn(duration: 400.ms).slideY(begin: 0.1),
          const SizedBox(height: 8),
          Text(
            'Our AI will analyze your local conditions and suggest the most profitable crops to grow.',
            style: GoogleFonts.notoSans(
              fontSize: 14,
              color: const Color(0xFF546E7A),
            ),
          ).animate().fadeIn(delay: 100.ms),
          const SizedBox(height: 24),

          _buildInputCard(
            label: 'Location',
            child: TextField(
              controller: _locationController,
              decoration: const InputDecoration(
                border: InputBorder.none,
                icon: Icon(Icons.location_on_outlined, color: Color(0xFF2E7D32)),
                hintText: 'e.g. Pune, Maharashtra',
              ),
            ),
          ).animate().fadeIn(delay: 200.ms),
          const SizedBox(height: 16),

          _buildDropdownCard(
            label: 'Soil Type',
            icon: Icons.layers_outlined,
            value: _soilType,
            items: _soilTypes,
            onChanged: (v) => setState(() => _soilType = v!),
          ).animate().fadeIn(delay: 300.ms),
          const SizedBox(height: 16),

          _buildDropdownCard(
            label: 'Water Availability',
            icon: Icons.water_drop_outlined,
            value: _waterAvailability,
            items: _waterTypes,
            onChanged: (v) => setState(() => _waterAvailability = v!),
          ).animate().fadeIn(delay: 400.ms),
          const SizedBox(height: 16),

          Row(
            children: [
              Expanded(
                child: _buildDropdownCard(
                  label: 'Farm Size',
                  icon: Icons.landscape_outlined,
                  value: _farmSize,
                  items: _farmSizes,
                  onChanged: (v) => setState(() => _farmSize = v!),
                ).animate().fadeIn(delay: 500.ms),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: _buildDropdownCard(
                  label: 'Season',
                  icon: Icons.wb_sunny_outlined,
                  value: _season,
                  items: _seasons,
                  onChanged: (v) => setState(() => _season = v!),
                ).animate().fadeIn(delay: 600.ms),
              ),
            ],
          ),
          const SizedBox(height: 40),

          SizedBox(
            width: double.infinity,
            height: 56,
            child: ElevatedButton(
              onPressed: _isLoading ? null : _getRecommendations,
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF2E7D32),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
                elevation: 4,
              ),
              child: _isLoading
                  ? const CircularProgressIndicator(color: Colors.white)
                  : Text(
                      'Get AI Recommendations',
                      style: GoogleFonts.poppins(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: Colors.white,
                      ),
                    ),
            ),
          ).animate().fadeIn(delay: 700.ms),
        ],
      ),
    );
  }

  Widget _buildDropdownCard({
    required String label,
    required IconData icon,
    required String value,
    required List<String> items,
    required ValueChanged<String?> onChanged,
  }) {
    return _buildInputCard(
      label: label,
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: value,
          isExpanded: true,
          icon: const Icon(Icons.keyboard_arrow_down_rounded, color: Color(0xFF2E7D32)),
          items: items.map((e) => DropdownMenuItem(value: e, child: Row(
            children: [
              Icon(icon, color: const Color(0xFF2E7D32), size: 20),
              const SizedBox(width: 12),
              Text(e, style: GoogleFonts.poppins(fontSize: 14)),
            ],
          ))).toList(),
          onChanged: onChanged,
        ),
      ),
    );
  }

  Widget _buildInputCard({required String label, required Widget child}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: GoogleFonts.poppins(fontSize: 13, fontWeight: FontWeight.w500, color: const Color(0xFF546E7A)),
        ),
        const SizedBox(height: 6),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 10, offset: const Offset(0, 4))],
          ),
          child: child,
        ),
      ],
    );
  }

  Widget _buildResults() {
    if (_recommendations!.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline_rounded, size: 64, color: Color(0xFFEF6C00)),
            const SizedBox(height: 16),
            Text('Failed to load recommendations.', style: GoogleFonts.poppins(fontSize: 16)),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () => setState(() => _recommendations = null),
              child: const Text('Try Again'),
            )
          ],
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: _recommendations!.length + 1,
      itemBuilder: (context, index) {
        if (index == 0) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back, color: Color(0xFF2E7D32)),
                    onPressed: () => setState(() => _recommendations = null),
                  ),
                  Text(
                    'AI Recommendations',
                    style: GoogleFonts.poppins(fontSize: 20, fontWeight: FontWeight.w700),
                  ),
                ],
              ),
              const SizedBox(height: 16),
            ],
          );
        }

        final rec = _recommendations![index - 1];
        return Container(
          margin: const EdgeInsets.only(bottom: 16),
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(20),
            boxShadow: [BoxShadow(color: const Color(0xFF2E7D32).withOpacity(0.1), blurRadius: 12, offset: const Offset(0, 6))],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Text(rec['emoji'] ?? '🌱', style: const TextStyle(fontSize: 32)),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(rec['name'] ?? '', style: GoogleFonts.poppins(fontSize: 18, fontWeight: FontWeight.bold, color: const Color(0xFF1B2B1D))),
                        Text('Rank #${rec['rank']}', style: GoogleFonts.notoSans(color: const Color(0xFFF9A825), fontWeight: FontWeight.w600)),
                      ],
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                    decoration: BoxDecoration(
                      color: const Color(0xFF2E7D32).withOpacity(0.1),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text('${((rec['suitability'] ?? 0) * 100).toInt()}% Match', style: GoogleFonts.poppins(color: const Color(0xFF2E7D32), fontWeight: FontWeight.w600)),
                  ),
                ],
              ),
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 12),
                child: Divider(color: Color(0xFFE8F5E9), thickness: 1.5),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  _buildStat('Yield', rec['expectedYield'] ?? '', Icons.trending_up),
                  _buildStat('Price', rec['marketPrice'] ?? '', Icons.currency_rupee),
                  _buildStat('Profit', rec['profitEstimate'] ?? '', Icons.account_balance_wallet),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  _buildStat('Season', rec['season'] ?? '', Icons.wb_sunny_outlined),
                  _buildStat('Water', rec['water'] ?? '', Icons.water_drop_outlined),
                  _buildStat('Duration', rec['duration'] ?? '', Icons.timer_outlined),
                ],
              ),
            ],
          ),
        ).animate().fadeIn(delay: Duration(milliseconds: index * 100)).slideY(begin: 0.1);
      },
    );
  }

  Widget _buildStat(String label, String value, IconData icon) {
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 14, color: const Color(0xFF546E7A)),
              const SizedBox(width: 4),
              Text(label, style: GoogleFonts.notoSans(fontSize: 11, color: const Color(0xFF546E7A))),
            ],
          ),
          const SizedBox(height: 2),
          Text(value, style: GoogleFonts.poppins(fontSize: 12, fontWeight: FontWeight.w600, color: const Color(0xFF1B2B1D))),
        ],
      ),
    );
  }
}
