import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../core/network/api_service.dart';

class SchemesScreen extends StatefulWidget {
  const SchemesScreen({super.key});

  @override
  State<SchemesScreen> createState() => _SchemesScreenState();
}

class _SchemesScreenState extends State<SchemesScreen> {
  final ApiService _apiService = ApiService();
  bool _isLoading = false;
  List<dynamic>? _schemes;
  
  final _cropController = TextEditingController(text: 'Wheat');
  final _stateController = TextEditingController(text: 'Maharashtra');

  @override
  void initState() {
    super.initState();
    _fetchSchemes();
  }

  Future<void> _fetchSchemes() async {
    setState(() {
      _isLoading = true;
      _schemes = null;
    });

    final response = await _apiService.fetchSchemes(
      _cropController.text,
      _stateController.text,
    );

    if (mounted) {
      setState(() {
        _isLoading = false;
        _schemes = response ?? [];
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
          'Govt Schemes',
          style: GoogleFonts.poppins(fontWeight: FontWeight.w600, fontSize: 18),
        ),
        elevation: 0,
      ),
      body: Column(
        children: [
          _buildFilterBar(),
          Expanded(child: _buildResults()),
        ],
      ),
    );
  }

  Widget _buildFilterBar() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 10, offset: const Offset(0, 4))],
      ),
      child: Row(
        children: [
          Expanded(
            child: _buildTextField('Crop', _cropController, Icons.grass),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: _buildTextField('State', _stateController, Icons.map),
          ),
          const SizedBox(width: 12),
          Container(
            height: 48,
            width: 48,
            decoration: BoxDecoration(
              color: const Color(0xFF2E7D32),
              borderRadius: BorderRadius.circular(12),
            ),
            child: IconButton(
              icon: const Icon(Icons.search, color: Colors.white),
              onPressed: _isLoading ? null : _fetchSchemes,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTextField(String hint, TextEditingController controller, IconData icon) {
    return Container(
      height: 48,
      decoration: BoxDecoration(
        color: const Color(0xFFF1F8E9),
        borderRadius: BorderRadius.circular(12),
      ),
      child: TextField(
        controller: controller,
        style: GoogleFonts.poppins(fontSize: 14),
        decoration: InputDecoration(
          border: InputBorder.none,
          prefixIcon: Icon(icon, color: const Color(0xFF2E7D32), size: 18),
          hintText: hint,
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        ),
      ),
    );
  }

  Widget _buildResults() {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator(color: Color(0xFF2E7D32)));
    }

    if (_schemes == null || _schemes!.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.description_outlined, size: 64, color: Color(0xFF90A4AE)),
            const SizedBox(height: 16),
            Text('No schemes found.', style: GoogleFonts.poppins(fontSize: 16, color: const Color(0xFF546E7A))),
          ],
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: _schemes!.length,
      itemBuilder: (context, index) {
        final scheme = _schemes![index];
        return Container(
          margin: const EdgeInsets.only(bottom: 16),
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(20),
            boxShadow: [BoxShadow(color: const Color(0xFF2E7D32).withOpacity(0.08), blurRadius: 12, offset: const Offset(0, 6))],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: const Color(0xFF2E7D32).withOpacity(0.1),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(Icons.account_balance, color: Color(0xFF2E7D32)),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          scheme['scheme_name'] ?? 'Govt Scheme',
                          style: GoogleFonts.poppins(fontSize: 16, fontWeight: FontWeight.bold, color: const Color(0xFF1B2B1D)),
                        ),
                        Text(
                          scheme['ministry'] ?? 'Ministry of Agriculture',
                          style: GoogleFonts.notoSans(fontSize: 12, color: const Color(0xFF546E7A)),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Text(
                scheme['description'] ?? '',
                style: GoogleFonts.notoSans(fontSize: 13, color: const Color(0xFF1B2B1D), height: 1.5),
              ),
              const SizedBox(height: 12),
              _buildBulletList('Eligibility', scheme['eligibility']),
              _buildBulletList('Benefits', scheme['benefits']),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: TextButton(
                  onPressed: () {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Opening portal in browser...')),
                    );
                  },
                  style: TextButton.styleFrom(
                    backgroundColor: const Color(0xFFE8F5E9),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  child: Text('Apply Now / View Details', style: GoogleFonts.poppins(color: const Color(0xFF2E7D32), fontWeight: FontWeight.w600)),
                ),
              ),
            ],
          ),
        ).animate().fadeIn(delay: Duration(milliseconds: index * 100)).slideY(begin: 0.1);
      },
    );
  }

  Widget _buildBulletList(String title, String? text) {
    if (text == null || text.isEmpty) return const SizedBox.shrink();
    return Padding(
      padding: const EdgeInsets.only(top: 8.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: GoogleFonts.poppins(fontSize: 13, fontWeight: FontWeight.w600, color: const Color(0xFF2E7D32))),
          const SizedBox(height: 4),
          Text(text, style: GoogleFonts.notoSans(fontSize: 12, color: const Color(0xFF546E7A))),
        ],
      ),
    );
  }
}
