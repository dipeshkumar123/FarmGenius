import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';

// ─────────────────────────────────────────────
// DATA MODEL
// ─────────────────────────────────────────────
class MandiItem {
  final String emoji;
  final String name;
  final String market;
  final int price;
  final int change;
  final String category;

  const MandiItem({
    required this.emoji,
    required this.name,
    required this.market,
    required this.price,
    required this.change,
    required this.category,
  });
}

const List<MandiItem> _allItems = [
  MandiItem(emoji: '🌾', name: 'Wheat',       market: 'Hubballi APMC',  price: 2180,  change: 35,   category: 'Cereals'),
  MandiItem(emoji: '🌽', name: 'Maize',       market: 'Davangere APMC', price: 1920,  change: -15,  category: 'Cereals'),
  MandiItem(emoji: '🫘', name: 'Soybean',     market: 'Dharwad APMC',   price: 4420,  change: 0,    category: 'Pulses'),
  MandiItem(emoji: '🍅', name: 'Tomato',      market: 'Bengaluru APMC', price: 850,   change: 120,  category: 'Vegetables'),
  MandiItem(emoji: '🧅', name: 'Onion',       market: 'Nasik APMC',     price: 1240,  change: -80,  category: 'Vegetables'),
  MandiItem(emoji: '🍚', name: 'Rice (Paddy)',market: 'Mysuru APMC',    price: 1960,  change: 20,   category: 'Cereals'),
  MandiItem(emoji: '🥜', name: 'Groundnut',   market: 'Ballari APMC',   price: 5620,  change: 150,  category: 'Pulses'),
  MandiItem(emoji: '🌶️', name: 'Chilli',      market: 'Guntur APMC',    price: 8500,  change: -200, category: 'Spices'),
];

const List<String> _categories = ['All', 'Cereals', 'Pulses', 'Vegetables', 'Spices'];

// ─────────────────────────────────────────────
// SCREEN
// ─────────────────────────────────────────────
class MandiPricesScreen extends StatefulWidget {
  const MandiPricesScreen({super.key});

  @override
  State<MandiPricesScreen> createState() => _MandiPricesScreenState();
}

class _MandiPricesScreenState extends State<MandiPricesScreen> {
  String _selectedCategory = 'All';
  String _searchQuery = '';
  final TextEditingController _searchController = TextEditingController();

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  List<MandiItem> get _filteredItems {
    return _allItems.where((item) {
      final matchesCategory =
          _selectedCategory == 'All' || item.category == _selectedCategory;
      final matchesSearch = item.name
          .toLowerCase()
          .contains(_searchQuery.toLowerCase());
      return matchesCategory && matchesSearch;
    }).toList();
  }

  Future<void> _onRefresh() async {
    await Future.delayed(const Duration(milliseconds: 1200));
    if (mounted) setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF1F8E9),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        centerTitle: false,
        title: Text(
          '📊 Mandi Prices',
          style: GoogleFonts.poppins(
            color: const Color(0xFF1B2B1D),
            fontWeight: FontWeight.w700,
            fontSize: 20,
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.tune_rounded, color: Color(0xFF2E7D32)),
            onPressed: () {},
          ),
        ],
      ),
      body: Column(
        children: [
          // ── A: SEARCH BAR ────────────────────────────────
          _SearchBar(
            controller: _searchController,
            onChanged: (v) => setState(() => _searchQuery = v),
          ).animate().fadeIn(duration: 350.ms).slideY(begin: -0.15),

          // ── B: CATEGORY FILTER CHIPS ─────────────────────
          _CategoryChips(
            categories: _categories,
            selected: _selectedCategory,
            onTap: (cat) => setState(() => _selectedCategory = cat),
          ).animate().fadeIn(delay: 100.ms, duration: 350.ms),

          const SizedBox(height: 8),

          // ── C: PRICE LIST ────────────────────────────────
          Expanded(
            child: RefreshIndicator(
              onRefresh: _onRefresh,
              color: const Color(0xFF2E7D32),
              child: _filteredItems.isEmpty
                  ? _EmptyState()
                  : ListView.builder(
                      physics: const AlwaysScrollableScrollPhysics(),
                      padding: const EdgeInsets.only(bottom: 24),
                      itemCount: _filteredItems.length,
                      itemBuilder: (context, index) {
                        final item = _filteredItems[index];
                        return _PriceCard(item: item, index: index);
                      },
                    ),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────
// SEARCH BAR WIDGET
// ─────────────────────────────────────────────
class _SearchBar extends StatelessWidget {
  final TextEditingController controller;
  final ValueChanged<String> onChanged;

  const _SearchBar({required this.controller, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 16, 16, 12),
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(30),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF2E7D32).withOpacity(0.12),
            blurRadius: 16,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          const Icon(Icons.search, color: Colors.grey, size: 22),
          const SizedBox(width: 12),
          Expanded(
            child: TextField(
              controller: controller,
              onChanged: onChanged,
              style: GoogleFonts.inter(fontSize: 15),
              decoration: const InputDecoration(
                hintText: 'Search commodity...',
                border: InputBorder.none,
                hintStyle: TextStyle(color: Colors.grey),
                isDense: true,
                contentPadding: EdgeInsets.symmetric(vertical: 14),
              ),
            ),
          ),
          if (controller.text.isNotEmpty)
            GestureDetector(
              onTap: () {
                controller.clear();
                onChanged('');
              },
              child: const Icon(Icons.close, color: Colors.grey, size: 18),
            ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────
// CATEGORY CHIPS WIDGET
// ─────────────────────────────────────────────
class _CategoryChips extends StatelessWidget {
  final List<String> categories;
  final String selected;
  final ValueChanged<String> onTap;

  const _CategoryChips({
    required this.categories,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 44,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        itemCount: categories.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (context, index) {
          final cat = categories[index];
          final isSelected = cat == selected;
          return GestureDetector(
            onTap: () => onTap(cat),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 220),
              height: 36,
              padding: const EdgeInsets.symmetric(horizontal: 18),
              decoration: BoxDecoration(
                color: isSelected ? const Color(0xFF2E7D32) : Colors.white,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: const Color(0xFF2E7D32),
                  width: isSelected ? 0 : 1.5,
                ),
                boxShadow: isSelected
                    ? [
                        BoxShadow(
                          color: const Color(0xFF2E7D32).withOpacity(0.30),
                          blurRadius: 8,
                          offset: const Offset(0, 3),
                        )
                      ]
                    : [],
              ),
              child: Center(
                child: Text(
                  cat,
                  style: GoogleFonts.inter(
                    color: isSelected ? Colors.white : const Color(0xFF2E7D32),
                    fontWeight: FontWeight.w600,
                    fontSize: 13,
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

// ─────────────────────────────────────────────
// PRICE CARD WIDGET
// ─────────────────────────────────────────────
class _PriceCard extends StatelessWidget {
  final MandiItem item;
  final int index;

  const _PriceCard({required this.item, required this.index});

  Color get _changeColor {
    if (item.change > 0) return const Color(0xFF388E3C);
    if (item.change < 0) return const Color(0xFFC62828);
    return const Color(0xFF546E7A);
  }

  IconData get _changeIcon {
    if (item.change > 0) return Icons.trending_up;
    if (item.change < 0) return Icons.trending_down;
    return Icons.trending_flat;
  }

  String get _changeText {
    if (item.change > 0) return '+₹${item.change}';
    if (item.change < 0) return '-₹${item.change.abs()}';
    return '₹0';
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF1B5E20).withOpacity(0.08),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              // Emoji circle
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: const Color(0xFFE8F5E9),
                  borderRadius: BorderRadius.circular(22),
                ),
                child: Center(
                  child: Text(
                    item.emoji,
                    style: const TextStyle(fontSize: 22),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              // Name + market
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      item.name,
                      style: GoogleFonts.poppins(
                        fontWeight: FontWeight.w700,
                        fontSize: 15,
                        color: const Color(0xFF1B2B1D),
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      item.market,
                      style: GoogleFonts.inter(
                        fontSize: 12,
                        color: Colors.grey[600],
                      ),
                    ),
                  ],
                ),
              ),
              // Price + change
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    '₹${_formatPrice(item.price)}/q',
                    style: GoogleFonts.poppins(
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                      color: const Color(0xFF1B2B1D),
                    ),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(_changeIcon, color: _changeColor, size: 15),
                      const SizedBox(width: 2),
                      Text(
                        _changeText,
                        style: GoogleFonts.inter(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: _changeColor,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            'Updated 2 hrs ago',
            style: GoogleFonts.inter(
              fontSize: 11,
              color: Colors.grey[500],
            ),
          ),
        ],
      ),
    )
        .animate()
        .fadeIn(delay: (index * 80).ms, duration: 350.ms)
        .slideX(begin: 0.3, curve: Curves.easeOut);
  }

  String _formatPrice(int price) {
    if (price >= 1000) {
      final s = price.toString();
      return '${s.substring(0, s.length - 3)},${s.substring(s.length - 3)}';
    }
    return price.toString();
  }
}

// ─────────────────────────────────────────────
// EMPTY STATE
// ─────────────────────────────────────────────
class _EmptyState extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text('🔍', style: TextStyle(fontSize: 48)),
          const SizedBox(height: 12),
          Text(
            'No commodities found',
            style: GoogleFonts.poppins(
              fontSize: 16,
              color: Colors.grey[600],
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Try a different search or category',
            style: GoogleFonts.inter(fontSize: 13, color: Colors.grey[500]),
          ),
        ],
      ),
    );
  }
}
