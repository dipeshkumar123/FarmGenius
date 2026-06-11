import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;

  // ─── Market price data ────────────────────────────────────────────────────
  final List<_PriceData> _prices = const [
    _PriceData(crop: 'Wheat', emoji: '🌾', price: 2180, change: 35),
    _PriceData(crop: 'Maize', emoji: '🌽', price: 1920, change: -15),
    _PriceData(crop: 'Soybean', emoji: '🫘', price: 4420, change: 0),
    _PriceData(crop: 'Rice', emoji: '🍚', price: 1960, change: 20),
  ];

  // ─── Quick actions ────────────────────────────────────────────────────────
  final List<_QuickAction> _actions = const [
    _QuickAction(
      label: 'Crop Rec.',
      icon: Icons.grass_rounded,
      color: Color(0xFF43A047),
      route: '/crop',
    ),
    _QuickAction(
      label: 'Scan Disease',
      icon: Icons.camera_alt_rounded,
      color: Color(0xFFEF6C00),
      route: '/scan',
    ),
    _QuickAction(
      label: 'Ask AI',
      icon: Icons.smart_toy_rounded,
      color: Color(0xFF1565C0),
      route: '/chat',
    ),
    _QuickAction(
      label: 'Mandi Prices',
      icon: Icons.store_rounded,
      color: Color(0xFF6A1B9A),
      route: '/market',
    ),
    _QuickAction(
      label: 'Weather',
      icon: Icons.cloud_rounded,
      color: Color(0xFF00838F),
      route: '/weather',
    ),
  ];

  void _onNavTap(int index) {
    setState(() => _selectedIndex = index);
    const routes = ['/home', '/chat', '/scan', '/market', '/profile'];
    if (index != 0) {
      Navigator.pushNamed(context, routes[index]);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF1F8E9),
      body: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeader(),
            _buildWeatherBanner(),
            _buildQuickActions(),
            _buildMarketPrices(),
            _buildAlertCard(),
            const SizedBox(height: 24),
          ],
        ),
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
  }

  // ─── Section A: Header ────────────────────────────────────────────────────
  Widget _buildHeader() {
    return Container(
      color: Colors.white,
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Good Morning, Kisan Bhai! 🌾',
                      style: GoogleFonts.poppins(
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                        color: const Color(0xFF1B2B1D),
                      ),
                    ),
                    const SizedBox(height: 2),
                    Row(
                      children: [
                        const Icon(
                          Icons.location_on_outlined,
                          size: 13,
                          color: Color(0xFF2E7D32),
                        ),
                        const SizedBox(width: 2),
                        Text(
                          'Bengaluru, Karnataka',
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                color: const Color(0xFF6D7D6F),
                                fontFamily: GoogleFonts.notoSans().fontFamily,
                              ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              IconButton(
                onPressed: () {},
                icon: const Icon(Icons.notifications_outlined),
                color: const Color(0xFF1B2B1D),
              ),
              const SizedBox(width: 4),
              CircleAvatar(
                radius: 20,
                backgroundColor: const Color(0xFF2E7D32).withValues(alpha: 0.15),
                child: const Icon(
                  Icons.person_rounded,
                  color: Color(0xFF2E7D32),
                  size: 22,
                ),
              ),
            ],
          ),
        ),
      ),
    ).animate().fadeIn(duration: 400.ms);
  }

  // ─── Section B: Weather Banner ────────────────────────────────────────────
  Widget _buildWeatherBanner() {
    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.centerLeft,
          end: Alignment.centerRight,
          colors: [Color(0xFF1565C0), Color(0xFF42A5F5)],
        ),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF1565C0).withValues(alpha: 0.3),
            blurRadius: 16,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Left: temperature + conditions
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '32°C',
                      style: GoogleFonts.poppins(
                        color: Colors.white,
                        fontSize: 40,
                        fontWeight: FontWeight.w700,
                        height: 1.0,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Partly Cloudy',
                      style: GoogleFonts.notoSans(
                        color: Colors.white70,
                        fontSize: 15,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        _buildWeatherMetric('💧', '65%', 'Humidity'),
                        const SizedBox(width: 16),
                        _buildWeatherMetric('💨', '14km/h', 'Wind'),
                      ],
                    ),
                  ],
                ),
              ),
              // Right: sun icon
              const Icon(
                Icons.wb_sunny_rounded,
                color: Color(0xFFFFD54F),
                size: 64,
              ),
            ],
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.18),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.check_circle_outline,
                    color: Colors.white70, size: 14),
                const SizedBox(width: 6),
                Text(
                  'Perfect day for sowing!',
                  style: GoogleFonts.notoSans(
                    color: Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    )
        .animate()
        .fadeIn(delay: 100.ms, duration: 500.ms)
        .slideX(begin: 0.2, curve: Curves.easeOut);
  }

  Widget _buildWeatherMetric(String emoji, String value, String label) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(emoji, style: const TextStyle(fontSize: 14)),
            const SizedBox(width: 4),
            Text(
              value,
              style: GoogleFonts.poppins(
                color: Colors.white,
                fontSize: 14,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
        Text(
          label,
          style: GoogleFonts.notoSans(
            color: Colors.white60,
            fontSize: 11,
          ),
        ),
      ],
    );
  }

  // ─── Section C: Quick Actions ─────────────────────────────────────────────
  Widget _buildQuickActions() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
          child: Text(
            'Quick Actions',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.w700,
                  color: const Color(0xFF1B2B1D),
                  fontFamily: GoogleFonts.poppins().fontFamily,
                ),
          ),
        ).animate().fadeIn(delay: 200.ms),
        GridView.count(
          crossAxisCount: 3,
          padding: const EdgeInsets.all(16),
          crossAxisSpacing: 12,
          mainAxisSpacing: 12,
          childAspectRatio: 0.9,
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          children: List.generate(_actions.length, (index) {
            return _QuickActionCard(
              action: _actions[index],
              delay: 200 + (index * 80),
            );
          }),
        ),
      ],
    );
  }

  // ─── Section D: Market Prices ─────────────────────────────────────────────
  Widget _buildMarketPrices() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 16, top: 16, bottom: 8),
          child: Text(
            "Today's Market Prices",
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.w700,
                  color: const Color(0xFF1B2B1D),
                  fontFamily: GoogleFonts.poppins().fontFamily,
                ),
          ),
        ).animate().fadeIn(delay: 350.ms),
        SizedBox(
          height: 130,
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: 16),
            physics: const BouncingScrollPhysics(),
            itemCount: _prices.length,
            itemBuilder: (context, index) {
              return _PriceCard(data: _prices[index]);
            },
          ),
        ),
      ],
    );
  }

  // ─── Section E: Alert Card ────────────────────────────────────────────────
  Widget _buildAlertCard() {
    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: const Border(
          left: BorderSide(color: Color(0xFFEF6C00), width: 4),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(
                Icons.warning_amber_rounded,
                color: Color(0xFFEF6C00),
                size: 22,
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  '🚨 Pest Advisory: High aphid risk in your area this week.',
                  style: GoogleFonts.notoSans(
                    fontSize: 14,
                    color: const Color(0xFF1B2B1D),
                    height: 1.4,
                  ),
                ),
              ),
            ],
          ),
          Align(
            alignment: Alignment.centerRight,
            child: TextButton(
              onPressed: () {},
              style: TextButton.styleFrom(
                foregroundColor: const Color(0xFFEF6C00),
                textStyle: GoogleFonts.poppins(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
                padding: EdgeInsets.zero,
                minimumSize: const Size(0, 32),
              ),
              child: const Text('Read More →'),
            ),
          ),
        ],
      ),
    )
        .animate()
        .fadeIn(delay: 450.ms, duration: 500.ms)
        .slideY(begin: 0.1, curve: Curves.easeOut);
  }

  // ─── Bottom Navigation ────────────────────────────────────────────────────
  Widget _buildBottomNav() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.08),
            blurRadius: 16,
            offset: const Offset(0, -4),
          ),
        ],
      ),
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 8),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _NavItem(
                icon: Icons.home_rounded,
                label: 'Home',
                isSelected: _selectedIndex == 0,
                onTap: () => _onNavTap(0),
              ),
              _NavItem(
                icon: Icons.chat_bubble_rounded,
                label: 'Chat',
                isSelected: _selectedIndex == 1,
                onTap: () => _onNavTap(1),
              ),

              // FAB-style Scan button
              GestureDetector(
                onTap: () => _onNavTap(2),
                child: Container(
                  width: 58,
                  height: 58,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [Color(0xFF43A047), Color(0xFF2E7D32)],
                    ),
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: const Color(0xFF2E7D32).withValues(alpha: 0.4),
                        blurRadius: 12,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: const Icon(
                    Icons.qr_code_scanner_rounded,
                    color: Colors.white,
                    size: 28,
                  ),
                ),
              ),

              _NavItem(
                icon: Icons.show_chart_rounded,
                label: 'Market',
                isSelected: _selectedIndex == 3,
                onTap: () => _onNavTap(3),
              ),
              _NavItem(
                icon: Icons.person_rounded,
                label: 'Profile',
                isSelected: _selectedIndex == 4,
                onTap: () => _onNavTap(4),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─── Quick Action Card ────────────────────────────────────────────────────────

class _QuickAction {
  final String label;
  final IconData icon;
  final Color color;
  final String route;

  const _QuickAction({
    required this.label,
    required this.icon,
    required this.color,
    required this.route,
  });
}

class _QuickActionCard extends StatelessWidget {
  final _QuickAction action;
  final int delay;

  const _QuickActionCard({required this.action, required this.delay});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => Navigator.pushNamed(context, action.route),
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.06),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: action.color.withValues(alpha: 0.12),
                shape: BoxShape.circle,
              ),
              child: Icon(action.icon, color: action.color, size: 26),
            ),
            const SizedBox(height: 8),
            Text(
              action.label,
              textAlign: TextAlign.center,
              style: GoogleFonts.poppins(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: const Color(0xFF1B2B1D),
              ),
            ),
          ],
        ),
      ),
    )
        .animate()
        .scale(
          delay: Duration(milliseconds: delay),
          duration: 350.ms,
          begin: const Offset(0.7, 0.7),
          curve: Curves.easeOut,
        )
        .fadeIn(delay: Duration(milliseconds: delay));
  }
}

// ─── Price Card ───────────────────────────────────────────────────────────────

class _PriceData {
  final String crop;
  final String emoji;
  final int price;
  final int change;

  const _PriceData({
    required this.crop,
    required this.emoji,
    required this.price,
    required this.change,
  });
}

class _PriceCard extends StatelessWidget {
  final _PriceData data;

  const _PriceCard({required this.data});

  @override
  Widget build(BuildContext context) {
    final isUp = data.change > 0;
    final isDown = data.change < 0;
    final changeColor = isUp
        ? const Color(0xFF2E7D32)
        : isDown
            ? const Color(0xFFC62828)
            : const Color(0xFF757575);
    final changeArrow = isUp ? '▲' : isDown ? '▼' : '─';
    final changeText = isUp
        ? '+${data.change}'
        : isDown
            ? '${data.change}'
            : '0';

    return Container(
      width: 160,
      margin: const EdgeInsets.only(right: 12),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 6,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              Text(data.emoji, style: const TextStyle(fontSize: 20)),
              const SizedBox(width: 6),
              Text(
                data.crop,
                style: GoogleFonts.poppins(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: const Color(0xFF1B2B1D),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            '₹${data.price}/q',
            style: GoogleFonts.poppins(
              fontSize: 18,
              fontWeight: FontWeight.w700,
              color: const Color(0xFF1B2B1D),
            ),
          ),
          Row(
            children: [
              Text(
                changeArrow,
                style: TextStyle(
                  color: changeColor,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(width: 3),
              Text(
                changeText,
                style: GoogleFonts.notoSans(
                  color: changeColor,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ─── Nav Item ─────────────────────────────────────────────────────────────────

class _NavItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool isSelected;
  final VoidCallback onTap;

  const _NavItem({
    required this.icon,
    required this.label,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final color =
        isSelected ? const Color(0xFF2E7D32) : const Color(0xFF9E9E9E);

    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: SizedBox(
        width: 56,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              decoration: BoxDecoration(
                color: isSelected
                    ? const Color(0xFF2E7D32).withValues(alpha: 0.12)
                    : Colors.transparent,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, color: color, size: 24),
            ),
            const SizedBox(height: 2),
            Text(
              label,
              style: GoogleFonts.poppins(
                fontSize: 10,
                fontWeight:
                    isSelected ? FontWeight.w600 : FontWeight.w400,
                color: color,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
