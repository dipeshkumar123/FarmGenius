import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:smooth_page_indicator/smooth_page_indicator.dart';

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final PageController _pageController = PageController();
  int _currentPage = 0;

  final List<_OnboardingData> _slides = [
    _OnboardingData(
      icon: Icons.auto_awesome,
      iconColor: const Color(0xFFFFC107),
      gradientColors: [const Color(0xFFFFF8E1), const Color(0xFFFFECB3)],
      title: 'Get Personalized\nCrop Advice',
      subtitle:
          'Tell the AI about your soil and location — get tailored recommendations for maximum yield.',
    ),
    _OnboardingData(
      icon: Icons.camera_alt_rounded,
      iconColor: const Color(0xFFEF6C00),
      gradientColors: [const Color(0xFFFFF3E0), const Color(0xFFFFE0B2)],
      title: 'Scan & Diagnose\nCrop Diseases',
      subtitle:
          'Point camera at a leaf. Get instant disease diagnosis + a step-by-step treatment plan.',
    ),
    _OnboardingData(
      icon: Icons.trending_up_rounded,
      iconColor: const Color(0xFF7B1FA2),
      gradientColors: [const Color(0xFFF3E5F5), const Color(0xFFE1BEE7)],
      title: 'Know When\nto Sell',
      subtitle:
          'Real-time mandi prices and AI-powered trend forecasts for your crops — sell at peak value.',
    ),
    _OnboardingData(
      icon: Icons.record_voice_over_rounded,
      iconColor: const Color(0xFF1565C0),
      gradientColors: [const Color(0xFFE3F2FD), const Color(0xFFBBDEFB)],
      title: 'Voice-First,\nAny Language',
      subtitle:
          'Ask questions in Hindi, Kannada, Telugu, Tamil — FarmGenius understands you perfectly.',
    ),
  ];

  void _goToPage(int page) {
    _pageController.animateToPage(
      page,
      duration: const Duration(milliseconds: 400),
      curve: Curves.easeInOut,
    );
  }

  void _nextOrFinish() {
    if (_currentPage < _slides.length - 1) {
      _goToPage(_currentPage + 1);
    } else {
      Navigator.pushReplacementNamed(context, '/login');
    }
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF1F8E9),
      body: SafeArea(
        child: Stack(
          children: [
            // ── Page content ────────────────────────────────────
            PageView.builder(
              controller: _pageController,
              itemCount: _slides.length,
              onPageChanged: (i) => setState(() => _currentPage = i),
              itemBuilder: (context, index) {
                return _OnboardingPage(data: _slides[index]);
              },
            ),

            // ── Skip button ──────────────────────────────────────
            Positioned(
              top: 12,
              right: 16,
              child: TextButton(
                onPressed: () =>
                    Navigator.pushReplacementNamed(context, '/login'),
                style: TextButton.styleFrom(
                  foregroundColor: const Color(0xFF2E7D32),
                  textStyle: GoogleFonts.poppins(
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                child: const Text('Skip'),
              ).animate().fadeIn(delay: 300.ms),
            ),

            // ── Bottom controls ──────────────────────────────────
            Positioned(
              bottom: 36,
              left: 0,
              right: 0,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Dot indicator
                  SmoothPageIndicator(
                    controller: _pageController,
                    count: _slides.length,
                    effect: WormEffect(
                      dotWidth: 10,
                      dotHeight: 10,
                      spacing: 8,
                      dotColor: Colors.grey.shade300,
                      activeDotColor: const Color(0xFF2E7D32),
                    ),
                  ),

                  const SizedBox(height: 32),

                  // Next / Get Started button
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 32),
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 300),
                      width: double.infinity,
                      height: 56,
                      child: ElevatedButton(
                        onPressed: _nextOrFinish,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF2E7D32),
                          foregroundColor: Colors.white,
                          minimumSize: const Size(double.infinity, 56),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                          elevation: 4,
                          shadowColor:
                              const Color(0xFF2E7D32).withOpacity(0.4),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(
                              _currentPage == _slides.length - 1
                                  ? 'Get Started'
                                  : 'Next',
                              style: GoogleFonts.poppins(
                                fontSize: 17,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            const SizedBox(width: 8),
                            const Icon(Icons.arrow_forward_rounded, size: 20),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ─── Slide data model ────────────────────────────────────────────────────────

class _OnboardingData {
  final IconData icon;
  final Color iconColor;
  final List<Color> gradientColors;
  final String title;
  final String subtitle;

  const _OnboardingData({
    required this.icon,
    required this.iconColor,
    required this.gradientColors,
    required this.title,
    required this.subtitle,
  });
}

// ─── Single slide ────────────────────────────────────────────────────────────

class _OnboardingPage extends StatelessWidget {
  final _OnboardingData data;

  const _OnboardingPage({required this.data});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(32, 80, 32, 160),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Icon bubble
          Container(
            width: 160,
            height: 160,
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: data.gradientColors,
              ),
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                  color: data.iconColor.withOpacity(0.25),
                  blurRadius: 32,
                  spreadRadius: 8,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            child: Icon(
              data.icon,
              size: 80,
              color: data.iconColor,
            ),
          )
              .animate()
              .fadeIn(duration: 500.ms)
              .slideY(begin: -0.15, curve: Curves.easeOut),

          const SizedBox(height: 40),

          // Title
          Text(
            data.title,
            textAlign: TextAlign.center,
            style: GoogleFonts.poppins(
              fontSize: 28,
              fontWeight: FontWeight.w700,
              color: const Color(0xFF1B2B1D),
              height: 1.25,
            ),
          )
              .animate()
              .fadeIn(delay: 100.ms, duration: 500.ms)
              .slideX(begin: 0.1, curve: Curves.easeOut),

          const SizedBox(height: 16),

          // Subtitle
          Text(
            data.subtitle,
            textAlign: TextAlign.center,
            style: GoogleFonts.notoSans(
              fontSize: 16,
              color: const Color(0xFF5A7A5C),
              height: 1.5,
            ),
          )
              .animate()
              .fadeIn(delay: 200.ms, duration: 500.ms)
              .slideX(begin: 0.1, curve: Curves.easeOut),
        ],
      ),
    );
  }
}
