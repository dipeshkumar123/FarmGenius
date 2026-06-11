import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  late AnimationController _iconController;
  late AnimationController _dot1Controller;
  late AnimationController _dot2Controller;
  late AnimationController _dot3Controller;

  late Animation<double> _iconScale;
  late Animation<double> _iconOpacity;

  late Animation<double> _dot1Anim;
  late Animation<double> _dot2Anim;
  late Animation<double> _dot3Anim;

  @override
  void initState() {
    super.initState();

    // ── Icon animation (scale + fade) ──────────────────────────
    _iconController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    );

    _iconScale = CurvedAnimation(
      parent: _iconController,
      curve: Curves.easeOut,
    ).drive(Tween<double>(begin: 0.3, end: 1.0));

    _iconOpacity = CurvedAnimation(
      parent: _iconController,
      curve: Curves.easeOut,
    ).drive(Tween<double>(begin: 0.0, end: 1.0));

    _iconController.forward();

    // ── Staggered dot animations ───────────────────────────────
    _dot1Controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    )..repeat(reverse: true);

    _dot2Controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );

    _dot3Controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );

    _dot1Anim = _dot1Controller.drive(Tween<double>(begin: 0.3, end: 1.0));
    _dot2Anim = _dot2Controller.drive(Tween<double>(begin: 0.3, end: 1.0));
    _dot3Anim = _dot3Controller.drive(Tween<double>(begin: 0.3, end: 1.0));

    // Stagger the dot starts
    Future.delayed(const Duration(milliseconds: 200), () {
      if (mounted) _dot2Controller.repeat(reverse: true);
    });
    Future.delayed(const Duration(milliseconds: 400), () {
      if (mounted) _dot3Controller.repeat(reverse: true);
    });

    // ── Auto-navigate after 2.5s ───────────────────────────────
    Future.delayed(const Duration(milliseconds: 2500), () {
      if (mounted) {
        Navigator.pushReplacementNamed(context, '/home');
      }
    });
  }

  @override
  void dispose() {
    _iconController.dispose();
    _dot1Controller.dispose();
    _dot2Controller.dispose();
    _dot3Controller.dispose();
    super.dispose();
  }

  Widget _buildDot(Animation<double> anim) {
    return AnimatedBuilder(
      animation: anim,
      builder: (context, _) {
        return Opacity(
          opacity: anim.value,
          child: Container(
            width: 10,
            height: 10,
            margin: const EdgeInsets.symmetric(horizontal: 5),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(5),
            ),
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Color(0xFF1B5E20),
              Color(0xFF2E7D32),
              Color(0xFF388E3C),
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // ── Animated leaf icon ────────────────────────────
              AnimatedBuilder(
                animation: _iconController,
                builder: (context, _) {
                  return Transform.scale(
                    scale: _iconScale.value,
                    child: Opacity(
                      opacity: _iconOpacity.value,
                      child: Container(
                        width: 130,
                        height: 130,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.15),
                          shape: BoxShape.circle,
                          boxShadow: [
                            BoxShadow(
                              color: Colors.white.withOpacity(0.12),
                              blurRadius: 40,
                              spreadRadius: 10,
                            ),
                          ],
                        ),
                        child: const Icon(
                          Icons.eco,
                          size: 80,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  );
                },
              ),

              const SizedBox(height: 24),

              // ── App name ──────────────────────────────────────
              Text(
                'FarmGenius',
                style: GoogleFonts.poppins(
                  color: Colors.white,
                  fontSize: 36,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.5,
                ),
              ),

              // ── Tagline ───────────────────────────────────────
              Text(
                'Smart Farming Assistant',
                style: GoogleFonts.notoSans(
                  color: Colors.white70,
                  fontSize: 16,
                ),
              ),

              const SizedBox(height: 48),

              // ── Loading dots ──────────────────────────────────
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  _buildDot(_dot1Anim),
                  _buildDot(_dot2Anim),
                  _buildDot(_dot3Anim),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
