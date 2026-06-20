import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'core/storage/offline_manager.dart';
import 'core/theme/app_theme.dart';
import 'features/auth/splash_screen.dart';
import 'features/auth/onboarding_screen.dart';
import 'features/auth/login_screen.dart';
import 'features/home/home_screen.dart';
import 'features/chat/voice_chat_screen.dart';
import 'features/disease/disease_detect_screen.dart';
import 'features/market/mandi_prices_screen.dart';
import 'features/weather/weather_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Lock to portrait orientation — farming app is portrait-first.
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  // Set system UI overlay style to match our green-on-white brand.
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
      systemNavigationBarColor: Colors.white,
      systemNavigationBarIconBrightness: Brightness.dark,
    ),
  );

  // Initialise offline storage (Hive).
  // OfflineManager.init() calls Hive.initFlutter() internally, then
  // opens all required boxes — no direct Hive call needed here.
  await OfflineManager.init();

  runApp(
    // Riverpod scope wraps the entire widget tree.
    const ProviderScope(child: FarmGeniusApp()),
  );
}

class FarmGeniusApp extends StatelessWidget {
  const FarmGeniusApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FarmGenius',
      debugShowCheckedModeBanner: false,

      // ── Brand theme (Material 3, Poppins + Noto Sans) ──────────────────
      theme: AppTheme.light,

      // ── Named route table ──────────────────────────────────────────────
      // Simple named-route navigation — clean, no extra packages needed
      // at this stage. GoRouter can replace this in a future iteration.
      initialRoute: '/splash',
      routes: {
        '/splash':   (_) => const SplashScreen(),
        '/onboarding': (_) => const OnboardingScreen(),
        '/login':    (_) => const LoginScreen(),
        '/home':     (_) => const HomeScreen(),
        '/chat':     (_) => const VoiceChatScreen(),
        '/scan':     (_) => const DiseaseDetectScreen(),
        '/market':   (_) => const MandiPricesScreen(),
        '/weather':  (_) => const WeatherScreen(),
      },

      // Custom page transition — subtle fade for a polished feel.
      onGenerateRoute: (settings) {
        final routes = {
          '/splash':     const SplashScreen(),
          '/onboarding': const OnboardingScreen(),
          '/login':      const LoginScreen(),
          '/home':       const HomeScreen(),
          '/chat':       const VoiceChatScreen(),
          '/scan':       const DiseaseDetectScreen(),
          '/market':     const MandiPricesScreen(),
          '/weather':    const WeatherScreen(),
        };

        final page = routes[settings.name];
        if (page == null) return null;

        return PageRouteBuilder(
          settings: settings,
          pageBuilder: (_, __, ___) => page,
          transitionsBuilder: (_, animation, __, child) {
            return FadeTransition(
              opacity: CurvedAnimation(
                parent: animation,
                curve: Curves.easeInOut,
              ),
              child: child,
            );
          },
          transitionDuration: const Duration(milliseconds: 250),
        );
      },
    );
  }
}
