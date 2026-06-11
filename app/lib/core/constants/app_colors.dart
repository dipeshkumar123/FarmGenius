import 'package:flutter/material.dart';

/// Centralised colour palette for FarmGenius.
///
/// All UI components must reference colours from this class so that a future
/// dark-mode or whitelabel variant can be introduced by changing a single file.
class AppColors {
  // ─── Private constructor prevents instantiation ───────────────────────────
  const AppColors._();

  // ─── Primary: deep forest-green ──────────────────────────────────────────
  static const Color primary = Color(0xFF2E7D32);
  static const Color primaryLight = Color(0xFF60AD5E);
  static const Color primaryDark = Color(0xFF1B5E20);

  // ─── Secondary: earthy olive-green ───────────────────────────────────────
  static const Color secondary = Color(0xFF558B2F);

  // ─── Accent: warm amber (harvest gold) ───────────────────────────────────
  static const Color accent = Color(0xFFF9A825);
  static const Color accentDark = Color(0xFFF57F17);

  // ─── Backgrounds & surfaces ───────────────────────────────────────────────
  static const Color background = Color(0xFFF1F8E9);
  static const Color surface = Color(0xFFFFFFFF);
  static const Color surfaceVariant = Color(0xFFE8F5E9);

  // ─── Semantic colours ────────────────────────────────────────────────────
  static const Color error = Color(0xFFC62828);
  static const Color warning = Color(0xFFEF6C00);
  static const Color success = Color(0xFF388E3C);

  // ─── Text ────────────────────────────────────────────────────────────────
  static const Color textPrimary = Color(0xFF1B2B1D);
  static const Color textSecondary = Color(0xFF546E7A);
  static const Color textOnPrimary = Color(0xFFFFFFFF);

  // ─── Decorative ──────────────────────────────────────────────────────────
  static const Color divider = Color(0xFFC8E6C9);

  // ─── Weather feature gradient ────────────────────────────────────────────
  static const Color weatherBlueStart = Color(0xFF1565C0);
  static const Color weatherBlueEnd = Color(0xFF42A5F5);
}
