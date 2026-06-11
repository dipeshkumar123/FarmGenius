/// Spacing scale used throughout the FarmGenius UI.
///
/// All paddings, margins, and gaps must be drawn from this class to ensure a
/// consistent 4-pt baseline grid across the app.
class AppSpacing {
  // ─── Private constructor prevents instantiation ───────────────────────────
  const AppSpacing._();

  /// 4 dp — micro gap, icon padding.
  static const double xs = 4.0;

  /// 8 dp — tight spacing inside cards, chip padding.
  static const double sm = 8.0;

  /// 16 dp — standard section padding, list tile inner gap.
  static const double md = 16.0;

  /// 24 dp — generous section spacing, card internal padding.
  static const double lg = 24.0;

  /// 32 dp — between major page sections.
  static const double xl = 32.0;

  /// 48 dp — hero areas, illustration breathing room.
  static const double xxl = 48.0;
}

/// Border-radius tokens.
///
/// Using named radii keeps widget code readable:
///   `borderRadius: BorderRadius.circular(AppRadius.md)`
class AppRadius {
  // ─── Private constructor prevents instantiation ───────────────────────────
  const AppRadius._();

  /// 8 dp — chips, small badges.
  static const double sm = 8.0;

  /// 16 dp — cards, dialogs, bottom-sheets.
  static const double md = 16.0;

  /// 24 dp — large panels, bottom navigation sheets.
  static const double lg = 24.0;

  /// 100 dp — pill-shaped buttons and tags.
  static const double full = 100.0;
}
