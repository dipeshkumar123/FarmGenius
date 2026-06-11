import 'package:flutter/material.dart';
import '../../core/constants/app_colors.dart';
import '../../core/constants/app_spacing.dart';

/// A branded white card with a subtle green-tinted drop shadow.
///
/// Use this as the primary container for all dashboard panels, list items,
/// and detail sections inside FarmGenius.
///
/// Shadow specification (per design):
///   colour: #2E7D32 @ 8% opacity, blurRadius 12, offset Offset(0, 4)
///
/// Example:
/// ```dart
/// AppCard(
///   child: ListTile(
///     title: Text('Tomato — Early Blight'),
///     subtitle: Text('Confidence: 94%'),
///   ),
/// )
/// ```
class AppCard extends StatelessWidget {
  const AppCard({
    super.key,
    required this.child,
    this.padding,
    this.margin,
  });

  /// The widget rendered inside the card.
  final Widget child;

  /// Inner padding. Defaults to [AppSpacing.md] on all sides.
  final EdgeInsetsGeometry? padding;

  /// Outer margin around the card. Defaults to zero (caller controls spacing).
  final EdgeInsetsGeometry? margin;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: margin ?? EdgeInsets.zero,
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(AppRadius.md),
        boxShadow: const [
          BoxShadow(
            // Green-tinted shadow — 0x14 ≈ 8% of 0xFF (255 × 0.08 ≈ 20 → 0x14).
            color: Color(0x142E7D32),
            blurRadius: 12,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(AppRadius.md),
        child: Padding(
          padding: padding ?? const EdgeInsets.all(AppSpacing.md),
          child: child,
        ),
      ),
    );
  }
}
