import 'package:flutter/material.dart';
import 'package:shimmer/shimmer.dart';
import '../../core/constants/app_spacing.dart';

/// A green-tinted shimmer placeholder used while async content loads.
///
/// Uses the [shimmer] package to animate between the app's surface-variant
/// colours, matching FarmGenius's agricultural palette.
///
/// Example — replace a card while prices load:
/// ```dart
/// isLoading
///     ? LoadingShimmer(height: 80, borderRadius: AppRadius.md)
///     : PriceCard(data: prices)
/// ```
///
/// For full-width list placeholders, wrap multiple instances:
/// ```dart
/// Column(
///   children: List.generate(
///     4,
///     (_) => Padding(
///       padding: const EdgeInsets.only(bottom: AppSpacing.sm),
///       child: LoadingShimmer(height: 72),
///     ),
///   ),
/// )
/// ```
class LoadingShimmer extends StatelessWidget {
  const LoadingShimmer({
    super.key,
    required this.height,
    this.borderRadius = AppRadius.md,
    this.width,
  });

  /// Height of the shimmer placeholder in logical pixels.
  final double height;

  /// Corner radius. Defaults to [AppRadius.md] (16 dp).
  final double borderRadius;

  /// Optional width. Defaults to [double.infinity] (full-width).
  final double? width;

  @override
  Widget build(BuildContext context) {
    return Shimmer.fromColors(
      // Base: light green surface — matches AppColors.surfaceVariant.
      baseColor: const Color(0xFFE8F5E9),
      // Highlight: slightly lighter background — matches AppColors.background.
      highlightColor: const Color(0xFFF1F8E9),
      // Period tuned for a calm, non-jarring animation on low-end devices.
      period: const Duration(milliseconds: 1200),
      child: Container(
        height: height,
        width: width ?? double.infinity,
        decoration: BoxDecoration(
          // The shimmer package overrides the colour, but we still need a
          // non-transparent colour for the shimmer to render correctly.
          color: const Color(0xFFE8F5E9),
          borderRadius: BorderRadius.circular(borderRadius),
        ),
      ),
    );
  }
}
