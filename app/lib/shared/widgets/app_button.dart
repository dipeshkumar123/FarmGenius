import 'package:flutter/material.dart';
import '../../core/constants/app_colors.dart';
import '../../core/constants/app_spacing.dart';

/// A full-width, pill-shaped primary action button used throughout FarmGenius.
///
/// Features:
///   - Shows a white [CircularProgressIndicator] when [isLoading] is true.
///   - Automatically disables itself when [isLoading] is `true` or
///     [onPressed] is `null`, preventing double-taps.
///   - Height is fixed at 52 dp to match the [ElevatedButtonTheme] minimum size.
///
/// Example:
/// ```dart
/// AppButton(
///   label: 'Get Advice',
///   onPressed: _handleSubmit,
///   isLoading: _isSubmitting,
/// )
/// ```
class AppButton extends StatelessWidget {
  const AppButton({
    super.key,
    required this.label,
    required this.onPressed,
    this.isLoading = false,
  });

  /// Button label displayed in the centre.
  final String label;

  /// Callback invoked when the button is tapped. Pass `null` to disable.
  final VoidCallback? onPressed;

  /// When `true`, replaces the label with a loading spinner and prevents taps.
  final bool isLoading;

  @override
  Widget build(BuildContext context) {
    final bool isDisabled = isLoading || onPressed == null;

    return SizedBox(
      width: double.infinity,
      height: 52,
      child: ElevatedButton(
        // Disable the button while loading or when no callback is provided.
        onPressed: isDisabled ? null : onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.primary,
          disabledBackgroundColor: AppColors.primaryLight.withOpacity(0.5),
          shape: const StadiumBorder(),
          elevation: 0,
          shadowColor: Colors.transparent,
          padding: const EdgeInsets.symmetric(horizontal: AppSpacing.lg),
        ),
        child: AnimatedSwitcher(
          duration: const Duration(milliseconds: 200),
          child: isLoading
              ? const SizedBox(
                  key: ValueKey('loading'),
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(
                    color: AppColors.textOnPrimary,
                    strokeWidth: 2,
                  ),
                )
              : Text(
                  key: const ValueKey('label'),
                  label,
                  style: const TextStyle(
                    color: AppColors.textOnPrimary,
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.3,
                  ),
                ),
        ),
      ),
    );
  }
}
