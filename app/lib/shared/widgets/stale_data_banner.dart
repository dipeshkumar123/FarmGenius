import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

/// A non-intrusive banner displayed when locally-cached data is being shown
/// because the device is offline or a fresh fetch failed.
///
/// The banner renders a warm amber/yellow strip with:
///   - A wifi_off icon to communicate the offline state at a glance.
///   - A human-readable "updated X minutes/hours ago" label.
///   - A "Refresh" text button that triggers [onRefresh].
///
/// Place it directly below the app bar or at the top of a scrollable list:
/// ```dart
/// Column(
///   children: [
///     StaleDataBanner(
///       lastUpdated: cachedAt,
///       onRefresh: _reloadPrices,
///     ),
///     Expanded(child: PriceList(prices: cachedPrices)),
///   ],
/// )
/// ```
class StaleDataBanner extends StatelessWidget {
  const StaleDataBanner({
    super.key,
    required this.lastUpdated,
    required this.onRefresh,
  });

  /// The [DateTime] when the data was last successfully fetched.
  final DateTime lastUpdated;

  /// Callback invoked when the user taps the "Refresh" button.
  final VoidCallback onRefresh;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: const BoxDecoration(
        // Warm amber tint — visually distinct from primary green, signals caution
        // without using alarming error-red.
        color: Color(0xFFFFF8E1),
        border: Border(
          bottom: BorderSide(color: Color(0xFFFFE082), width: 1),
        ),
      ),
      child: Row(
        children: [
          // ── Offline icon ────────────────────────────────────────────────
          const Icon(
            Icons.wifi_off_rounded,
            size: 16,
            color: Color(0xFFEF6C00),
          ),
          const SizedBox(width: 6),

          // ── Stale data message ──────────────────────────────────────────
          Expanded(
            child: Text(
              _buildTimeAgoText(),
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w500,
                color: Color(0xFF795548),
              ),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ),

          // ── Refresh action ──────────────────────────────────────────────
          TextButton(
            onPressed: onRefresh,
            style: TextButton.styleFrom(
              foregroundColor: const Color(0xFFEF6C00),
              padding: const EdgeInsets.symmetric(horizontal: 8),
              minimumSize: const Size(0, 30),
              tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              textStyle: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w700,
              ),
            ),
            child: const Text('Refresh'),
          ),
        ],
      ),
    );
  }

  // ─── Human-readable time-ago text ──────────────────────────────────────────
  String _buildTimeAgoText() {
    final Duration diff = DateTime.now().difference(lastUpdated);

    if (diff.inMinutes < 1) {
      return 'Showing cached data — updated just now';
    } else if (diff.inMinutes < 60) {
      final int mins = diff.inMinutes;
      return 'Showing cached data — updated $mins ${mins == 1 ? 'min' : 'mins'} ago';
    } else if (diff.inHours < 24) {
      final int hrs = diff.inHours;
      return 'Showing cached data — updated $hrs ${hrs == 1 ? 'hour' : 'hours'} ago';
    } else {
      // For data older than a day, show the actual date so the farmer knows
      // how stale it really is (mandi prices change daily — this matters).
      final String formatted = DateFormat('d MMM, h:mm a').format(lastUpdated);
      return 'Cached data from $formatted';
    }
  }
}
