// src/components/ui/StaleDataBanner.tsx
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowClockwise, Clock, X } from 'phosphor-react';
import { useState } from 'react';

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────
interface StaleDataBannerProps {
  /**
   * How many minutes ago the data was fetched.
   * Used to display "X minutes ago" or "X hours ago".
   */
  staleSinceMinutes: number;
  /**
   * Callback fired when the user taps "Refresh".
   * Typically triggers a React Query refetch.
   */
  onRefresh: () => void;
  /**
   * Whether a refresh is currently in progress.
   * Shows a spinner on the refresh button when true.
   */
  isRefreshing?: boolean;
  /**
   * Whether the banner is visible. Use this to control visibility
   * from the parent (e.g. hide after successful refresh).
   * Defaults to true.
   */
  visible?: boolean;
  /**
   * Whether to allow the user to dismiss the banner manually.
   * Defaults to true.
   */
  dismissible?: boolean;
  /** Additional Tailwind classes on the banner container. */
  className?: string;
}

// ─────────────────────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────────────────────
function formatStaleDuration(minutes: number): string {
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? '' : 's'} ago`;
  const hours = Math.round(minutes / 60);
  return `${hours} hour${hours === 1 ? '' : 's'} ago`;
}

// ─────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────
/**
 * StaleDataBanner — an orange/amber notification strip that appears
 * when cached data is being shown instead of fresh API data.
 *
 * Slides down from the top with spring animation.
 * Includes a refresh button and optional dismiss (X) control.
 *
 * Usage:
 * ```tsx
 * const { data, refetch, isFetching } = useQuery(...);
 * const isStale = dataUpdatedAt < Date.now() - 30 * 60 * 1000;
 *
 * <StaleDataBanner
 *   staleSinceMinutes={45}
 *   onRefresh={refetch}
 *   isRefreshing={isFetching}
 *   visible={isStale}
 * />
 * ```
 */
export function StaleDataBanner({
  staleSinceMinutes,
  onRefresh,
  isRefreshing = false,
  visible = true,
  dismissible = true,
  className = '',
}: StaleDataBannerProps) {
  const [isDismissed, setIsDismissed] = useState(false);

  const shouldShow = visible && !isDismissed;

  return (
    <AnimatePresence mode="wait">
      {shouldShow && (
        <motion.div
          key="stale-banner"
          initial={{ y: -60, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: -60, opacity: 0 }}
          transition={{
            type: 'spring',
            stiffness: 380,
            damping: 30,
          }}
          className={`
            flex items-center gap-3 px-4 py-2.5 mx-0
            bg-amber-50 border border-amber-200 rounded-md
            shadow-[0_2px_8px_rgba(239,108,0,0.12)]
            ${className}
          `}
          role="status"
          aria-live="polite"
          aria-label="Stale data warning"
        >
          {/* Clock icon */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.15, type: 'spring', stiffness: 500 }}
            className="shrink-0"
          >
            <Clock
              size={16}
              weight="fill"
              className="text-amber-600"
              aria-hidden="true"
            />
          </motion.div>

          {/* Text content */}
          <div className="flex-1 min-w-0">
            <p className="text-sm font-noto text-amber-800 leading-tight">
              Showing cached data from{' '}
              <span className="font-semibold">
                {formatStaleDuration(staleSinceMinutes)}
              </span>
              .{' '}
              <button
                onClick={onRefresh}
                disabled={isRefreshing}
                className="underline underline-offset-2 font-semibold hover:text-amber-900
                           disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
                aria-label="Refresh data now"
              >
                Tap to refresh.
              </button>
            </p>
          </div>

          {/* Refresh button */}
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="shrink-0 flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs
                       font-poppins font-semibold bg-amber-200 text-amber-900
                       hover:bg-amber-300 active:scale-95 transition-all duration-200
                       disabled:opacity-60 disabled:cursor-not-allowed min-h-[32px]"
            aria-label={isRefreshing ? 'Refreshing data…' : 'Refresh data'}
          >
            <motion.span
              animate={{ rotate: isRefreshing ? 360 : 0 }}
              transition={{
                duration: 0.8,
                repeat: isRefreshing ? Infinity : 0,
                
              }}
            >
              <ArrowClockwise size={13} weight="bold" aria-hidden="true" />
            </motion.span>
            {isRefreshing ? 'Updating…' : 'Refresh'}
          </button>

          {/* Dismiss button */}
          {dismissible && !isRefreshing && (
            <button
              onClick={() => setIsDismissed(true)}
              className="shrink-0 flex items-center justify-center w-7 h-7 rounded-full
                         text-amber-600 hover:bg-amber-200 active:scale-90
                         transition-all duration-200"
              aria-label="Dismiss stale data warning"
            >
              <X size={14} weight="bold" />
            </button>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ─────────────────────────────────────────────────────────────
// Inline stale indicator (for use inside cards, not full-width)
// ─────────────────────────────────────────────────────────────
interface StaleIndicatorProps {
  staleSinceMinutes: number;
  onRefresh?: () => void;
  className?: string;
}

/**
 * StaleIndicator — a compact pill badge for use inside cards.
 * Shows "Updated 2h ago" with an optional refresh tap.
 */
export function StaleIndicator({
  staleSinceMinutes,
  onRefresh,
  className = '',
}: StaleIndicatorProps) {
  return (
    <motion.button
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={onRefresh}
      className={`inline-flex items-center gap-1 px-2 py-1 rounded-full
                 bg-amber-100 text-amber-700 text-xs font-poppins font-semibold
                 cursor-pointer hover:bg-amber-200 transition-colors duration-150 ${className}`}
      disabled={!onRefresh}
      aria-label={`Data is ${formatStaleDuration(staleSinceMinutes)} old. ${onRefresh ? 'Tap to refresh.' : ''}`}
    >
      <Clock size={11} weight="fill" />
      {formatStaleDuration(staleSinceMinutes)}
      {onRefresh && <ArrowClockwise size={11} weight="bold" />}
    </motion.button>
  );
}

export default StaleDataBanner;
