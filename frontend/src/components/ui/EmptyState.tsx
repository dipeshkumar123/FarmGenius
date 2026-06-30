// src/components/ui/EmptyState.tsx
import { motion } from 'framer-motion';
import { ArrowClockwise } from 'phosphor-react';
import { useTranslation } from 'react-i18next';

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────
interface EmptyStateProps {
  /**
   * A React element (usually a Phosphor icon) displayed at the top.
   * Rendered inside a floating-animation wrapper automatically.
   *
   * Example:
   * ```tsx
   * icon={<Plant size={48} weight="duotone" className="text-primary" />}
   * ```
   */
  icon: React.ReactNode;
  /**
   * Short, farmer-friendly heading.
   * Keep to 1-4 words. Written in Poppins semibold.
   */
  title: string;
  /**
   * One or two sentence description in simple language.
   * Written in Noto Sans.
   */
  description?: string;
  /**
   * Label for the primary action button.
   * If omitted, no button is rendered.
   */
  actionLabel?: string;
  /**
   * Callback for the primary action button.
   */
  onAction?: () => void;
  /**
   * Secondary action label (e.g., "Go Back").
   */
  secondaryLabel?: string;
  /**
   * Callback for the secondary action.
   */
  onSecondaryAction?: () => void;
  /**
   * Whether the action button should show a loading spinner.
   * Useful when re-fetching data after the user taps "Try Again".
   */
  isLoading?: boolean;
  /**
   * Visual style variant.
   * 'default' → green theme (standard empty states)
   * 'error'   → red theme (fetch errors)
   * 'offline' → amber theme (no internet)
   */
  variant?: 'default' | 'error' | 'offline';
  /** Additional Tailwind classes on the root container. */
  className?: string;
}

// ─────────────────────────────────────────────────────────────
// Floating icon animation
// ─────────────────────────────────────────────────────────────
const floatVariants = {
  initial: { y: 0 },
  animate: {
    y: [-8, 0, -8],
    transition: {
      duration: 3.2,
      repeat: Infinity,
      
    },
  },
};

// ─────────────────────────────────────────────────────────────
// Container stagger variants
// ─────────────────────────────────────────────────────────────
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.12,
      delayChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.4, },
  },
};

// ─────────────────────────────────────────────────────────────
// Variant style maps
// ─────────────────────────────────────────────────────────────
const variantStyles = {
  default: {
    iconBg: 'bg-surface-variant',
    iconRing: 'ring-farm-divider',
    titleColor: 'text-text-primary',
    descColor: 'text-text-secondary',
    buttonClass: 'btn-primary',
  },
  error: {
    iconBg: 'bg-red-50',
    iconRing: 'ring-red-200',
    titleColor: 'text-farm-error',
    descColor: 'text-text-secondary',
    buttonClass: 'btn-primary',
  },
  offline: {
    iconBg: 'bg-amber-50',
    iconRing: 'ring-amber-200',
    titleColor: 'text-amber-800',
    descColor: 'text-amber-700',
    buttonClass: 'bg-amber-500 hover:bg-amber-600 text-white font-poppins font-semibold py-3 px-6 rounded-full active:scale-95 transition-all duration-200',
  },
};

// ─────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────
/**
 * EmptyState — a centred illustration-style empty screen.
 * Used for:
 * - No data states (e.g., "No price data available")
 * - Error states (e.g., "Could not connect to server")
 * - Offline states
 * - First-run onboarding prompts (e.g., "Start your first chat")
 *
 * Usage:
 * ```tsx
 * import { Plant } from 'phosphor-react';
 * import { EmptyState } from '../ui/EmptyState';
 *
 * <EmptyState
 *   icon={<Plant size={52} weight="duotone" className="text-primary" />}
 *   title="No crops added yet"
 *   description="Add your crops to get personalised advice, weather alerts, and market prices."
 *   actionLabel="Add My First Crop"
 *   onAction={handleAddCrop}
 * />
 * ```
 */
export function EmptyState({
  icon,
  title,
  description,
  actionLabel,
  onAction,
  secondaryLabel,
  onSecondaryAction,
  isLoading = false,
  variant = 'default',
  className = '',
}: EmptyStateProps) {
  const { t } = useTranslation();
  const styles = variantStyles[variant];

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className={`flex flex-col items-center justify-center px-6 py-12 text-center ${className}`}
      role="status"
      aria-label={title}
    >
      {/* Floating icon container */}
      <motion.div variants={itemVariants}>
        <motion.div
          variants={floatVariants}
          initial="initial"
          animate="animate"
          className={`
            w-24 h-24 rounded-full flex items-center justify-center mb-6
            ${styles.iconBg} ring-2 ${styles.iconRing}
            shadow-[0_8px_24px_rgba(46,125,50,0.12)]
          `}
          aria-hidden="true"
        >
          {icon}
        </motion.div>
      </motion.div>

      {/* Title */}
      <motion.h3
        variants={itemVariants}
        className={`font-poppins font-semibold text-xl mb-2 ${styles.titleColor}`}
      >
        {title}
      </motion.h3>

      {/* Description */}
      {description && (
        <motion.p
          variants={itemVariants}
          className={`font-noto text-base leading-relaxed max-w-xs mb-8 ${styles.descColor}`}
        >
          {description}
        </motion.p>
      )}

      {/* Action buttons */}
      {(actionLabel || secondaryLabel) && (
        <motion.div
          variants={itemVariants}
          className="flex flex-col sm:flex-row items-center gap-3 w-full max-w-xs"
        >
          {actionLabel && onAction && (
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.96 }}
              onClick={onAction}
              disabled={isLoading}
              className={`w-full flex items-center justify-center gap-2 min-h-[48px]
                         ${styles.buttonClass}
                         disabled:opacity-60 disabled:cursor-not-allowed`}
              aria-busy={isLoading}
            >
              {isLoading ? (
                <>
                  <motion.span
                    animate={{ rotate: 360 }}
                    transition={{ duration: 0.8, repeat: Infinity, }}
                  >
                    <ArrowClockwise size={16} weight="bold" />
                  </motion.span>
                  <span>{t('components.empty_state.loading', 'Loading…')}</span>
                </>
              ) : (
                actionLabel
              )}
            </motion.button>
          )}

          {secondaryLabel && onSecondaryAction && (
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.96 }}
              onClick={onSecondaryAction}
              className="w-full btn-ghost min-h-[48px]"
            >
              {secondaryLabel}
            </motion.button>
          )}
        </motion.div>
      )}
    </motion.div>
  );
}

// ─────────────────────────────────────────────────────────────
// Preset: Network error
// ─────────────────────────────────────────────────────────────
interface NetworkErrorProps {
  onRetry: () => void;
  isRetrying?: boolean;
  message?: string;
  className?: string;
}

/**
 * NetworkError — a pre-configured EmptyState for API failures.
 * Import directly instead of configuring EmptyState manually.
 */
export function NetworkError({
  onRetry,
  isRetrying = false,
  message,
  className = '',
}: NetworkErrorProps) {
  const { t } = useTranslation();
  const displayMessage = message ?? t('components.empty_state.default_network_error', 'Could not connect to the server. Please check your internet connection and try again.');
  // Dynamic import to avoid circular deps — use inline SVG for error icon
  const ErrorIcon = (
    <svg
      width="48"
      height="48"
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <circle cx="24" cy="24" r="20" fill="#FFCDD2" />
      <path
        d="M24 14v12M24 32v2"
        stroke="#C62828"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
    </svg>
  );

  return (
    <EmptyState
      icon={ErrorIcon}
      title={t('components.empty_state.something_went_wrong', 'Something went wrong')}
      description={displayMessage}
      actionLabel={isRetrying ? t('components.empty_state.trying_again', 'Trying again…') : t('components.empty_state.try_again', 'Try Again')}
      onAction={onRetry}
      isLoading={isRetrying}
      variant="error"
      className={className}
    />
  );
}

// ─────────────────────────────────────────────────────────────
// Preset: No search results
// ─────────────────────────────────────────────────────────────
interface NoResultsProps {
  query?: string;
  onClear?: () => void;
  className?: string;
}

/**
 * NoResults — pre-configured EmptyState for empty search results.
 */
export function NoResults({ query, onClear, className = '' }: NoResultsProps) {
  const { t } = useTranslation();
  const SearchIcon = (
    <svg
      width="48"
      height="48"
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <circle cx="21" cy="21" r="14" stroke="#C8E6C9" strokeWidth="3" fill="#E8F5E9" />
      <path
        d="M31 31l8 8"
        stroke="#C8E6C9"
        strokeWidth="3"
        strokeLinecap="round"
      />
      <path
        d="M17 21h8M21 17v8"
        stroke="#2E7D32"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );

  const descriptionText = query
    ? t('components.empty_state.no_search_match', {
        query,
        defaultValue: `We couldn't find anything for "${query}". Try a different crop name or district.`
      })
    : t('components.empty_state.no_filters_match', 'No items match your search. Try adjusting your filters.');

  return (
    <EmptyState
      icon={SearchIcon}
      title={t('components.empty_state.no_results', 'No results found')}
      description={descriptionText}
      actionLabel={onClear ? t('components.empty_state.clear_search', 'Clear Search') : undefined}
      onAction={onClear}
      variant="default"
      className={className}
    />
  );
}

export default EmptyState;
