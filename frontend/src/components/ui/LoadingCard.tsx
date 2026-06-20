// src/components/ui/LoadingCard.tsx
import { motion } from 'framer-motion';

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────
interface LoadingCardProps {
  /**
   * Tailwind height utility class, e.g. 'h-32' or 'h-48'.
   * Defaults to 'h-32'.
   */
  height?: string;
  /**
   * Additional Tailwind classes for the card container.
   */
  className?: string;
  /**
   * Whether to show an internal content structure (lines + icon area).
   * Defaults to true.
   */
  showStructure?: boolean;
}

// ─────────────────────────────────────────────────────────────
// Single shimmer card
// ─────────────────────────────────────────────────────────────
/**
 * LoadingCard — a shimmer placeholder card.
 * Uses the `.shimmer` CSS animation defined in index.css.
 */
export function LoadingCard({
  height = 'h-32',
  className = '',
  showStructure = true,
}: LoadingCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className={`bg-white rounded-md overflow-hidden shadow-card ${height} ${className}`}
      aria-busy="true"
      aria-label="Loading content"
      role="status"
    >
      {showStructure ? (
        <div className="p-4 h-full flex flex-col gap-3">
          {/* Top row: icon stub + title stub */}
          <div className="flex items-center gap-3">
            <div className="shimmer w-10 h-10 rounded-md shrink-0" />
            <div className="flex-1 flex flex-col gap-2">
              <div className="shimmer h-3.5 rounded-full w-3/5" />
              <div className="shimmer h-2.5 rounded-full w-2/5" />
            </div>
          </div>
          {/* Content lines */}
          <div className="flex-1 flex flex-col justify-end gap-2">
            <div className="shimmer h-2.5 rounded-full w-full" />
            <div className="shimmer h-2.5 rounded-full w-4/5" />
            <div className="shimmer h-2.5 rounded-full w-3/5" />
          </div>
        </div>
      ) : (
        <div className="shimmer w-full h-full" />
      )}
    </motion.div>
  );
}

// ─────────────────────────────────────────────────────────────
// Slim inline loading bar (single-line skeleton)
// ─────────────────────────────────────────────────────────────
interface LoadingLineProps {
  width?: string;
  className?: string;
}

export function LoadingLine({ width = 'w-full', className = '' }: LoadingLineProps) {
  return (
    <div
      className={`shimmer h-3 rounded-full ${width} ${className}`}
      aria-hidden="true"
    />
  );
}

// ─────────────────────────────────────────────────────────────
// Grid of 4 skeleton cards (2 columns)
// ─────────────────────────────────────────────────────────────
interface LoadingGridProps {
  /** Number of skeleton cards to show. Defaults to 4. */
  count?: number;
  /** Tailwind height for each card. Defaults to 'h-40'. */
  cardHeight?: string;
  /** Additional class on the grid container. */
  className?: string;
}

/**
 * LoadingGrid — renders a responsive 2-column grid of shimmer cards.
 * Ideal for dashboard card sections.
 */
export function LoadingGrid({
  count = 4,
  cardHeight = 'h-40',
  className = '',
}: LoadingGridProps) {
  return (
    <div className={`grid grid-cols-2 gap-3 lg:gap-4 ${className}`}>
      {Array.from({ length: count }).map((_, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.06, duration: 0.3 }}
        >
          <LoadingCard height={cardHeight} />
        </motion.div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// List of 3 stacked skeleton cards (full width)
// ─────────────────────────────────────────────────────────────
interface LoadingListProps {
  /** Number of skeleton rows to show. Defaults to 3. */
  count?: number;
  /** Tailwind height for each row. Defaults to 'h-20'. */
  rowHeight?: string;
  /** Additional class on the list container. */
  className?: string;
}

/**
 * LoadingList — renders a vertical stack of shimmer rows.
 * Ideal for price lists, chat history, or scheme lists.
 */
export function LoadingList({
  count = 3,
  rowHeight = 'h-20',
  className = '',
}: LoadingListProps) {
  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      {Array.from({ length: count }).map((_, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, x: -12 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.07, duration: 0.3 }}
        >
          <LoadingCard height={rowHeight} showStructure />
        </motion.div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Full-page loading state (for initial data fetches)
// ─────────────────────────────────────────────────────────────
interface LoadingPageProps {
  /** Optional label shown below the spinner. */
  label?: string;
}

/**
 * LoadingPage — full-screen centred spinner with a friendly message.
 */
export function LoadingPage({ label = 'Loading…' }: LoadingPageProps) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
      {/* Spinning leaf */}
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1.2, repeat: Infinity, }}
        className="w-12 h-12 rounded-full border-4 border-surface-variant border-t-primary"
      />
      <p className="text-sm font-noto text-text-secondary">{label}</p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Chat skeleton (alternating left/right bubbles)
// ─────────────────────────────────────────────────────────────
/**
 * LoadingChat — shimmer chat bubble placeholders.
 */
export function LoadingChat({ count = 4 }: { count?: number }) {
  return (
    <div className="flex flex-col gap-4 px-4 py-3">
      {Array.from({ length: count }).map((_, i) => {
        const isUser = i % 2 === 1;
        return (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: isUser ? 20 : -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.08, duration: 0.3 }}
            className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
          >
            {!isUser && (
              <div className="shimmer w-8 h-8 rounded-full mr-2 shrink-0 self-end" />
            )}
            <div
              className={`shimmer rounded-lg rounded-${isUser ? 'tr' : 'tl'}-none h-12
                         ${isUser ? 'w-48' : 'w-56'}`}
            />
          </motion.div>
        );
      })}
    </div>
  );
}

export default LoadingCard;
