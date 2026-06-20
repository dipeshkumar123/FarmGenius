// src/components/ui/ConfidenceBar.tsx
import { motion } from 'framer-motion';

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────
interface ConfidenceBarProps {
  /**
   * Score from 0 to 1.
   * >= 0.85 → green (high confidence)
   * >= 0.65 → amber/yellow (medium confidence)
   * <  0.65 → red (low confidence — use with caution)
   */
  score: number;
  /**
   * Human-readable label shown above the bar.
   * e.g. "Confidence", "AI Confidence", "Diagnosis Confidence"
   */
  label?: string;
  /**
   * Whether to display the numeric percentage to the right of the bar.
   * Defaults to true.
   */
  showPercent?: boolean;
  /**
   * Whether to display a text confidence level description
   * (e.g., "High", "Medium", "Low"). Defaults to true.
   */
  showLevel?: boolean;
  /**
   * Optional animation delay in seconds. Useful for staggered reveals.
   * Defaults to 0.
   */
  animationDelay?: number;
  /**
   * Size variant for the bar height.
   * 'sm' = 4px, 'md' = 8px (default), 'lg' = 12px
   */
  size?: 'sm' | 'md' | 'lg';
  /** Additional Tailwind classes on the root container. */
  className?: string;
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

/** Returns Tailwind color classes based on score */
function getColorClasses(score: number): {
  fill: string;
  text: string;
  bg: string;
  levelLabel: string;
  dotColor: string;
} {
  if (score >= 0.85) {
    return {
      fill: 'bg-farm-success',
      text: 'text-farm-success',
      bg: 'bg-green-100',
      levelLabel: 'High',
      dotColor: 'bg-farm-success',
    };
  }
  if (score >= 0.65) {
    return {
      fill: 'bg-accent',
      text: 'text-amber-600',
      bg: 'bg-amber-100',
      levelLabel: 'Medium',
      dotColor: 'bg-accent',
    };
  }
  return {
    fill: 'bg-farm-error',
    text: 'text-farm-error',
    bg: 'bg-red-100',
    levelLabel: 'Low',
    dotColor: 'bg-farm-error',
  };
}

/** Returns bar height class based on size prop */
function getBarHeight(size: 'sm' | 'md' | 'lg'): string {
  switch (size) {
    case 'sm': return 'h-1.5';
    case 'lg': return 'h-3';
    default:   return 'h-2';
  }
}

// ─────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────
/**
 * ConfidenceBar — an animated horizontal progress bar indicating
 * AI confidence for disease detection or chatbot responses.
 *
 * Usage:
 * ```tsx
 * <ConfidenceBar score={0.92} label="Diagnosis Confidence" showPercent />
 * <ConfidenceBar score={0.71} label="AI Confidence" size="sm" />
 * <ConfidenceBar score={0.45} label="Match Score" showLevel={false} />
 * ```
 */
export function ConfidenceBar({
  score,
  label,
  showPercent = true,
  showLevel = true,
  animationDelay = 0,
  size = 'md',
  className = '',
}: ConfidenceBarProps) {
  // Clamp score to [0, 1]
  const clampedScore = Math.min(1, Math.max(0, score));
  const percent = Math.round(clampedScore * 100);
  const colors = getColorClasses(clampedScore);
  const barHeight = getBarHeight(size);

  return (
    <div className={`w-full ${className}`}>
      {/* Label row */}
      {(label || showPercent || showLevel) && (
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-2">
            {label && (
              <span className="text-xs font-poppins font-semibold text-text-secondary uppercase tracking-wide">
                {label}
              </span>
            )}
            {showLevel && (
              <span
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs
                           font-poppins font-semibold ${colors.bg} ${colors.text}`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${colors.dotColor}`} />
                {colors.levelLabel}
              </span>
            )}
          </div>
          {showPercent && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: animationDelay + 0.4, duration: 0.3 }}
              className={`text-sm font-poppins font-bold tabular-nums ${colors.text}`}
            >
              {percent}%
            </motion.span>
          )}
        </div>
      )}

      {/* Track */}
      <div
        className={`w-full ${barHeight} rounded-full bg-surface-variant overflow-hidden`}
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label ? `${label}: ${percent}%` : `${percent}% confidence`}
      >
        {/* Animated fill */}
        <motion.div
          className={`h-full rounded-full ${colors.fill} relative overflow-hidden`}
          initial={{ width: '0%' }}
          animate={{ width: `${clampedScore * 100}%` }}
          transition={{
            delay: animationDelay,
            duration: 0.8,
             // Spring-like overshoot
          }}
        >
          {/* Gloss shimmer sweep */}
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
            initial={{ x: '-100%' }}
            animate={{ x: '200%' }}
            transition={{
              delay: animationDelay + 0.8,
              duration: 0.6,
              
            }}
          />
        </motion.div>
      </div>

      {/* Farmer-friendly interpretation */}
      {clampedScore < 0.65 && (
        <motion.p
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: animationDelay + 0.5, duration: 0.3 }}
          className="mt-1.5 text-xs font-noto text-farm-error"
        >
          Low confidence — please retake the photo or consult your local KVK.
        </motion.p>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Compact inline badge variant (for chat responses)
// ─────────────────────────────────────────────────────────────
interface ConfidenceBadgeProps {
  score: number;
  className?: string;
}

/**
 * ConfidenceBadge — a compact pill badge showing confidence score.
 * Useful inside chat bubbles or small card corners.
 */
export function ConfidenceBadge({ score, className = '' }: ConfidenceBadgeProps) {
  const clampedScore = Math.min(1, Math.max(0, score));
  const percent = Math.round(clampedScore * 100);
  const colors = getColorClasses(clampedScore);

  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.2 }}
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs
                 font-poppins font-semibold ${colors.bg} ${colors.text} ${className}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${colors.dotColor}`} />
      {percent}% confident
    </motion.span>
  );
}

// ─────────────────────────────────────────────────────────────
// Multi-class confidence list (for disease detection results)
// ─────────────────────────────────────────────────────────────
interface ConfidenceResult {
  label: string;
  score: number;
}

interface ConfidenceListProps {
  /** Array of label+score pairs, sorted descending by score */
  results: ConfidenceResult[];
  /** Max number of results to display. Defaults to 3. */
  maxItems?: number;
  className?: string;
}

/**
 * ConfidenceList — shows a ranked list of detection results with bars.
 * Perfect for top-3 disease prediction results.
 */
export function ConfidenceList({
  results,
  maxItems = 3,
  className = '',
}: ConfidenceListProps) {
  const displayResults = results.slice(0, maxItems);

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      {displayResults.map((result, i) => (
        <motion.div
          key={result.label}
          initial={{ opacity: 0, x: -16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.1, duration: 0.35 }}
          className="flex flex-col gap-1"
        >
          <div className="flex items-center justify-between">
            <span className={`text-sm font-noto ${i === 0 ? 'font-semibold text-text-primary' : 'text-text-secondary'}`}>
              {result.label}
            </span>
            {i === 0 && (
              <span className="text-xs badge badge-green">Best match</span>
            )}
          </div>
          <ConfidenceBar
            score={result.score}
            showPercent
            showLevel={false}
            size="sm"
            animationDelay={i * 0.1 + 0.1}
          />
        </motion.div>
      ))}
    </div>
  );
}

export default ConfidenceBar;
