// src/components/ui/PageWrapper.tsx
import { motion } from 'framer-motion';

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────
interface PageWrapperProps {
  /** The page content to render */
  children: React.ReactNode;
  /** Optional extra Tailwind classes on the wrapper div */
  className?: string;
  noPadding?: boolean;
}

// ─────────────────────────────────────────────────────────────
// Animation variants
// ─────────────────────────────────────────────────────────────
const pageVariants = {
  initial: {
    opacity: 0,
    y: 16,
  },
  animate: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.35,
      
    },
  },
  exit: {
    opacity: 0,
    y: -8,
    transition: {
      duration: 0.2,
      
    },
  },
};

// ─────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────
/**
 * PageWrapper
 *
 * Wraps every protected page with a Framer Motion fade+slide transition.
 * Place this as the outermost element inside every page component.
 *
 * Usage:
 * ```tsx
 * export default function DashboardPage() {
 *   return (
 *     <PageWrapper>
 *       <div>...</div>
 *     </PageWrapper>
 *   );
 * }
 * ```
 */
export function PageWrapper({ children, className = '' }: PageWrapperProps) {
  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className={`min-h-screen ${className}`}
    >
      {children}
    </motion.div>
  );
}

// ─────────────────────────────────────────────────────────────
// Staggered section variant (for use inside pages)
// ─────────────────────────────────────────────────────────────
export const sectionVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: i * 0.08,
      duration: 0.4,
      
    },
  }),
};

/**
 * AnimatedSection — a motion.div that fades/slides in with an optional stagger index.
 * Use `custom={index}` to set the stagger delay.
 *
 * Usage:
 * ```tsx
 * <AnimatedSection custom={0}>First card</AnimatedSection>
 * <AnimatedSection custom={1}>Second card</AnimatedSection>
 * ```
 */
interface AnimatedSectionProps {
  children: React.ReactNode;
  className?: string;
  custom?: number;
}

export function AnimatedSection({ children, className = '', custom = 0 }: AnimatedSectionProps) {
  return (
    <motion.div
      variants={sectionVariants}
      initial="hidden"
      animate="visible"
      custom={custom}
      className={className}
    >
      {children}
    </motion.div>
  );
}

export default PageWrapper;
