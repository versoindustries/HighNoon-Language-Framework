// useAnnouncer.ts - Screen reader announcement hook for accessibility
// Provides live region announcements for dynamic content changes

import { useCallback, useRef, useEffect } from 'react';

type Politeness = 'polite' | 'assertive';

interface AnnouncerOptions {
    /** Politeness level - 'assertive' for urgent, 'polite' for non-urgent */
    politeness?: Politeness;
    /** Clear previous announcement before new one */
    clearPrevious?: boolean;
    /** Delay before announcement (ms) */
    delay?: number;
}

/**
 * useAnnouncer - Hook for screen reader announcements
 *
 * Creates a live region and provides methods to announce messages
 * to screen reader users. Essential for dynamic content updates.
 *
 * @example
 * const { announce } = useAnnouncer();
 * announce('Training completed successfully!', { politeness: 'assertive' });
 */
export function useAnnouncer() {
    const politeRef = useRef<HTMLDivElement | null>(null);
    const assertiveRef = useRef<HTMLDivElement | null>(null);
    const timeoutRef = useRef<number | null>(null);

    // Create live regions on mount
    useEffect(() => {
        // Check if already created
        const existingPolite = document.getElementById('sr-announcer-polite');
        const existingAssertive = document.getElementById('sr-announcer-assertive');

        if (!existingPolite) {
            const politeRegion = document.createElement('div');
            politeRegion.id = 'sr-announcer-polite';
            politeRegion.setAttribute('role', 'status');
            politeRegion.setAttribute('aria-live', 'polite');
            politeRegion.setAttribute('aria-atomic', 'true');
            politeRegion.className = 'sr-only';
            document.body.appendChild(politeRegion);
            politeRef.current = politeRegion;
        } else {
            politeRef.current = existingPolite as HTMLDivElement;
        }

        if (!existingAssertive) {
            const assertiveRegion = document.createElement('div');
            assertiveRegion.id = 'sr-announcer-assertive';
            assertiveRegion.setAttribute('role', 'alert');
            assertiveRegion.setAttribute('aria-live', 'assertive');
            assertiveRegion.setAttribute('aria-atomic', 'true');
            assertiveRegion.className = 'sr-only';
            document.body.appendChild(assertiveRegion);
            assertiveRef.current = assertiveRegion;
        } else {
            assertiveRef.current = existingAssertive as HTMLDivElement;
        }

        return () => {
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
            }
        };
    }, []);

    // Announce a message
    const announce = useCallback((
        message: string,
        options: AnnouncerOptions = {}
    ) => {
        const {
            politeness = 'polite',
            clearPrevious = true,
            delay = 0,
        } = options;

        const region = politeness === 'assertive' ? assertiveRef.current : politeRef.current;
        if (!region) return;

        const doAnnounce = () => {
            if (clearPrevious) {
                region.textContent = '';
                // Small delay to ensure screen readers detect the change
                requestAnimationFrame(() => {
                    region.textContent = message;
                });
            } else {
                region.textContent = message;
            }
        };

        if (delay > 0) {
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
            }
            timeoutRef.current = window.setTimeout(doAnnounce, delay);
        } else {
            doAnnounce();
        }
    }, []);

    // Clear all announcements
    const clear = useCallback(() => {
        if (politeRef.current) politeRef.current.textContent = '';
        if (assertiveRef.current) assertiveRef.current.textContent = '';
    }, []);

    // Announce page navigation
    const announcePageChange = useCallback((pageName: string) => {
        announce(`Navigated to ${pageName} page`, { politeness: 'polite', delay: 100 });
    }, [announce]);

    // Announce loading state
    const announceLoading = useCallback((isLoading: boolean, context?: string) => {
        if (isLoading) {
            announce(`Loading ${context || 'content'}...`, { politeness: 'polite' });
        } else {
            announce(`${context || 'Content'} loaded`, { politeness: 'polite', delay: 100 });
        }
    }, [announce]);

    // Announce error
    const announceError = useCallback((message: string) => {
        announce(`Error: ${message}`, { politeness: 'assertive' });
    }, [announce]);

    // Announce success
    const announceSuccess = useCallback((message: string) => {
        announce(message, { politeness: 'polite' });
    }, [announce]);

    return {
        announce,
        clear,
        announcePageChange,
        announceLoading,
        announceError,
        announceSuccess,
    };
}

/**
 * Hook for focus management during route transitions
 */
export function useFocusManagement() {
    const previousFocusRef = useRef<HTMLElement | null>(null);

    // Store current focus before navigation
    const storeFocus = useCallback(() => {
        previousFocusRef.current = document.activeElement as HTMLElement;
    }, []);

    // Restore focus after navigation
    const restoreFocus = useCallback(() => {
        if (previousFocusRef.current && document.contains(previousFocusRef.current)) {
            previousFocusRef.current.focus();
        }
    }, []);

    // Focus main content after navigation
    const focusMainContent = useCallback(() => {
        const mainContent = document.getElementById('main-content');
        if (mainContent) {
            // Set tabindex temporarily to make it focusable
            mainContent.setAttribute('tabindex', '-1');
            mainContent.focus();
            // Reset focus to allow screen reader virtual cursor to move naturally
            setTimeout(() => {
                mainContent.removeAttribute('tabindex');
            }, 100);
        }
    }, []);

    // Focus first heading in main content
    const focusFirstHeading = useCallback(() => {
        const mainContent = document.getElementById('main-content');
        if (mainContent) {
            const heading = mainContent.querySelector('h1, h2, h3');
            if (heading instanceof HTMLElement) {
                heading.setAttribute('tabindex', '-1');
                heading.focus();
                setTimeout(() => {
                    heading.removeAttribute('tabindex');
                }, 100);
            }
        }
    }, []);

    return {
        storeFocus,
        restoreFocus,
        focusMainContent,
        focusFirstHeading,
    };
}

export default useAnnouncer;
