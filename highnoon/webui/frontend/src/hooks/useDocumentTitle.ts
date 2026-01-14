// useDocumentTitle.ts - Dynamic document title hook for training status
// Updates browser tab with real-time training information

import { useEffect, useRef, useCallback } from 'react';

interface DocumentTitleOptions {
    /** Base application title */
    appName?: string;
    /** Separator between sections */
    separator?: string;
    /** Restore original title on unmount */
    restoreOnUnmount?: boolean;
}

interface TrainingStatus {
    /** Current training step */
    step?: number;
    /** Total steps (optional) */
    totalSteps?: number;
    /** Current loss value */
    loss?: number;
    /** Training phase */
    phase?: 'warmup' | 'training' | 'validation' | 'completed' | 'paused' | 'error';
    /** Job/experiment name */
    jobName?: string;
    /** Custom suffix */
    suffix?: string;
}

/**
 * useDocumentTitle - Dynamic browser tab title hook
 *
 * Features:
 * - Updates browser tab with training status
 * - Shows step progress, loss value, and phase
 * - Supports custom formatting
 * - Restores original title on cleanup
 *
 * Example tab titles:
 * - "Step 1500 | Loss: 1.234 | HighNoon"
 * - "⚡ Training | my-experiment | HighNoon"
 * - "✓ Completed | 0.123 loss | HighNoon"
 */
export function useDocumentTitle({
    appName = 'HighNoon',
    separator = ' | ',
    restoreOnUnmount = true,
}: DocumentTitleOptions = {}) {
    const originalTitle = useRef(document.title);

    // Format loss value for display
    const formatLoss = useCallback((loss: number): string => {
        if (loss < 0.0001) return loss.toExponential(2);
        if (loss < 1) return loss.toFixed(4);
        if (loss < 100) return loss.toFixed(2);
        return loss.toFixed(0);
    }, []);

    // Get phase icon/prefix
    const getPhaseIndicator = useCallback((phase: TrainingStatus['phase']): string => {
        switch (phase) {
            case 'warmup': return '🔥';
            case 'training': return '⚡';
            case 'validation': return '📊';
            case 'completed': return '✓';
            case 'paused': return '⏸️';
            case 'error': return '❌';
            default: return '';
        }
    }, []);

    // Build title string from status
    const buildTitle = useCallback((status: TrainingStatus): string => {
        const parts: string[] = [];

        // Add phase indicator
        if (status.phase) {
            const indicator = getPhaseIndicator(status.phase);
            if (indicator) {
                parts.push(indicator);
            }
        }

        // Add step progress
        if (status.step !== undefined) {
            if (status.totalSteps) {
                const percent = Math.round((status.step / status.totalSteps) * 100);
                parts.push(`Step ${status.step} (${percent}%)`);
            } else {
                parts.push(`Step ${status.step}`);
            }
        }

        // Add loss
        if (status.loss !== undefined) {
            parts.push(`Loss: ${formatLoss(status.loss)}`);
        }

        // Add job name
        if (status.jobName) {
            parts.push(status.jobName);
        }

        // Add custom suffix
        if (status.suffix) {
            parts.push(status.suffix);
        }

        // Always add app name
        parts.push(appName);

        return parts.join(separator);
    }, [appName, separator, formatLoss, getPhaseIndicator]);

    // Update document title
    const setTitle = useCallback((status: TrainingStatus) => {
        document.title = buildTitle(status);
    }, [buildTitle]);

    // Set simple title
    const setSimpleTitle = useCallback((title: string) => {
        const parts = title ? [title, appName] : [appName];
        document.title = parts.join(separator);
    }, [appName, separator]);

    // Reset to app name only
    const resetTitle = useCallback(() => {
        document.title = appName;
    }, [appName]);

    // Restore original title on unmount
    useEffect(() => {
        return () => {
            if (restoreOnUnmount) {
                document.title = originalTitle.current;
            }
        };
    }, [restoreOnUnmount]);

    return {
        setTitle,
        setSimpleTitle,
        resetTitle,
        originalTitle: originalTitle.current,
    };
}

/**
 * Hook for simple static document title
 */
export function useStaticDocumentTitle(
    title: string,
    options: DocumentTitleOptions = {}
) {
    const { setSimpleTitle } = useDocumentTitle(options);

    useEffect(() => {
        setSimpleTitle(title);
    }, [title, setSimpleTitle]);
}

export default useDocumentTitle;
