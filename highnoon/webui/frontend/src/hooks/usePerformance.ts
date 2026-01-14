// HighNoon Dashboard - Performance Hooks
// Utility hooks for throttling, debouncing, and virtual scrolling

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';

// =============================================================================
// USE THROTTLED VALUE
// =============================================================================

/**
 * Throttle a value update to at most once per interval.
 * Useful for expensive re-renders triggered by rapidly changing values.
 */
export function useThrottledValue<T>(value: T, intervalMs: number = 100): T {
    const [throttledValue, setThrottledValue] = useState<T>(value);
    const lastUpdate = useRef<number>(0);
    const pendingValue = useRef<T>(value);
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        pendingValue.current = value;
        const now = Date.now();
        const elapsed = now - lastUpdate.current;

        if (elapsed >= intervalMs) {
            // Enough time has passed, update immediately
            setThrottledValue(value);
            lastUpdate.current = now;
        } else {
            // Schedule an update
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
            }
            timeoutRef.current = setTimeout(() => {
                setThrottledValue(pendingValue.current);
                lastUpdate.current = Date.now();
            }, intervalMs - elapsed);
        }

        return () => {
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
            }
        };
    }, [value, intervalMs]);

    return throttledValue;
}

// =============================================================================
// USE DEBOUNCED VALUE
// =============================================================================

/**
 * Debounce a value update - only update after the value stops changing.
 * Useful for search inputs or form validation.
 */
export function useDebouncedValue<T>(value: T, delayMs: number = 300): T {
    const [debouncedValue, setDebouncedValue] = useState<T>(value);

    useEffect(() => {
        const handler = setTimeout(() => {
            setDebouncedValue(value);
        }, delayMs);

        return () => {
            clearTimeout(handler);
        };
    }, [value, delayMs]);

    return debouncedValue;
}

// =============================================================================
// USE DEBOUNCED CALLBACK
// =============================================================================

/**
 * Create a debounced version of a callback function.
 */
export function useDebouncedCallback<T extends (...args: any[]) => any>(
    callback: T,
    delayMs: number = 300
): (...args: Parameters<T>) => void {
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const callbackRef = useRef(callback);

    // Update callback ref on each render
    callbackRef.current = callback;

    return useCallback((...args: Parameters<T>) => {
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
        }
        timeoutRef.current = setTimeout(() => {
            callbackRef.current(...args);
        }, delayMs);
    }, [delayMs]);
}

// =============================================================================
// USE VIRTUAL LIST
// =============================================================================

interface VirtualListOptions {
    /** Total number of items */
    itemCount: number;
    /** Height of each item in pixels */
    itemHeight: number;
    /** Number of extra items to render above/below viewport */
    overscan?: number;
    /** Container height (optional - uses ref measurement if not provided) */
    containerHeight?: number;
}

interface VirtualListResult {
    /** Ref to attach to the scrollable container */
    containerRef: React.RefObject<HTMLDivElement | null>;
    /** Ref to attach to the content wrapper */
    contentRef: React.RefObject<HTMLDivElement | null>;
    /** Start index of visible items */
    startIndex: number;
    /** End index of visible items (exclusive) */
    endIndex: number;
    /** Total height of all items */
    totalHeight: number;
    /** Offset for the first visible item */
    offsetY: number;
    /** Array of indices to render */
    visibleIndices: number[];
}

/**
 * Basic virtual scrolling for large lists.
 * Renders only the visible items plus overscan for smooth scrolling.
 */
export function useVirtualList({
    itemCount,
    itemHeight,
    overscan = 3,
    containerHeight: providedHeight,
}: VirtualListOptions): VirtualListResult {
    const containerRef = useRef<HTMLDivElement>(null);
    const contentRef = useRef<HTMLDivElement>(null);
    const [scrollTop, setScrollTop] = useState(0);
    const [measuredHeight, setMeasuredHeight] = useState(0);

    // Measure container height
    useEffect(() => {
        if (!containerRef.current) return;

        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                setMeasuredHeight(entry.contentRect.height);
            }
        });

        resizeObserver.observe(containerRef.current);
        return () => resizeObserver.disconnect();
    }, []);

    // Handle scroll events
    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        const handleScroll = () => {
            setScrollTop(container.scrollTop);
        };

        container.addEventListener('scroll', handleScroll, { passive: true });
        return () => container.removeEventListener('scroll', handleScroll);
    }, []);

    const containerHeight = providedHeight ?? measuredHeight;
    const totalHeight = itemCount * itemHeight;

    // Calculate visible range
    const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
    const visibleCount = Math.ceil(containerHeight / itemHeight) + 2 * overscan;
    const endIndex = Math.min(itemCount, startIndex + visibleCount);

    const offsetY = startIndex * itemHeight;

    const visibleIndices = useMemo(() => {
        const indices: number[] = [];
        for (let i = startIndex; i < endIndex; i++) {
            indices.push(i);
        }
        return indices;
    }, [startIndex, endIndex]);

    return {
        containerRef,
        contentRef,
        startIndex,
        endIndex,
        totalHeight,
        offsetY,
        visibleIndices,
    };
}

// =============================================================================
// USE INTERSECTION OBSERVER
// =============================================================================

/**
 * Hook for lazy loading elements when they enter the viewport.
 */
export function useIntersectionObserver(
    options: IntersectionObserverInit = {}
): [React.RefObject<HTMLDivElement | null>, boolean] {
    const ref = useRef<HTMLDivElement | null>(null);
    const [isIntersecting, setIsIntersecting] = useState(false);

    useEffect(() => {
        const element = ref.current;
        if (!element) return;

        const observer = new IntersectionObserver(([entry]) => {
            setIsIntersecting(entry.isIntersecting);
        }, options);

        observer.observe(element);
        return () => observer.disconnect();
    }, [options.root, options.rootMargin, options.threshold]);

    return [ref, isIntersecting];
}

// =============================================================================
// USE LOADING STATE
// =============================================================================

interface LoadingState {
    isLoading: boolean;
    error: Error | null;
    startLoading: () => void;
    stopLoading: (error?: Error) => void;
    reset: () => void;
}

/**
 * Hook for managing loading/error states in async operations.
 */
export function useLoadingState(initialLoading = false): LoadingState {
    const [isLoading, setIsLoading] = useState(initialLoading);
    const [error, setError] = useState<Error | null>(null);

    const startLoading = useCallback(() => {
        setIsLoading(true);
        setError(null);
    }, []);

    const stopLoading = useCallback((err?: Error) => {
        setIsLoading(false);
        if (err) setError(err);
    }, []);

    const reset = useCallback(() => {
        setIsLoading(false);
        setError(null);
    }, []);

    return { isLoading, error, startLoading, stopLoading, reset };
}

// =============================================================================
// USE ASYNC ACTION
// =============================================================================

/**
 * Wrap an async function with loading state management.
 * Returns a wrapped function and the loading state.
 */
export function useAsyncAction<T extends (...args: any[]) => Promise<any>>(
    action: T
): [(...args: Parameters<T>) => Promise<ReturnType<T>>, LoadingState] {
    const state = useLoadingState();
    const actionRef = useRef(action);
    actionRef.current = action;

    const wrappedAction = useCallback(async (...args: Parameters<T>): Promise<ReturnType<T>> => {
        state.startLoading();
        try {
            const result = await actionRef.current(...args);
            state.stopLoading();
            return result;
        } catch (err) {
            state.stopLoading(err instanceof Error ? err : new Error(String(err)));
            throw err;
        }
    }, []);

    return [wrappedAction, state];
}

// =============================================================================
// ADAPTIVE QUALITY SYSTEM
// =============================================================================

export type QualityTier = 'HIGH' | 'MEDIUM' | 'LOW';

interface AdaptiveQualitySettings {
    /** Current quality tier */
    tier: QualityTier;
    /** Current FPS (frames per second) */
    fps: number;
    /** Average frame time in milliseconds */
    frameTimeMs: number;
    /** Recommended particle count multiplier */
    particleMultiplier: number;
    /** Whether post-processing should be enabled */
    enablePostProcessing: boolean;
    /** Whether to use high-quality geometries */
    useHighQualityGeometry: boolean;
    /** Shadow map resolution */
    shadowMapResolution: number;
}

interface AdaptiveQualityOptions {
    /** Target FPS (default: 60) */
    targetFps?: number;
    /** FPS threshold for downgrade to MEDIUM (default: 45) */
    mediumThreshold?: number;
    /** FPS threshold for downgrade to LOW (default: 30) */
    lowThreshold?: number;
    /** Whether to auto-adjust quality (default: true) */
    autoAdjust?: boolean;
    /** Sample window size for averaging (default: 60 frames) */
    sampleWindow?: number;
}

/**
 * Adaptive quality system for 3D visualizations.
 * Monitors frame time and automatically adjusts quality settings.
 *
 * Features:
 * - Real-time FPS monitoring using performance.now()
 * - Three quality tiers: HIGH, MEDIUM, LOW
 * - Auto-adjusts particle counts and post-processing
 * - Smoothed frame time averaging to prevent jitter
 */
export function useAdaptiveQuality({
    targetFps = 60,
    mediumThreshold = 45,
    lowThreshold = 30,
    autoAdjust = true,
    sampleWindow = 60,
}: AdaptiveQualityOptions = {}): AdaptiveQualitySettings {
    const [tier, setTier] = useState<QualityTier>('HIGH');
    const [fps, setFps] = useState(60);
    const [frameTimeMs, setFrameTimeMs] = useState(16.67);

    const frameTimesRef = useRef<number[]>([]);
    const lastFrameTimeRef = useRef(performance.now());
    const tierLockTimeRef = useRef(0);

    // Calculate settings based on tier
    const settings = useMemo((): AdaptiveQualitySettings => {
        switch (tier) {
            case 'HIGH':
                return {
                    tier,
                    fps,
                    frameTimeMs,
                    particleMultiplier: 1.0,
                    enablePostProcessing: true,
                    useHighQualityGeometry: true,
                    shadowMapResolution: 2048,
                };
            case 'MEDIUM':
                return {
                    tier,
                    fps,
                    frameTimeMs,
                    particleMultiplier: 0.5,
                    enablePostProcessing: true,
                    useHighQualityGeometry: false,
                    shadowMapResolution: 1024,
                };
            case 'LOW':
                return {
                    tier,
                    fps,
                    frameTimeMs,
                    particleMultiplier: 0.25,
                    enablePostProcessing: false,
                    useHighQualityGeometry: false,
                    shadowMapResolution: 512,
                };
        }
    }, [tier, fps, frameTimeMs]);

    // Frame time monitoring effect
    useEffect(() => {
        let animationFrameId: number;

        const measureFrame = () => {
            const now = performance.now();
            const deltaMs = now - lastFrameTimeRef.current;
            lastFrameTimeRef.current = now;

            // Add to sample window
            frameTimesRef.current.push(deltaMs);
            if (frameTimesRef.current.length > sampleWindow) {
                frameTimesRef.current.shift();
            }

            // Calculate average
            const avgFrameTime = frameTimesRef.current.reduce((a, b) => a + b, 0)
                / frameTimesRef.current.length;
            const currentFps = 1000 / avgFrameTime;

            setFrameTimeMs(avgFrameTime);
            setFps(Math.round(currentFps));

            // Auto-adjust quality (with hysteresis to prevent rapid switching)
            if (autoAdjust && now - tierLockTimeRef.current > 2000) {
                const newTier =
                    currentFps >= mediumThreshold ? 'HIGH' :
                        currentFps >= lowThreshold ? 'MEDIUM' : 'LOW';

                if (newTier !== tier) {
                    setTier(newTier);
                    tierLockTimeRef.current = now;
                }
            }

            animationFrameId = requestAnimationFrame(measureFrame);
        };

        animationFrameId = requestAnimationFrame(measureFrame);

        return () => {
            cancelAnimationFrame(animationFrameId);
        };
    }, [autoAdjust, mediumThreshold, lowThreshold, sampleWindow, tier]);

    return settings;
}

// =============================================================================
// USE 3D PERFORMANCE MONITOR
// =============================================================================

interface Performance3DMetrics {
    /** Current quality settings */
    quality: AdaptiveQualitySettings;
    /** GPU memory usage estimate (if available) */
    gpuMemoryMB: number | null;
    /** Draw call count estimate */
    drawCalls: number;
    /** Triangle count estimate */
    triangles: number;
    /** Whether performance is constrained */
    isConstrained: boolean;
    /** Force a specific quality tier */
    setQualityTier: (tier: QualityTier) => void;
}

interface Performance3DOptions extends AdaptiveQualityOptions {
    /** GL context for advanced metrics (optional) */
    gl?: WebGL2RenderingContext | WebGLRenderingContext | null;
}

/**
 * Comprehensive 3D performance monitoring hook.
 * Extends adaptive quality with additional GPU metrics.
 */
export function use3DPerformance({
    gl,
    ...qualityOptions
}: Performance3DOptions = {}): Performance3DMetrics {
    const quality = useAdaptiveQuality(qualityOptions);
    const [gpuMemoryMB, setGpuMemoryMB] = useState<number | null>(null);
    const [drawCalls, setDrawCalls] = useState(0);
    const [triangles, setTriangles] = useState(0);
    const [forcedTier, setForcedTier] = useState<QualityTier | null>(null);

    // Try to get GPU memory info (Chrome-only extension)
    useEffect(() => {
        if (gl) {
            const ext = gl.getExtension('WEBGL_debug_renderer_info');
            if (ext) {
                // Memory info is not standard, but some browsers expose it
                const debugInfo = {
                    vendor: gl.getParameter(ext.UNMASKED_VENDOR_WEBGL),
                    renderer: gl.getParameter(ext.UNMASKED_RENDERER_WEBGL),
                };
                // Log GPU info for debugging
                console.debug('[3D Performance] GPU:', debugInfo.renderer);
            }
        }
    }, [gl]);

    // Derive final quality (respect forced tier if set)
    const effectiveQuality = useMemo((): AdaptiveQualitySettings => {
        if (forcedTier) {
            return {
                ...quality,
                tier: forcedTier,
                particleMultiplier: forcedTier === 'HIGH' ? 1.0 : forcedTier === 'MEDIUM' ? 0.5 : 0.25,
                enablePostProcessing: forcedTier !== 'LOW',
                useHighQualityGeometry: forcedTier === 'HIGH',
            };
        }
        return quality;
    }, [quality, forcedTier]);

    const setQualityTier = useCallback((tier: QualityTier) => {
        setForcedTier(tier);
    }, []);

    return {
        quality: effectiveQuality,
        gpuMemoryMB,
        drawCalls,
        triangles,
        isConstrained: quality.tier !== 'HIGH',
        setQualityTier,
    };
}
