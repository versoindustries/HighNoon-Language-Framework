// performance.tsx - Performance optimization utilities
// Code splitting, lazy loading, and performance monitoring

import React, { lazy, Suspense } from 'react';
import type { ComponentType, ReactNode, ComponentProps } from 'react';

/**
 * Create a lazily loaded component with a loading fallback
 *
 * @example
 * const HPO = lazyWithFallback(() => import('./pages/HPO'), <LoadingSpinner />);
 */
export function lazyWithFallback<T extends ComponentType<any>>(
    importFn: () => Promise<{ default: T }>,
    fallback: ReactNode = null
) {
    const LazyComponent = lazy(importFn);

    return function LazyWithFallback(props: ComponentProps<T>) {
        return (
            <Suspense fallback={fallback}>
                <LazyComponent {...props} />
            </Suspense>
        );
    };
}

/**
 * Preload a lazy component before it's needed
 * Call this on hover or when you anticipate navigation
 *
 * @example
 * onMouseEnter={() => preloadComponent(() => import('./pages/HPO'))}
 */
export function preloadComponent(importFn: () => Promise<unknown>): void {
    // Fire and forget - starts loading the chunk
    importFn().catch(() => {
        // Silently fail - component will be loaded when actually needed
    });
}

/**
 * Measure component render performance
 */
export function measureRender(componentName: string): () => void {
    if (typeof performance === 'undefined') {
        return () => { };
    }

    const startTime = performance.now();
    const markName = `render-start-${componentName}`;
    performance.mark(markName);

    return () => {
        const endTime = performance.now();
        const duration = endTime - startTime;

        // Log slow renders in development
        if (import.meta.env.DEV && duration > 16) {
            console.warn(`[Performance] Slow render: ${componentName} took ${duration.toFixed(2)}ms`);
        }

        // Record performance entry
        try {
            performance.measure(`render-${componentName}`, markName);
        } catch {
            // Cleanup mark if measure fails
        }
    };
}

/**
 * Debounced resize handler for responsive components
 */
export function createResizeObserver(
    callback: (entry: ResizeObserverEntry) => void,
    debounceMs = 100
): ResizeObserver {
    let timeoutId: number | null = null;

    return new ResizeObserver((entries) => {
        if (timeoutId) {
            cancelAnimationFrame(timeoutId);
        }

        timeoutId = requestAnimationFrame(() => {
            for (const entry of entries) {
                callback(entry);
            }
        });
    });
}

/**
 * Request idle callback with fallback for Safari
 */
export function requestIdleCallbackPolyfill(
    callback: () => void,
    options?: { timeout?: number }
): number {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const g = window as any;
    if (typeof g.requestIdleCallback !== 'undefined') {
        return g.requestIdleCallback(callback, options);
    }

    return window.setTimeout(() => {
        callback();
    }, 1) as unknown as number;
}

/**
 * Cancel idle callback with fallback
 */
export function cancelIdleCallbackPolyfill(id: number): void {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const g = window as any;
    if (typeof g.cancelIdleCallback !== 'undefined') {
        g.cancelIdleCallback(id);
    } else {
        clearTimeout(id);
    }
}

/**
 * Memory-efficient batch processing
 * Process items in small batches to avoid blocking the main thread
 */
export async function processBatch<T, R>(
    items: T[],
    processor: (item: T) => R,
    batchSize = 50,
    delayBetweenBatches = 0
): Promise<R[]> {
    const results: R[] = [];

    for (let i = 0; i < items.length; i += batchSize) {
        const batch = items.slice(i, i + batchSize);
        const batchResults = batch.map(processor);
        results.push(...batchResults);

        if (delayBetweenBatches > 0 && i + batchSize < items.length) {
            await new Promise(resolve => setTimeout(resolve, delayBetweenBatches));
        }
    }

    return results;
}

/**
 * Image lazy loading with IntersectionObserver
 */
export function createImageObserver(
    loadCallback: (img: HTMLImageElement) => void,
    rootMargin = '50px'
): IntersectionObserver {
    return new IntersectionObserver(
        (entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target as HTMLImageElement;
                    loadCallback(img);
                    observer.unobserve(img);
                }
            });
        },
        {
            rootMargin,
            threshold: 0.1,
        }
    );
}

/**
 * Detect if the user prefers reduced motion
 */
export function prefersReducedMotion(): boolean {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Detect if the device is low-powered
 */
export function isLowPowerDevice(): boolean {
    if (typeof navigator === 'undefined') return false;

    // Check for low memory
    const memory = (navigator as unknown as { deviceMemory?: number }).deviceMemory;
    if (memory && memory < 4) return true;

    // Check for low CPU cores
    if (navigator.hardwareConcurrency && navigator.hardwareConcurrency < 4) return true;

    return false;
}

/**
 * Get optimal animation frame rate based on device capabilities
 */
export function getOptimalFrameRate(): number {
    if (prefersReducedMotion()) return 30;
    if (isLowPowerDevice()) return 30;
    return 60;
}

export default {
    lazyWithFallback,
    preloadComponent,
    measureRender,
    createResizeObserver,
    requestIdleCallbackPolyfill,
    cancelIdleCallbackPolyfill,
    processBatch,
    createImageObserver,
    prefersReducedMotion,
    isLowPowerDevice,
    getOptimalFrameRate,
};
