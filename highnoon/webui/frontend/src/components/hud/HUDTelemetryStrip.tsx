// HUDTelemetryStrip.tsx - Continuous strip chart for real-time metrics
// F1-style telemetry display showing multiple metric rows

import React, { useRef, useEffect, useMemo, useCallback } from 'react';
import './HUDTelemetryStrip.css';

interface MetricRow {
    id: string;
    label: string;
    values: number[];
    color: string;
    min?: number;
    max?: number;
    unit?: string;
    /** Current latest value */
    current?: number;
}

interface HUDTelemetryStripProps {
    /** Metric rows to display */
    metrics: MetricRow[];
    /** Time window in seconds */
    timeWindow?: number;
    /** Height per row in pixels */
    rowHeight?: number;
    /** Show value labels */
    showValues?: boolean;
    /** Show grid lines */
    showGrid?: boolean;
    /** Animation enabled */
    animated?: boolean;
    /** Container className */
    className?: string;
}

/**
 * HUDTelemetryStrip - F1-style horizontal strip chart
 *
 * Features:
 * - Multiple metric rows (loss, LR, gradient, VRAM)
 * - Scrolling time window display
 * - Canvas 2D for high performance
 * - Color-coded metrics with current value display
 * - Grid overlay for scale reference
 */
export function HUDTelemetryStrip({
    metrics,
    timeWindow = 300,
    rowHeight = 32,
    showValues = true,
    showGrid = true,
    animated = true,
    className = '',
}: HUDTelemetryStripProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const animationRef = useRef<number>(0);

    // Calculate dimensions
    const totalHeight = metrics.length * rowHeight;

    // Draw the telemetry strip
    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const width = container.clientWidth;
        const height = totalHeight;

        canvas.width = width * dpr;
        canvas.height = height * dpr;
        ctx.scale(dpr, dpr);

        // Clear canvas
        ctx.fillStyle = 'rgba(10, 10, 15, 0.95)';
        ctx.fillRect(0, 0, width, height);

        // Draw each metric row
        metrics.forEach((metric, rowIndex) => {
            const y = rowIndex * rowHeight;
            const values = metric.values;

            if (values.length === 0) return;

            // Calculate min/max for normalization
            const min = metric.min ?? Math.min(...values);
            const max = metric.max ?? Math.max(...values);
            const range = max - min || 1;

            // Draw row background gradient
            const bgGradient = ctx.createLinearGradient(0, y, 0, y + rowHeight);
            bgGradient.addColorStop(0, 'rgba(255, 255, 255, 0.02)');
            bgGradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.04)');
            bgGradient.addColorStop(1, 'rgba(255, 255, 255, 0.02)');
            ctx.fillStyle = bgGradient;
            ctx.fillRect(0, y, width, rowHeight);

            // Draw row separator
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, y + rowHeight);
            ctx.lineTo(width, y + rowHeight);
            ctx.stroke();

            // Draw grid lines if enabled
            if (showGrid) {
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
                ctx.setLineDash([2, 4]);

                // Horizontal mid-line
                ctx.beginPath();
                ctx.moveTo(0, y + rowHeight / 2);
                ctx.lineTo(width, y + rowHeight / 2);
                ctx.stroke();

                // Vertical time markers
                for (let i = 0; i < 5; i++) {
                    const x = (width / 5) * i;
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(x, y + rowHeight);
                    ctx.stroke();
                }

                ctx.setLineDash([]);
            }

            // Draw the metric line
            const pointWidth = width / Math.max(values.length - 1, 1);

            // Fill area under curve
            ctx.beginPath();
            ctx.moveTo(0, y + rowHeight);

            values.forEach((value, i) => {
                const x = i * pointWidth;
                const normalizedY = 1 - (value - min) / range;
                const py = y + normalizedY * (rowHeight - 4) + 2;

                if (i === 0) {
                    ctx.lineTo(x, py);
                } else {
                    ctx.lineTo(x, py);
                }
            });

            ctx.lineTo(width, y + rowHeight);
            ctx.closePath();

            const fillGradient = ctx.createLinearGradient(0, y, 0, y + rowHeight);
            const baseColor = metric.color;
            fillGradient.addColorStop(0, baseColor + '40');
            fillGradient.addColorStop(1, baseColor + '10');
            ctx.fillStyle = fillGradient;
            ctx.fill();

            // Draw the line itself
            ctx.beginPath();
            ctx.strokeStyle = metric.color;
            ctx.lineWidth = 2;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';

            values.forEach((value, i) => {
                const x = i * pointWidth;
                const normalizedY = 1 - (value - min) / range;
                const py = y + normalizedY * (rowHeight - 4) + 2;

                if (i === 0) {
                    ctx.moveTo(x, py);
                } else {
                    ctx.lineTo(x, py);
                }
            });

            ctx.stroke();

            // Draw glow effect on line
            ctx.strokeStyle = metric.color;
            ctx.lineWidth = 4;
            ctx.globalAlpha = 0.2;
            ctx.stroke();
            ctx.globalAlpha = 1;

            // Draw current value endpoint
            if (values.length > 0) {
                const lastValue = values[values.length - 1];
                const lastX = width;
                const lastNormalizedY = 1 - (lastValue - min) / range;
                const lastPy = y + lastNormalizedY * (rowHeight - 4) + 2;

                // Glow
                ctx.beginPath();
                ctx.arc(lastX - 2, lastPy, 6, 0, Math.PI * 2);
                ctx.fillStyle = metric.color + '40';
                ctx.fill();

                // Dot
                ctx.beginPath();
                ctx.arc(lastX - 2, lastPy, 3, 0, Math.PI * 2);
                ctx.fillStyle = metric.color;
                ctx.fill();
            }
        });

        if (animated) {
            animationRef.current = requestAnimationFrame(draw);
        }
    }, [metrics, totalHeight, rowHeight, showGrid, animated]);

    // Setup drawing loop
    useEffect(() => {
        draw();

        // Handle resize
        const handleResize = () => draw();
        window.addEventListener('resize', handleResize);

        return () => {
            cancelAnimationFrame(animationRef.current);
            window.removeEventListener('resize', handleResize);
        };
    }, [draw]);

    // Format current value for display
    const formatValue = useCallback((value: number | undefined, metric: MetricRow): string => {
        if (value === undefined) return '-';

        if (metric.id === 'lr') {
            return value.toExponential(1);
        }
        if (value >= 1000000) {
            return (value / 1000000).toFixed(1) + 'M';
        }
        if (value >= 1000) {
            return (value / 1000).toFixed(1) + 'K';
        }
        if (value < 1) {
            return value.toFixed(4);
        }
        return value.toFixed(2);
    }, []);

    return (
        <div className={`hud-telemetry-strip ${className}`} ref={containerRef}>
            <div className="hud-telemetry-strip__labels">
                {metrics.map((metric) => (
                    <div
                        key={metric.id}
                        className="hud-telemetry-strip__label"
                        style={{ height: rowHeight }}
                    >
                        <span className="hud-telemetry-strip__label-text">
                            {metric.label}
                        </span>
                        {showValues && (
                            <span
                                className="hud-telemetry-strip__value"
                                style={{ color: metric.color }}
                            >
                                {formatValue(metric.current ?? metric.values[metric.values.length - 1], metric)}
                                {metric.unit && <span className="hud-telemetry-strip__unit">{metric.unit}</span>}
                            </span>
                        )}
                    </div>
                ))}
            </div>

            <div className="hud-telemetry-strip__chart">
                <canvas ref={canvasRef} className="hud-telemetry-strip__canvas" />
            </div>

            <div className="hud-telemetry-strip__time-labels">
                <span>-5m</span>
                <span>-4m</span>
                <span>-3m</span>
                <span>-2m</span>
                <span>-1m</span>
                <span>Now</span>
            </div>
        </div>
    );
}

export default HUDTelemetryStrip;
