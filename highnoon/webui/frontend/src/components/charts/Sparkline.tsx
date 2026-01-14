// HighNoon Dashboard - Sparkline Component
// Lightweight SVG sparkline for inline metric visualization

import { useMemo } from 'react';
import './Sparkline.css';

interface SparklineProps {
    /** Array of numeric values to plot */
    data: number[];
    /** Width of the sparkline in pixels */
    width?: number;
    /** Height of the sparkline in pixels */
    height?: number;
    /** Line color (defaults to primary) */
    color?: string;
    /** Whether to show area fill under the line */
    fill?: boolean;
    /** Whether a decrease is good (inverts color for trend) */
    decreaseIsGood?: boolean;
    /** Show min/max dots */
    showExtremes?: boolean;
    /** CSS class name */
    className?: string;
}

export function Sparkline({
    data,
    width = 80,
    height = 24,
    color,
    fill = true,
    decreaseIsGood = false,
    showExtremes = false,
    className = '',
}: SparklineProps) {
    // Compute path and trend
    const { pathD, fillD, trend, minPoint, maxPoint } = useMemo(() => {
        if (!data || data.length < 2) {
            return { pathD: '', fillD: '', trend: 'stable', minPoint: null, maxPoint: null };
        }

        const validData = data.filter(v => v !== null && v !== undefined && !isNaN(v));
        if (validData.length < 2) {
            return { pathD: '', fillD: '', trend: 'stable', minPoint: null, maxPoint: null };
        }

        const min = Math.min(...validData);
        const max = Math.max(...validData);
        const range = max - min || 1;

        // Padding for the chart
        const padX = 2;
        const padY = 2;
        const chartWidth = width - padX * 2;
        const chartHeight = height - padY * 2;

        // Generate points
        const points = validData.map((value, index) => {
            const x = padX + (index / (validData.length - 1)) * chartWidth;
            // Invert Y since SVG Y increases downward
            const y = padY + chartHeight - ((value - min) / range) * chartHeight;
            return { x, y, value };
        });

        // Create line path
        const pathD = points.map((p, i) =>
            i === 0 ? `M ${p.x.toFixed(2)} ${p.y.toFixed(2)}` : `L ${p.x.toFixed(2)} ${p.y.toFixed(2)}`
        ).join(' ');

        // Create fill path (close the area)
        const fillD = pathD +
            ` L ${points[points.length - 1].x.toFixed(2)} ${height - padY}` +
            ` L ${padX} ${height - padY} Z`;

        // Determine trend based on last few values
        const recentCount = Math.min(5, validData.length);
        const recent = validData.slice(-recentCount);
        const firstHalf = recent.slice(0, Math.floor(recentCount / 2));
        const secondHalf = recent.slice(Math.floor(recentCount / 2));
        const avgFirst = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
        const avgSecond = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

        const trendThreshold = range * 0.05; // 5% of range
        let trend: 'up' | 'down' | 'stable';
        if (avgSecond > avgFirst + trendThreshold) {
            trend = 'up';
        } else if (avgSecond < avgFirst - trendThreshold) {
            trend = 'down';
        } else {
            trend = 'stable';
        }

        // Find extreme points for highlighting
        const minIdx = validData.indexOf(min);
        const maxIdx = validData.indexOf(max);
        const minPoint = points[minIdx];
        const maxPoint = points[maxIdx];

        return { pathD, fillD, trend, minPoint, maxPoint };
    }, [data, width, height]);

    // Determine colors based on trend and whether decrease is good
    const trendColor = useMemo(() => {
        if (color) return color;

        if (trend === 'stable') return 'var(--color-info)';

        const isGood = (trend === 'down' && decreaseIsGood) || (trend === 'up' && !decreaseIsGood);
        return isGood ? 'var(--color-success)' : 'var(--color-warning)';
    }, [color, trend, decreaseIsGood]);

    if (!pathD) {
        return (
            <svg
                className={`sparkline sparkline--empty ${className}`}
                width={width}
                height={height}
                aria-hidden="true"
            >
                <line
                    x1={2}
                    y1={height / 2}
                    x2={width - 2}
                    y2={height / 2}
                    stroke="var(--text-muted)"
                    strokeWidth={1}
                    strokeDasharray="3,3"
                    opacity={0.3}
                />
            </svg>
        );
    }

    return (
        <svg
            className={`sparkline sparkline--${trend} ${className}`}
            width={width}
            height={height}
            aria-hidden="true"
        >
            {/* Gradient definition for fill */}
            <defs>
                <linearGradient id={`sparkline-grad-${trendColor.replace(/[^a-z0-9]/gi, '')}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={trendColor} stopOpacity={0.3} />
                    <stop offset="100%" stopColor={trendColor} stopOpacity={0} />
                </linearGradient>
            </defs>

            {/* Area fill */}
            {fill && (
                <path
                    d={fillD}
                    fill={`url(#sparkline-grad-${trendColor.replace(/[^a-z0-9]/gi, '')})`}
                    className="sparkline__fill"
                />
            )}

            {/* Line */}
            <path
                d={pathD}
                fill="none"
                stroke={trendColor}
                strokeWidth={1.5}
                strokeLinecap="round"
                strokeLinejoin="round"
                className="sparkline__line"
            />

            {/* Current value dot (last point) */}
            {data.length > 0 && (
                <circle
                    cx={width - 2}
                    cy={height / 2}
                    r={2}
                    fill={trendColor}
                    className="sparkline__current"
                />
            )}

            {/* Min/Max points */}
            {showExtremes && minPoint && (
                <circle
                    cx={minPoint.x}
                    cy={minPoint.y}
                    r={2}
                    fill="var(--color-success)"
                    className="sparkline__min"
                />
            )}
            {showExtremes && maxPoint && minPoint !== maxPoint && (
                <circle
                    cx={maxPoint.x}
                    cy={maxPoint.y}
                    r={2}
                    fill="var(--color-warning)"
                    className="sparkline__max"
                />
            )}
        </svg>
    );
}

// =============================================================================
// TREND INDICATOR COMPONENT
// =============================================================================

interface TrendIndicatorProps {
    /** Direction of the trend */
    direction: 'up' | 'down' | 'stable';
    /** Numeric change (optional) */
    delta?: number;
    /** Whether this direction is positive/good */
    isPositive?: boolean;
    /** Size of the indicator */
    size?: 'sm' | 'md';
    /** Format function for delta */
    formatDelta?: (value: number) => string;
}

export function TrendIndicator({
    direction,
    delta,
    isPositive = true,
    size = 'sm',
    formatDelta = (v) => v.toFixed(2),
}: TrendIndicatorProps) {
    // Determine visual style
    const isGood = (direction === 'up' && isPositive) || (direction === 'down' && !isPositive);
    const isBad = (direction === 'down' && isPositive) || (direction === 'up' && !isPositive);

    const className = [
        'trend-indicator',
        `trend-indicator--${direction}`,
        `trend-indicator--${size}`,
        isGood ? 'trend-indicator--positive' : '',
        isBad ? 'trend-indicator--negative' : '',
    ].filter(Boolean).join(' ');

    return (
        <span className={className}>
            {direction === 'up' && (
                <svg viewBox="0 0 12 12" className="trend-indicator__icon">
                    <path d="M6 2L10 8H2L6 2Z" fill="currentColor" />
                </svg>
            )}
            {direction === 'down' && (
                <svg viewBox="0 0 12 12" className="trend-indicator__icon">
                    <path d="M6 10L2 4H10L6 10Z" fill="currentColor" />
                </svg>
            )}
            {direction === 'stable' && (
                <svg viewBox="0 0 12 12" className="trend-indicator__icon">
                    <rect x="2" y="5" width="8" height="2" fill="currentColor" />
                </svg>
            )}
            {delta !== undefined && (
                <span className="trend-indicator__delta">
                    {delta > 0 ? '+' : ''}{formatDelta(delta)}
                </span>
            )}
        </span>
    );
}
