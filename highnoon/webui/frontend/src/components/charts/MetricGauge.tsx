// MetricGauge.tsx - Animated circular gauge for key metrics
// SVG-based gauge with smooth animations and color thresholds

import { useEffect, useRef, useState } from 'react';
import './MetricGauge.css';

interface MetricGaugeProps {
    value: number | null;
    label: string;
    unit?: string;
    min?: number;
    max?: number;
    thresholds?: {
        good: number;
        warning: number;
        danger: number;
    };
    size?: 'sm' | 'md' | 'lg';
    inverse?: boolean; // If true, lower is better (like loss/perplexity)
    formatValue?: (value: number) => string;
    icon?: React.ReactNode;
}

const SIZE_CONFIG = {
    sm: { diameter: 80, strokeWidth: 6, fontSize: 14 },
    md: { diameter: 120, strokeWidth: 8, fontSize: 18 },
    lg: { diameter: 160, strokeWidth: 10, fontSize: 24 },
};

export function MetricGauge({
    value,
    label,
    unit = '',
    min = 0,
    max = 100,
    thresholds = { good: 80, warning: 50, danger: 20 },
    size = 'md',
    inverse = false,
    formatValue,
    icon,
}: MetricGaugeProps) {
    const [animatedValue, setAnimatedValue] = useState(0);
    const animationRef = useRef<number | null>(null);
    const previousValueRef = useRef<number>(0);

    const config = SIZE_CONFIG[size];
    const radius = (config.diameter - config.strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const center = config.diameter / 2;

    // Animate value changes
    useEffect(() => {
        if (value === null) {
            setAnimatedValue(0);
            return;
        }

        const startValue = previousValueRef.current;
        const endValue = Math.max(min, Math.min(max, value));
        const duration = 600; // ms
        const startTime = Date.now();

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function (ease-out cubic)
            const eased = 1 - Math.pow(1 - progress, 3);

            const current = startValue + (endValue - startValue) * eased;
            setAnimatedValue(current);

            if (progress < 1) {
                animationRef.current = requestAnimationFrame(animate);
            } else {
                previousValueRef.current = endValue;
            }
        };

        animationRef.current = requestAnimationFrame(animate);

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [value, min, max]);

    // Calculate percentage
    const percentage = Math.max(0, Math.min(100, ((animatedValue - min) / (max - min)) * 100));
    const strokeDashoffset = circumference - (percentage / 100) * circumference;

    // Determine color based on thresholds
    const getColor = () => {
        if (value === null) return 'var(--text-muted)';

        const pct = ((animatedValue - min) / (max - min)) * 100;

        if (inverse) {
            // For inverse metrics (lower is better), flip the logic
            if (pct <= thresholds.danger) return 'var(--color-success)';
            if (pct <= thresholds.warning) return 'var(--color-warning)';
            return 'var(--color-danger)';
        } else {
            // Normal metrics (higher is better)
            if (pct >= thresholds.good) return 'var(--color-success)';
            if (pct >= thresholds.warning) return 'var(--color-warning)';
            return 'var(--color-danger)';
        }
    };

    const color = getColor();
    const displayValue = value !== null
        ? (formatValue ? formatValue(animatedValue) : animatedValue.toFixed(2))
        : '—';

    return (
        <div className={`metric-gauge metric-gauge--${size}`}>
            <div className="metric-gauge__ring">
                <svg
                    width={config.diameter}
                    height={config.diameter}
                    className="metric-gauge__svg"
                >
                    {/* Background track */}
                    <circle
                        className="metric-gauge__track"
                        cx={center}
                        cy={center}
                        r={radius}
                        strokeWidth={config.strokeWidth}
                    />
                    {/* Progress arc */}
                    <circle
                        className="metric-gauge__progress"
                        cx={center}
                        cy={center}
                        r={radius}
                        strokeWidth={config.strokeWidth}
                        strokeDasharray={circumference}
                        strokeDashoffset={strokeDashoffset}
                        style={{ stroke: color }}
                        transform={`rotate(-90 ${center} ${center})`}
                    />
                    {/* Glow effect for high values */}
                    {percentage > 70 && (
                        <circle
                            className="metric-gauge__glow"
                            cx={center}
                            cy={center}
                            r={radius}
                            strokeWidth={config.strokeWidth + 4}
                            strokeDasharray={circumference}
                            strokeDashoffset={strokeDashoffset}
                            style={{ stroke: color }}
                            transform={`rotate(-90 ${center} ${center})`}
                        />
                    )}
                </svg>
                <div className="metric-gauge__value-container">
                    {icon && <span className="metric-gauge__icon">{icon}</span>}
                    <span
                        className="metric-gauge__value"
                        style={{
                            fontSize: config.fontSize,
                            color: value !== null ? color : undefined,
                        }}
                    >
                        {displayValue}
                    </span>
                    {unit && <span className="metric-gauge__unit">{unit}</span>}
                </div>
            </div>
            <span className="metric-gauge__label">{label}</span>
        </div>
    );
}

// Predefined gauge configurations for common metrics
export function LossGauge({ value }: { value: number | null }) {
    return (
        <MetricGauge
            value={value}
            label="Loss"
            min={0}
            max={1}
            thresholds={{ good: 20, warning: 50, danger: 80 }}
            inverse
            formatValue={(v) => v.toFixed(4)}
        />
    );
}

export function PerplexityGauge({ value }: { value: number | null }) {
    return (
        <MetricGauge
            value={value}
            label="Perplexity"
            min={1}
            max={100}
            thresholds={{ good: 20, warning: 50, danger: 80 }}
            inverse
            formatValue={(v) => v.toFixed(1)}
        />
    );
}

export function ConfidenceGauge({ value }: { value: number | null }) {
    return (
        <MetricGauge
            value={value !== null ? value * 100 : null}
            label="Confidence"
            unit="%"
            min={0}
            max={100}
            thresholds={{ good: 80, warning: 60, danger: 40 }}
            formatValue={(v) => v.toFixed(1)}
        />
    );
}

export default MetricGauge;
