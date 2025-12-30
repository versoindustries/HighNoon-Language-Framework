// HUDTachometer.tsx - Full-circle RPM-style gauge for rate metrics
// F1/Jet cockpit style with segment fills and redline indicator

import { useEffect, useRef, useState } from 'react';
import './HUDGauges.css';

interface HUDTachometerProps {
    value: number | null;
    label: string;
    max?: number;
    redline?: number;  // Percentage where redline starts
    segments?: number;
    formatValue?: (value: number) => string;
    size?: 'sm' | 'md' | 'lg';
    animated?: boolean;
}

const SIZE_CONFIG = {
    sm: { diameter: 80, barWidth: 4, innerRadius: 28 },
    md: { diameter: 110, barWidth: 5, innerRadius: 38 },
    lg: { diameter: 140, barWidth: 6, innerRadius: 50 },
};

export function HUDTachometer({
    value,
    label,
    max = 100,
    redline = 80,
    segments = 12,
    formatValue,
    size = 'md',
    animated = true,
}: HUDTachometerProps) {
    const [animatedValue, setAnimatedValue] = useState(0);
    const animationRef = useRef<number | null>(null);
    const previousValueRef = useRef<number>(0);

    const config = SIZE_CONFIG[size];
    const center = config.diameter / 2;
    const outerRadius = (config.diameter - config.barWidth) / 2;
    const gapAngle = 4; // Gap between segments in degrees

    // Animate value changes
    useEffect(() => {
        if (value === null) {
            setAnimatedValue(0);
            return;
        }

        if (!animated) {
            setAnimatedValue(Math.max(0, Math.min(max, value)));
            return;
        }

        const startValue = previousValueRef.current;
        const endValue = Math.max(0, Math.min(max, value));
        const duration = 600;
        const startTime = Date.now();

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
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
    }, [value, max, animated]);

    const percentage = (animatedValue / max) * 100;
    const filledSegments = Math.floor((percentage / 100) * segments);
    const inRedline = percentage >= redline;

    const displayValue = value !== null
        ? (formatValue ? formatValue(animatedValue) : animatedValue.toFixed(2))
        : '—';

    // Create segment arc path
    const createSegmentPath = (index: number) => {
        const segmentAngle = (360 - segments * gapAngle) / segments;
        const startAngle = -90 + index * (segmentAngle + gapAngle);
        const endAngle = startAngle + segmentAngle;

        const startRad = (startAngle * Math.PI) / 180;
        const endRad = (endAngle * Math.PI) / 180;

        const x1 = center + outerRadius * Math.cos(startRad);
        const y1 = center + outerRadius * Math.sin(startRad);
        const x2 = center + outerRadius * Math.cos(endRad);
        const y2 = center + outerRadius * Math.sin(endRad);

        return `M ${x1} ${y1} A ${outerRadius} ${outerRadius} 0 0 1 ${x2} ${y2}`;
    };

    // Determine segment color
    const getSegmentColor = (index: number, isFilled: boolean) => {
        if (!isFilled) return 'var(--bg-muted)';

        const segmentPct = ((index + 1) / segments) * 100;
        if (segmentPct >= redline) {
            return inRedline ? 'var(--color-danger)' : 'var(--color-danger-muted)';
        }
        if (segmentPct >= redline - 20) {
            return 'var(--color-warning)';
        }
        return 'var(--color-primary)';
    };

    return (
        <div className={`hud-tachometer hud-tachometer--${size} ${inRedline ? 'hud-tachometer--redline' : ''}`}>
            <svg
                width={config.diameter}
                height={config.diameter}
                className="hud-tachometer__svg"
            >
                {/* Inner glow ring */}
                <circle
                    className="hud-tachometer__inner-ring"
                    cx={center}
                    cy={center}
                    r={config.innerRadius}
                />

                {/* Segment arcs */}
                {[...Array(segments)].map((_, i) => {
                    const isFilled = i < filledSegments;
                    const isRedzone = ((i + 1) / segments) * 100 >= redline;
                    return (
                        <path
                            key={i}
                            className={`hud-tachometer__segment ${isFilled ? 'hud-tachometer__segment--filled' : ''} ${isRedzone ? 'hud-tachometer__segment--redzone' : ''}`}
                            d={createSegmentPath(i)}
                            fill="none"
                            strokeWidth={config.barWidth}
                            style={{ stroke: getSegmentColor(i, isFilled) }}
                        />
                    );
                })}

                {/* Center circle */}
                <circle
                    className="hud-tachometer__center"
                    cx={center}
                    cy={center}
                    r={config.innerRadius - 4}
                />
            </svg>

            {/* Value display */}
            <div className="hud-tachometer__value-container">
                <span className={`hud-tachometer__value ${inRedline ? 'hud-tachometer__value--redline' : ''}`}>
                    {displayValue}
                </span>
                <span className="hud-tachometer__label">{label}</span>
            </div>
        </div>
    );
}

export default HUDTachometer;
