// HUDSpeedometer.tsx - Arc-style speedometer gauge for core metrics
// F1/Jet cockpit style with animated needle and glow effects

import { useEffect, useRef, useState, useCallback } from 'react';
import './HUDGauges.css';

interface HUDSpeedometerProps {
    value: number | null;
    label: string;
    sublabel?: string;
    min?: number;
    max?: number;
    unit?: string;
    thresholds?: {
        good: number;
        warning: number;
        danger: number;
    };
    inverse?: boolean;  // Lower is better (like loss)
    formatValue?: (value: number) => string;
    size?: 'sm' | 'md' | 'lg';
    animated?: boolean;
}

const SIZE_CONFIG = {
    sm: { diameter: 100, strokeWidth: 6, needleLength: 35, fontSize: '0.875rem' },
    md: { diameter: 140, strokeWidth: 8, needleLength: 50, fontSize: '1.125rem' },
    lg: { diameter: 180, strokeWidth: 10, needleLength: 65, fontSize: '1.5rem' },
};

export function HUDSpeedometer({
    value,
    label,
    sublabel,
    min = 0,
    max = 1,
    unit = '',
    thresholds = { good: 30, warning: 60, danger: 85 },
    inverse = false,
    formatValue,
    size = 'md',
    animated = true,
}: HUDSpeedometerProps) {
    const [animatedValue, setAnimatedValue] = useState(0);
    const animationRef = useRef<number | null>(null);
    const previousValueRef = useRef<number>(0);

    const config = SIZE_CONFIG[size];
    const radius = (config.diameter - config.strokeWidth) / 2;
    const center = config.diameter / 2;

    // Arc spans 270 degrees (-225 to 45 degrees)
    const startAngle = -225;
    const endAngle = 45;
    const arcLength = endAngle - startAngle; // 270 degrees

    // Animate value changes
    useEffect(() => {
        if (value === null) {
            setAnimatedValue(0);
            return;
        }

        if (!animated) {
            setAnimatedValue(Math.max(min, Math.min(max, value)));
            return;
        }

        const startValue = previousValueRef.current;
        const endValue = Math.max(min, Math.min(max, value));
        const duration = 800;
        const startTime = Date.now();

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease-out cubic with slight overshoot for mechanical feel
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
    }, [value, min, max, animated]);

    // Calculate percentage and needle angle
    const percentage = Math.max(0, Math.min(100, ((animatedValue - min) / (max - min)) * 100));
    const needleAngle = startAngle + (percentage / 100) * arcLength;

    // Create arc path for background track
    const createArcPath = (startDeg: number, endDeg: number, r: number) => {
        const startRad = (startDeg * Math.PI) / 180;
        const endRad = (endDeg * Math.PI) / 180;
        const x1 = center + r * Math.cos(startRad);
        const y1 = center + r * Math.sin(startRad);
        const x2 = center + r * Math.cos(endRad);
        const y2 = center + r * Math.sin(endRad);
        const largeArc = endDeg - startDeg > 180 ? 1 : 0;
        return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
    };

    // Calculate color based on thresholds
    const getColor = useCallback(() => {
        if (value === null) return 'var(--text-muted)';

        const pct = percentage;
        if (inverse) {
            if (pct <= thresholds.good) return 'var(--color-success)';
            if (pct <= thresholds.warning) return 'var(--color-warning)';
            return 'var(--color-danger)';
        } else {
            if (pct >= (100 - thresholds.good)) return 'var(--color-success)';
            if (pct >= (100 - thresholds.warning)) return 'var(--color-warning)';
            return 'var(--color-danger)';
        }
    }, [value, percentage, inverse, thresholds]);

    const color = getColor();
    const displayValue = value !== null
        ? (formatValue ? formatValue(animatedValue) : animatedValue.toFixed(4))
        : '—';

    // Create colored arc segments for the gauge face
    const createSegmentPath = (startPct: number, endPct: number, r: number) => {
        const segStartAngle = startAngle + (startPct / 100) * arcLength;
        const segEndAngle = startAngle + (endPct / 100) * arcLength;
        return createArcPath(segStartAngle, segEndAngle, r);
    };

    return (
        <div className={`hud-speedometer hud-speedometer--${size}`}>
            <div className="hud-speedometer__ring">
                <svg
                    width={config.diameter}
                    height={config.diameter}
                    className="hud-speedometer__svg"
                >
                    {/* Outer glow ring */}
                    <circle
                        className="hud-speedometer__outer-glow"
                        cx={center}
                        cy={center}
                        r={radius + 4}
                    />

                    {/* Background track */}
                    <path
                        className="hud-speedometer__track"
                        d={createArcPath(startAngle, endAngle, radius)}
                        fill="none"
                        strokeWidth={config.strokeWidth}
                    />

                    {/* Color zone segments - Good (green) */}
                    <path
                        className="hud-speedometer__zone hud-speedometer__zone--good"
                        d={inverse
                            ? createSegmentPath(0, thresholds.good, radius)
                            : createSegmentPath(100 - thresholds.good, 100, radius)}
                        fill="none"
                        strokeWidth={config.strokeWidth - 2}
                    />

                    {/* Color zone segments - Warning (yellow) */}
                    <path
                        className="hud-speedometer__zone hud-speedometer__zone--warning"
                        d={inverse
                            ? createSegmentPath(thresholds.good, thresholds.warning, radius)
                            : createSegmentPath(100 - thresholds.warning, 100 - thresholds.good, radius)}
                        fill="none"
                        strokeWidth={config.strokeWidth - 2}
                    />

                    {/* Color zone segments - Danger (red) */}
                    <path
                        className="hud-speedometer__zone hud-speedometer__zone--danger"
                        d={inverse
                            ? createSegmentPath(thresholds.warning, 100, radius)
                            : createSegmentPath(0, 100 - thresholds.warning, radius)}
                        fill="none"
                        strokeWidth={config.strokeWidth - 2}
                    />

                    {/* Progress arc with glow */}
                    <path
                        className="hud-speedometer__progress-glow"
                        d={createArcPath(startAngle, needleAngle, radius)}
                        fill="none"
                        strokeWidth={config.strokeWidth + 6}
                        style={{ stroke: color }}
                    />
                    <path
                        className="hud-speedometer__progress"
                        d={createArcPath(startAngle, needleAngle, radius)}
                        fill="none"
                        strokeWidth={config.strokeWidth}
                        style={{ stroke: color }}
                    />

                    {/* Tick marks */}
                    {[...Array(10)].map((_, i) => {
                        const tickAngle = startAngle + (i / 9) * arcLength;
                        const tickRad = (tickAngle * Math.PI) / 180;
                        const innerR = radius - config.strokeWidth / 2 - 6;
                        const outerR = radius - config.strokeWidth / 2 - 2;
                        return (
                            <line
                                key={i}
                                className="hud-speedometer__tick"
                                x1={center + innerR * Math.cos(tickRad)}
                                y1={center + innerR * Math.sin(tickRad)}
                                x2={center + outerR * Math.cos(tickRad)}
                                y2={center + outerR * Math.sin(tickRad)}
                            />
                        );
                    })}

                    {/* Center hub */}
                    <circle
                        className="hud-speedometer__hub"
                        cx={center}
                        cy={center}
                        r={8}
                    />

                    {/* Needle */}
                    <g
                        className="hud-speedometer__needle-group"
                        transform={`rotate(${needleAngle} ${center} ${center})`}
                    >
                        <line
                            className="hud-speedometer__needle"
                            x1={center}
                            y1={center}
                            x2={center + config.needleLength}
                            y2={center}
                            style={{ stroke: color }}
                        />
                        <circle
                            className="hud-speedometer__needle-dot"
                            cx={center + config.needleLength}
                            cy={center}
                            r={3}
                            style={{ fill: color }}
                        />
                    </g>
                </svg>

                {/* Digital readout */}
                <div className="hud-speedometer__readout">
                    <span
                        className="hud-speedometer__value"
                        style={{ fontSize: config.fontSize, color: value !== null ? color : undefined }}
                    >
                        {displayValue}
                    </span>
                    {unit && <span className="hud-speedometer__unit">{unit}</span>}
                </div>
            </div>

            <div className="hud-speedometer__labels">
                <span className="hud-speedometer__label">{label}</span>
                {sublabel && <span className="hud-speedometer__sublabel">{sublabel}</span>}
            </div>
        </div>
    );
}

export default HUDSpeedometer;
