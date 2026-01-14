// ParetoFrontierChart.tsx - Multi-objective Pareto frontier visualization
// Displays trade-offs between loss, memory, and throughput

import { useState, useMemo, useCallback } from 'react';
import { Info, Target, TrendingUp } from 'lucide-react';
import type { HPOTrialInfo } from '../../api/types';
import './ParetoFrontierChart.css';

interface ParetoFrontierChartProps {
    trials: HPOTrialInfo[];
    xMetric?: ParetoMetric;
    yMetric?: ParetoMetric;
    bestTrialId?: string | number | null;
    onTrialSelect?: (trialId: string | number) => void;
    height?: number;
}

type ParetoMetric = 'loss' | 'perplexity' | 'memory' | 'throughput' | 'composite_score';

interface MetricConfig {
    key: ParetoMetric;
    label: string;
    getValue: (t: HPOTrialInfo) => number | null;
    lowerIsBetter: boolean;
    format: (v: number) => string;
}

const METRICS: MetricConfig[] = [
    {
        key: 'loss',
        label: 'Loss',
        getValue: (t) => t.loss,
        lowerIsBetter: true,
        format: (v) => v.toFixed(4),
    },
    {
        key: 'perplexity',
        label: 'Perplexity',
        getValue: (t) => t.perplexity ?? null,
        lowerIsBetter: true,
        format: (v) => v.toFixed(2),
    },
    {
        key: 'memory',
        label: 'Peak Memory (MB)',
        getValue: (t) => t.peak_memory_mb ?? null,
        lowerIsBetter: true,
        format: (v) => v >= 1024 ? `${(v / 1024).toFixed(1)}G` : `${v.toFixed(0)}M`,
    },
    {
        key: 'throughput',
        label: 'Throughput (tok/s)',
        getValue: (t) => t.throughput_tokens_per_sec ?? null,
        lowerIsBetter: false,
        format: (v) => v >= 1000 ? `${(v / 1000).toFixed(1)}K` : v.toFixed(0),
    },
    {
        key: 'composite_score',
        label: 'Composite Score',
        getValue: (t) => t.composite_score ?? null,
        lowerIsBetter: false,
        format: (v) => v.toFixed(4),
    },
];

function getMetricConfig(key: ParetoMetric): MetricConfig {
    return METRICS.find(m => m.key === key) || METRICS[0];
}

// Check if point A dominates point B
function dominates(
    aX: number, aY: number,
    bX: number, bY: number,
    xLowerBetter: boolean,
    yLowerBetter: boolean
): boolean {
    const aXBetter = xLowerBetter ? aX <= bX : aX >= bX;
    const aYBetter = yLowerBetter ? aY <= bY : aY >= bY;
    const aXStrictlyBetter = xLowerBetter ? aX < bX : aX > bX;
    const aYStrictlyBetter = yLowerBetter ? aY < bY : aY > bY;

    return aXBetter && aYBetter && (aXStrictlyBetter || aYStrictlyBetter);
}

export function ParetoFrontierChart({
    trials,
    xMetric = 'loss',
    yMetric = 'memory',
    bestTrialId,
    onTrialSelect,
    height = 300,
}: ParetoFrontierChartProps) {
    const [selectedXMetric, setSelectedXMetric] = useState<ParetoMetric>(xMetric);
    const [selectedYMetric, setSelectedYMetric] = useState<ParetoMetric>(yMetric);
    const [hoveredTrialId, setHoveredTrialId] = useState<string | number | null>(null);

    const xConfig = getMetricConfig(selectedXMetric);
    const yConfig = getMetricConfig(selectedYMetric);

    // Compute data points and Pareto frontier
    const { points, paretoFrontier, bounds } = useMemo(() => {
        const completedTrials = trials.filter(t => t.status === 'completed');

        // Extract valid data points
        const validPoints = completedTrials.map(trial => {
            const x = xConfig.getValue(trial);
            const y = yConfig.getValue(trial);
            if (x === null || y === null || isNaN(x) || isNaN(y)) return null;
            return { trial, x, y };
        }).filter((p): p is { trial: HPOTrialInfo; x: number; y: number } => p !== null);

        if (validPoints.length === 0) {
            return { points: [], paretoFrontier: [], bounds: { xMin: 0, xMax: 1, yMin: 0, yMax: 1 } };
        }

        // Compute bounds
        const xValues = validPoints.map(p => p.x);
        const yValues = validPoints.map(p => p.y);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);

        // Add 10% padding
        const xRange = xMax - xMin || 1;
        const yRange = yMax - yMin || 1;
        const bounds = {
            xMin: xMin - xRange * 0.1,
            xMax: xMax + xRange * 0.1,
            yMin: yMin - yRange * 0.1,
            yMax: yMax + yRange * 0.1,
        };

        // Find Pareto frontier
        const paretoFrontier: typeof validPoints = [];
        validPoints.forEach(point => {
            const isDominated = validPoints.some(other =>
                other !== point &&
                dominates(other.x, other.y, point.x, point.y, xConfig.lowerIsBetter, yConfig.lowerIsBetter)
            );
            if (!isDominated) {
                paretoFrontier.push(point);
            }
        });

        // Sort Pareto frontier by X for line drawing
        paretoFrontier.sort((a, b) => a.x - b.x);

        return { points: validPoints, paretoFrontier, bounds };
    }, [trials, xConfig, yConfig]);

    // SVG coordinate helpers
    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    const chartWidth = 100; // SVG viewbox percentage
    const chartHeight = height - margin.top - margin.bottom;

    const scaleX = useCallback((value: number) => {
        const normalized = (value - bounds.xMin) / (bounds.xMax - bounds.xMin);
        return margin.left + normalized * (chartWidth - margin.left - margin.right);
    }, [bounds]);

    const scaleY = useCallback((value: number) => {
        const normalized = (value - bounds.yMin) / (bounds.yMax - bounds.yMin);
        return margin.top + chartHeight - normalized * chartHeight;
    }, [bounds, chartHeight]);

    const handleTrialClick = useCallback((trialId: string | number) => {
        onTrialSelect?.(trialId);
    }, [onTrialSelect]);

    if (points.length < 2) {
        return (
            <div className="pareto-chart pareto-chart--empty">
                <Info size={24} />
                <p>Need at least 2 completed trials with valid metrics</p>
            </div>
        );
    }

    const hoveredPoint = points.find(p => p.trial.trial_id === hoveredTrialId);

    return (
        <div className="pareto-chart">
            <div className="pareto-chart__header">
                <h4 className="pareto-chart__title">
                    <Target size={16} />
                    Pareto Frontier
                </h4>
                <div className="pareto-chart__controls">
                    <label className="pareto-chart__axis-select">
                        <span>X:</span>
                        <select
                            value={selectedXMetric}
                            onChange={(e) => setSelectedXMetric(e.target.value as ParetoMetric)}
                        >
                            {METRICS.map(m => (
                                <option key={m.key} value={m.key}>{m.label}</option>
                            ))}
                        </select>
                    </label>
                    <label className="pareto-chart__axis-select">
                        <span>Y:</span>
                        <select
                            value={selectedYMetric}
                            onChange={(e) => setSelectedYMetric(e.target.value as ParetoMetric)}
                        >
                            {METRICS.map(m => (
                                <option key={m.key} value={m.key}>{m.label}</option>
                            ))}
                        </select>
                    </label>
                </div>
            </div>

            <svg className="pareto-chart__svg" viewBox={`0 0 100 ${height}`} preserveAspectRatio="xMidYMid meet">
                {/* Grid lines */}
                <defs>
                    <pattern id="pareto-grid" width="10" height="10" patternUnits="userSpaceOnUse">
                        <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="0.5" />
                    </pattern>
                </defs>
                <rect x={margin.left} y={margin.top} width={chartWidth - margin.left - margin.right} height={chartHeight} fill="url(#pareto-grid)" />

                {/* Axes */}
                <line
                    x1={margin.left}
                    y1={margin.top}
                    x2={margin.left}
                    y2={height - margin.bottom}
                    stroke="rgba(255,255,255,0.2)"
                    strokeWidth="1"
                />
                <line
                    x1={margin.left}
                    y1={height - margin.bottom}
                    x2={chartWidth - margin.right}
                    y2={height - margin.bottom}
                    stroke="rgba(255,255,255,0.2)"
                    strokeWidth="1"
                />

                {/* Axis labels */}
                <text
                    x={(chartWidth + margin.left - margin.right) / 2}
                    y={height - 5}
                    className="pareto-chart__axis-label"
                    textAnchor="middle"
                >
                    {xConfig.label}
                </text>
                <text
                    x={10}
                    y={(height - margin.top - margin.bottom) / 2 + margin.top}
                    className="pareto-chart__axis-label"
                    textAnchor="middle"
                    transform={`rotate(-90, 10, ${(height - margin.top - margin.bottom) / 2 + margin.top})`}
                >
                    {yConfig.label}
                </text>

                {/* Pareto frontier line */}
                {paretoFrontier.length > 1 && (
                    <polyline
                        className="pareto-chart__frontier-line"
                        points={paretoFrontier.map(p => `${scaleX(p.x)},${scaleY(p.y)}`).join(' ')}
                        fill="none"
                        stroke="#10b981"
                        strokeWidth="2"
                        strokeDasharray="4,2"
                    />
                )}

                {/* Non-Pareto points (dominated) */}
                {points.filter(p => !paretoFrontier.includes(p)).map(point => (
                    <circle
                        key={String(point.trial.trial_id)}
                        cx={scaleX(point.x)}
                        cy={scaleY(point.y)}
                        r={point.trial.trial_id === bestTrialId ? 4 : 3}
                        className={`pareto-chart__point pareto-chart__point--dominated ${point.trial.trial_id === hoveredTrialId ? 'pareto-chart__point--hovered' : ''
                            }`}
                        onMouseEnter={() => setHoveredTrialId(point.trial.trial_id)}
                        onMouseLeave={() => setHoveredTrialId(null)}
                        onClick={() => handleTrialClick(point.trial.trial_id)}
                    />
                ))}

                {/* Pareto-optimal points */}
                {paretoFrontier.map(point => (
                    <circle
                        key={`pareto-${point.trial.trial_id}`}
                        cx={scaleX(point.x)}
                        cy={scaleY(point.y)}
                        r={point.trial.trial_id === bestTrialId ? 5 : 4}
                        className={`pareto-chart__point pareto-chart__point--optimal ${point.trial.trial_id === bestTrialId ? 'pareto-chart__point--best' : ''
                            } ${point.trial.trial_id === hoveredTrialId ? 'pareto-chart__point--hovered' : ''}`}
                        onMouseEnter={() => setHoveredTrialId(point.trial.trial_id)}
                        onMouseLeave={() => setHoveredTrialId(null)}
                        onClick={() => handleTrialClick(point.trial.trial_id)}
                    />
                ))}
            </svg>

            {/* Tooltip */}
            {hoveredPoint && (
                <div className="pareto-chart__tooltip">
                    <div className="pareto-chart__tooltip-header">
                        Trial {hoveredPoint.trial.trial_id}
                        {paretoFrontier.some(p => p.trial.trial_id === hoveredPoint.trial.trial_id) && (
                            <span className="pareto-badge">Pareto</span>
                        )}
                        {hoveredPoint.trial.trial_id === bestTrialId && (
                            <span className="best-badge">★ Best</span>
                        )}
                    </div>
                    <div className="pareto-chart__tooltip-row">
                        <span>{xConfig.label}:</span>
                        <span>{xConfig.format(hoveredPoint.x)}</span>
                    </div>
                    <div className="pareto-chart__tooltip-row">
                        <span>{yConfig.label}:</span>
                        <span>{yConfig.format(hoveredPoint.y)}</span>
                    </div>
                </div>
            )}

            {/* Legend */}
            <div className="pareto-chart__legend">
                <span className="pareto-chart__legend-item">
                    <span className="pareto-chart__legend-dot pareto-chart__legend-dot--optimal" />
                    Pareto-optimal ({paretoFrontier.length})
                </span>
                <span className="pareto-chart__legend-item">
                    <span className="pareto-chart__legend-dot pareto-chart__legend-dot--dominated" />
                    Dominated ({points.length - paretoFrontier.length})
                </span>
            </div>
        </div>
    );
}

export default ParetoFrontierChart;
