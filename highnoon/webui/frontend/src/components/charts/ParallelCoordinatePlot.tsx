// ParallelCoordinatePlot.tsx - Multi-dimensional hyperparameter visualization
// Displays all trial hyperparameters as parallel coordinates with loss coloring

import { useState, useMemo, useCallback } from 'react';
import { Info, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';
import type { HPOTrialInfo } from '../../api/types';
import './ParallelCoordinatePlot.css';

interface ParallelCoordinatePlotProps {
    trials: HPOTrialInfo[];
    colorMetric?: 'loss' | 'composite_score' | 'perplexity';
    bestTrialId?: string | number | null;
    onTrialSelect?: (trialId: string | number) => void;
    height?: number;
}

// Extract numeric hyperparams that vary across trials
interface AxisConfig {
    key: string;
    label: string;
    min: number;
    max: number;
    isLog: boolean;
    values: (number | null)[];
}

// Color scale from red (high loss) to green (low loss)
function lossToColor(loss: number, minLoss: number, maxLoss: number): string {
    const range = maxLoss - minLoss || 1;
    const normalized = 1 - (loss - minLoss) / range; // 0 = worst, 1 = best

    // HSL interpolation: red (0°) -> yellow (60°) -> green (120°)
    const hue = normalized * 120;
    const saturation = 70;
    const lightness = 50;

    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

// Format axis value for display
function formatAxisValue(value: number, isLog: boolean): string {
    if (isLog && value > 0) {
        return value.toExponential(1);
    }
    if (value >= 1000000) {
        return `${(value / 1000000).toFixed(1)}M`;
    }
    if (value >= 1000) {
        return `${(value / 1000).toFixed(1)}K`;
    }
    if (value < 0.01 && value > 0) {
        return value.toExponential(1);
    }
    return value.toFixed(value < 1 ? 4 : 1);
}

// Known hyperparameter display names
const PARAM_LABELS: Record<string, string> = {
    learning_rate: 'Learning Rate',
    embedding_dim: 'Embed Dim',
    num_reasoning_blocks: 'Blocks',
    num_moe_experts: 'Experts',
    batch_size: 'Batch',
    mamba_state_dim: 'Mamba Dim',
    superposition_dim: 'Super Dim',
    wlam_heads: 'WLAM Heads',
    weight_decay: 'Weight Decay',
    grad_clip: 'Grad Clip',
    hd_sample_length: 'HD Sample',
    vocab_size: 'Vocab',
};

// Parameters that should use log scale
const LOG_SCALE_PARAMS = new Set([
    'learning_rate',
    'weight_decay',
]);

export function ParallelCoordinatePlot({
    trials,
    colorMetric = 'loss',
    bestTrialId,
    onTrialSelect,
    height = 320,
}: ParallelCoordinatePlotProps) {
    const [hoveredTrialId, setHoveredTrialId] = useState<string | number | null>(null);
    const [selectedAxes, setSelectedAxes] = useState<string[]>([]);

    // Extract and configure axes from trial hyperparams
    const { axes, trialPaths, colorRange } = useMemo(() => {
        // Collect all hyperparams across trials
        const paramValues: Record<string, (number | null)[]> = {};

        const completedTrials = trials.filter(t =>
            t.status === 'completed' && t.loss !== null && t.hyperparams
        );

        if (completedTrials.length === 0) {
            return { axes: [], trialPaths: [], colorRange: { min: 0, max: 1 } };
        }

        // Always include these core params
        const coreParams = [
            'learning_rate',
            'embedding_dim',
            'num_reasoning_blocks',
            'num_moe_experts',
            'batch_size',
        ];

        completedTrials.forEach(trial => {
            const hp = trial.hyperparams || {};

            // Add core params
            coreParams.forEach(key => {
                if (!(key in paramValues)) paramValues[key] = [];
                const val = hp[key] ?? (key === 'learning_rate' ? trial.learning_rate : null);
                paramValues[key].push(typeof val === 'number' ? val : null);
            });

            // Add additional numeric params
            Object.entries(hp).forEach(([key, value]) => {
                if (coreParams.includes(key)) return;
                if (typeof value !== 'number') return;
                if (key.startsWith('_')) return; // Skip internal params

                if (!(key in paramValues)) paramValues[key] = [];
                paramValues[key].push(value);
            });
        });

        // Filter to params that vary
        const validAxes: AxisConfig[] = [];
        Object.entries(paramValues).forEach(([key, values]) => {
            const numericValues = values.filter((v): v is number => v !== null);
            if (numericValues.length < 2) return;

            const min = Math.min(...numericValues);
            const max = Math.max(...numericValues);
            if (min === max) return; // No variance

            validAxes.push({
                key,
                label: PARAM_LABELS[key] || key.replace(/_/g, ' '),
                min,
                max,
                isLog: LOG_SCALE_PARAMS.has(key),
                values,
            });
        });

        // Sort: core params first, then alphabetically
        validAxes.sort((a, b) => {
            const aCore = coreParams.indexOf(a.key);
            const bCore = coreParams.indexOf(b.key);
            if (aCore !== -1 && bCore !== -1) return aCore - bCore;
            if (aCore !== -1) return -1;
            if (bCore !== -1) return 1;
            return a.label.localeCompare(b.label);
        });

        // Limit to 8 axes for readability
        const displayAxes = selectedAxes.length > 0
            ? validAxes.filter(a => selectedAxes.includes(a.key))
            : validAxes.slice(0, 8);

        // Calculate color range
        const colorValues = completedTrials.map(t => {
            if (colorMetric === 'loss') return t.loss ?? Infinity;
            if (colorMetric === 'composite_score') return t.composite_score ?? Infinity;
            if (colorMetric === 'perplexity') return t.perplexity ?? Infinity;
            return t.loss ?? Infinity;
        }).filter(v => v !== Infinity && !isNaN(v));

        const colorRange = {
            min: Math.min(...colorValues),
            max: Math.max(...colorValues),
        };

        // Build path data for each trial
        const trialPaths = completedTrials.map((trial, trialIndex) => {
            const points: { x: number; y: number }[] = [];

            displayAxes.forEach((axis, axisIndex) => {
                const value = axis.values[trialIndex];
                if (value === null) return;

                const x = axisIndex;
                let y: number;

                if (axis.isLog && axis.min > 0 && axis.max > 0) {
                    // Log scale
                    const logMin = Math.log10(axis.min);
                    const logMax = Math.log10(axis.max);
                    const logVal = Math.log10(value);
                    y = (logVal - logMin) / (logMax - logMin);
                } else {
                    // Linear scale
                    y = (value - axis.min) / (axis.max - axis.min);
                }

                points.push({ x, y: 1 - y }); // Invert Y so high values are at top
            });

            const colorValue = colorMetric === 'loss' ? trial.loss
                : colorMetric === 'composite_score' ? trial.composite_score
                    : trial.perplexity;

            return {
                trial,
                points,
                color: lossToColor(colorValue ?? colorRange.max, colorRange.min, colorRange.max),
                isBest: trial.trial_id === bestTrialId,
            };
        });

        return { axes: displayAxes, trialPaths, colorRange };
    }, [trials, colorMetric, bestTrialId, selectedAxes]);

    const handleTrialClick = useCallback((trialId: string | number) => {
        onTrialSelect?.(trialId);
    }, [onTrialSelect]);

    // SVG dimensions
    const margin = { top: 30, right: 20, bottom: 40, left: 20 };
    const width = 100; // Percentage-based for responsive
    const chartHeight = height - margin.top - margin.bottom;
    const axisSpacing = 100 / (axes.length - 1 || 1);

    if (trials.length === 0 || axes.length < 2) {
        return (
            <div className="parallel-coords parallel-coords--empty">
                <Info size={24} />
                <p>Need at least 2 completed trials with varying hyperparameters</p>
            </div>
        );
    }

    return (
        <div className="parallel-coords">
            <div className="parallel-coords__header">
                <h4 className="parallel-coords__title">
                    Hyperparameter Space
                    <span className="parallel-coords__count">
                        {trialPaths.length} trials
                    </span>
                </h4>
                <div className="parallel-coords__legend">
                    <span className="parallel-coords__legend-label">Loss:</span>
                    <div className="parallel-coords__gradient" />
                    <span className="parallel-coords__legend-min">
                        {formatAxisValue(colorRange.min, false)}
                    </span>
                    <span className="parallel-coords__legend-max">
                        {formatAxisValue(colorRange.max, false)}
                    </span>
                </div>
            </div>

            <svg
                className="parallel-coords__svg"
                viewBox={`0 0 100 ${height}`}
                preserveAspectRatio="none"
                style={{ height }}
            >
                {/* Axes */}
                {axes.map((axis, i) => {
                    const x = margin.left / 100 * 100 + i * axisSpacing * (100 - margin.left - margin.right) / 100;
                    const xPercent = margin.left / 100 * 100 + i * (100 - margin.left - margin.right) / (axes.length - 1 || 1);

                    return (
                        <g key={axis.key} className="parallel-coords__axis">
                            {/* Axis line */}
                            <line
                                x1={`${xPercent}%`}
                                y1={margin.top}
                                x2={`${xPercent}%`}
                                y2={height - margin.bottom}
                                className="parallel-coords__axis-line"
                            />

                            {/* Axis label */}
                            <text
                                x={`${xPercent}%`}
                                y={margin.top - 12}
                                className="parallel-coords__axis-label"
                                textAnchor="middle"
                            >
                                {axis.label}
                            </text>

                            {/* Min value */}
                            <text
                                x={`${xPercent}%`}
                                y={height - margin.bottom + 16}
                                className="parallel-coords__axis-value"
                                textAnchor="middle"
                            >
                                {formatAxisValue(axis.min, axis.isLog)}
                            </text>

                            {/* Max value */}
                            <text
                                x={`${xPercent}%`}
                                y={margin.top - 2}
                                className="parallel-coords__axis-value parallel-coords__axis-value--max"
                                textAnchor="middle"
                            >
                                {formatAxisValue(axis.max, axis.isLog)}
                            </text>
                        </g>
                    );
                })}

                {/* Trial paths (non-hovered first) */}
                {trialPaths
                    .filter(p => p.trial.trial_id !== hoveredTrialId && !p.isBest)
                    .map(({ trial, points, color }) => (
                        <polyline
                            key={String(trial.trial_id)}
                            className="parallel-coords__path"
                            points={points.map((p, i) => {
                                const xPercent = margin.left + i * (100 - margin.left - margin.right) / (axes.length - 1 || 1);
                                const yPos = margin.top + p.y * chartHeight;
                                return `${xPercent},${yPos}`;
                            }).join(' ')}
                            stroke={color}
                            strokeWidth={1.5}
                            fill="none"
                            opacity={0.4}
                            onMouseEnter={() => setHoveredTrialId(trial.trial_id)}
                            onMouseLeave={() => setHoveredTrialId(null)}
                            onClick={() => handleTrialClick(trial.trial_id)}
                        />
                    ))
                }

                {/* Best trial path */}
                {trialPaths
                    .filter(p => p.isBest)
                    .map(({ trial, points, color }) => (
                        <polyline
                            key={`best-${trial.trial_id}`}
                            className="parallel-coords__path parallel-coords__path--best"
                            points={points.map((p, i) => {
                                const xPercent = margin.left + i * (100 - margin.left - margin.right) / (axes.length - 1 || 1);
                                const yPos = margin.top + p.y * chartHeight;
                                return `${xPercent},${yPos}`;
                            }).join(' ')}
                            stroke="#10b981"
                            strokeWidth={3}
                            fill="none"
                            opacity={0.9}
                            onMouseEnter={() => setHoveredTrialId(trial.trial_id)}
                            onMouseLeave={() => setHoveredTrialId(null)}
                            onClick={() => handleTrialClick(trial.trial_id)}
                        />
                    ))
                }

                {/* Hovered trial path (on top) */}
                {trialPaths
                    .filter(p => p.trial.trial_id === hoveredTrialId)
                    .map(({ trial, points, color }) => (
                        <polyline
                            key={`hover-${trial.trial_id}`}
                            className="parallel-coords__path parallel-coords__path--hovered"
                            points={points.map((p, i) => {
                                const xPercent = margin.left + i * (100 - margin.left - margin.right) / (axes.length - 1 || 1);
                                const yPos = margin.top + p.y * chartHeight;
                                return `${xPercent},${yPos}`;
                            }).join(' ')}
                            stroke={color}
                            strokeWidth={3}
                            fill="none"
                            opacity={1}
                        />
                    ))
                }
            </svg>

            {/* Tooltip */}
            {hoveredTrialId && (
                <div className="parallel-coords__tooltip">
                    {(() => {
                        const trial = trials.find(t => t.trial_id === hoveredTrialId);
                        if (!trial) return null;
                        return (
                            <>
                                <div className="parallel-coords__tooltip-header">
                                    Trial {trial.trial_id}
                                    {trial.trial_id === bestTrialId && <span className="best-badge">★ Best</span>}
                                </div>
                                <div className="parallel-coords__tooltip-row">
                                    <span>Loss:</span>
                                    <span>{trial.loss?.toFixed(6) ?? '—'}</span>
                                </div>
                                {trial.perplexity && (
                                    <div className="parallel-coords__tooltip-row">
                                        <span>Perplexity:</span>
                                        <span>{trial.perplexity.toFixed(2)}</span>
                                    </div>
                                )}
                                <div className="parallel-coords__tooltip-row">
                                    <span>Status:</span>
                                    <span>{trial.status}</span>
                                </div>
                            </>
                        );
                    })()}
                </div>
            )}
        </div>
    );
}

export default ParallelCoordinatePlot;
