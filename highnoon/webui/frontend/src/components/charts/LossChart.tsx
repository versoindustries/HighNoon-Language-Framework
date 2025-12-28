// LossChart.tsx - Real-time loss convergence visualization
// Canvas-based chart for high-performance rendering of training metrics

import { useEffect, useRef, useState, useCallback } from 'react';
import './LossChart.css';

interface DataPoint {
    step: number;
    loss: number;
    trialId: string | number;
}

interface TrialData {
    trialId: string | number;
    points: DataPoint[];
    status: 'running' | 'completed' | 'pruned' | 'failed';
    isBest?: boolean;
}

interface LossChartProps {
    trials: TrialData[];
    currentTrialId?: string | number | null;
    bestTrialId?: string | number | null;
    height?: number;
    showLegend?: boolean;
    animationEnabled?: boolean;
}

// Color palette for trials
const TRIAL_COLORS = [
    '#6366f1', // Indigo (primary)
    '#8b5cf6', // Purple
    '#ec4899', // Pink
    '#f59e0b', // Amber
    '#10b981', // Emerald
    '#3b82f6', // Blue
    '#f43f5e', // Rose
    '#14b8a6', // Teal
    '#a855f7', // Violet
    '#eab308', // Yellow
];

const BEST_TRIAL_COLOR = '#10b981'; // Emerald green for best
const CURRENT_TRIAL_COLOR = '#6366f1'; // Primary indigo for current
const GRID_COLOR = 'rgba(255, 255, 255, 0.06)';
const AXIS_COLOR = 'rgba(255, 255, 255, 0.15)';
const LABEL_COLOR = 'rgba(255, 255, 255, 0.5)';

export function LossChart({
    trials,
    currentTrialId,
    bestTrialId,
    height = 280,
    showLegend = true,
    animationEnabled = true,
}: LossChartProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const animationFrameRef = useRef<number | null>(null);
    const [dimensions, setDimensions] = useState({ width: 600, height });
    const [hoveredPoint, setHoveredPoint] = useState<{
        x: number;
        y: number;
        point: DataPoint;
        trial: TrialData;
    } | null>(null);

    // Get color for a trial
    const getTrialColor = useCallback((trial: TrialData, index: number): string => {
        if (trial.trialId === bestTrialId) return BEST_TRIAL_COLOR;
        if (trial.trialId === currentTrialId) return CURRENT_TRIAL_COLOR;
        return TRIAL_COLORS[index % TRIAL_COLORS.length];
    }, [bestTrialId, currentTrialId]);

    // Compute chart bounds
    const computeBounds = useCallback(() => {
        let minLoss = Infinity;
        let maxLoss = -Infinity;
        let maxStep = 0;

        trials.forEach(trial => {
            trial.points.forEach(point => {
                if (point.loss < minLoss) minLoss = point.loss;
                if (point.loss > maxLoss) maxLoss = point.loss;
                if (point.step > maxStep) maxStep = point.step;
            });
        });

        // Add padding
        const lossRange = maxLoss - minLoss || 1;
        minLoss = Math.max(0, minLoss - lossRange * 0.1);
        maxLoss = maxLoss + lossRange * 0.1;
        maxStep = Math.max(maxStep, 10);

        return { minLoss, maxLoss, maxStep };
    }, [trials]);

    // Draw the chart
    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        if (!canvas || !ctx) return;

        const { width, height } = dimensions;
        const dpr = window.devicePixelRatio || 1;

        // Set canvas size accounting for device pixel ratio
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        ctx.scale(dpr, dpr);

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Chart area (with margins for axes)
        const margin = { top: 20, right: 20, bottom: 40, left: 60 };
        const chartWidth = width - margin.left - margin.right;
        const chartHeight = height - margin.top - margin.bottom;

        if (trials.length === 0 || trials.every(t => t.points.length === 0)) {
            // Empty state
            ctx.fillStyle = LABEL_COLOR;
            ctx.font = '14px Inter, system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Waiting for training data...', width / 2, height / 2);
            return;
        }

        const { minLoss, maxLoss, maxStep } = computeBounds();

        // Scale functions
        const scaleX = (step: number) => margin.left + (step / maxStep) * chartWidth;
        const scaleY = (loss: number) => margin.top + chartHeight - ((loss - minLoss) / (maxLoss - minLoss)) * chartHeight;

        // Draw grid
        ctx.strokeStyle = GRID_COLOR;
        ctx.lineWidth = 1;

        // Horizontal grid lines (5 lines)
        for (let i = 0; i <= 4; i++) {
            const y = margin.top + (chartHeight / 4) * i;
            ctx.beginPath();
            ctx.moveTo(margin.left, y);
            ctx.lineTo(width - margin.right, y);
            ctx.stroke();
        }

        // Vertical grid lines (5 lines)
        for (let i = 0; i <= 4; i++) {
            const x = margin.left + (chartWidth / 4) * i;
            ctx.beginPath();
            ctx.moveTo(x, margin.top);
            ctx.lineTo(x, height - margin.bottom);
            ctx.stroke();
        }

        // Draw axes
        ctx.strokeStyle = AXIS_COLOR;
        ctx.lineWidth = 2;

        // Y-axis
        ctx.beginPath();
        ctx.moveTo(margin.left, margin.top);
        ctx.lineTo(margin.left, height - margin.bottom);
        ctx.stroke();

        // X-axis
        ctx.beginPath();
        ctx.moveTo(margin.left, height - margin.bottom);
        ctx.lineTo(width - margin.right, height - margin.bottom);
        ctx.stroke();

        // Draw axis labels
        ctx.fillStyle = LABEL_COLOR;
        ctx.font = '11px Inter, system-ui, sans-serif';

        // Y-axis labels
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let i = 0; i <= 4; i++) {
            const value = minLoss + ((maxLoss - minLoss) / 4) * (4 - i);
            const y = margin.top + (chartHeight / 4) * i;
            ctx.fillText(value.toFixed(4), margin.left - 8, y);
        }

        // X-axis labels
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (let i = 0; i <= 4; i++) {
            const value = Math.round((maxStep / 4) * i);
            const x = margin.left + (chartWidth / 4) * i;
            ctx.fillText(value.toString(), x, height - margin.bottom + 8);
        }

        // Axis titles
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.font = '12px Inter, system-ui, sans-serif';

        // Y-axis title
        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('Loss', 0, 0);
        ctx.restore();

        // X-axis title
        ctx.textAlign = 'center';
        ctx.fillText('Training Steps', width / 2, height - 8);

        // Draw trial lines
        trials.forEach((trial, trialIndex) => {
            if (trial.points.length === 0) return;

            const color = getTrialColor(trial, trialIndex);
            const isBest = trial.trialId === bestTrialId;
            const isCurrent = trial.trialId === currentTrialId;

            ctx.strokeStyle = color;
            ctx.lineWidth = isBest || isCurrent ? 2.5 : 1.5;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';

            // Set line dash for non-best, non-current trials
            if (!isBest && !isCurrent) {
                ctx.globalAlpha = 0.6;
            }

            ctx.beginPath();
            trial.points.forEach((point, i) => {
                const x = scaleX(point.step);
                const y = scaleY(point.loss);
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            ctx.stroke();

            // Draw gradient fill for best trial
            if (isBest && trial.points.length > 1) {
                const gradient = ctx.createLinearGradient(0, margin.top, 0, height - margin.bottom);
                gradient.addColorStop(0, 'rgba(16, 185, 129, 0.2)');
                gradient.addColorStop(1, 'rgba(16, 185, 129, 0)');

                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.moveTo(scaleX(trial.points[0].step), height - margin.bottom);
                trial.points.forEach(point => {
                    ctx.lineTo(scaleX(point.step), scaleY(point.loss));
                });
                ctx.lineTo(scaleX(trial.points[trial.points.length - 1].step), height - margin.bottom);
                ctx.closePath();
                ctx.fill();
            }

            // Reset alpha
            ctx.globalAlpha = 1;

            // Draw latest point marker for current trial
            if (isCurrent && trial.points.length > 0) {
                const lastPoint = trial.points[trial.points.length - 1];
                const x = scaleX(lastPoint.step);
                const y = scaleY(lastPoint.loss);

                // Pulsing dot
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fill();

                // Outer ring (animated)
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.globalAlpha = 0.5;
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, Math.PI * 2);
                ctx.stroke();
                ctx.globalAlpha = 1;
            }
        });

    }, [dimensions, trials, computeBounds, getTrialColor, bestTrialId, currentTrialId]);

    // Handle resize
    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width } = entry.contentRect;
                setDimensions({ width, height });
            }
        });

        resizeObserver.observe(container);
        return () => resizeObserver.disconnect();
    }, [height]);

    // Draw on data change
    useEffect(() => {
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
        }
        animationFrameRef.current = requestAnimationFrame(draw);

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, [draw]);

    // Mouse hover handling
    const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Find nearest point
        const margin = { top: 20, right: 20, bottom: 40, left: 60 };
        const { minLoss, maxLoss, maxStep } = computeBounds();
        const chartWidth = dimensions.width - margin.left - margin.right;
        const chartHeight = dimensions.height - margin.top - margin.bottom;

        let nearestPoint: typeof hoveredPoint = null;
        let minDist = 20; // Max hover distance in pixels

        trials.forEach((trial) => {
            trial.points.forEach((point) => {
                const px = margin.left + (point.step / maxStep) * chartWidth;
                const py = margin.top + chartHeight - ((point.loss - minLoss) / (maxLoss - minLoss)) * chartHeight;
                const dist = Math.sqrt((x - px) ** 2 + (y - py) ** 2);

                if (dist < minDist) {
                    minDist = dist;
                    nearestPoint = { x: px, y: py, point, trial };
                }
            });
        });

        setHoveredPoint(nearestPoint);
    }, [computeBounds, dimensions, trials]);

    const handleMouseLeave = useCallback(() => {
        setHoveredPoint(null);
    }, []);

    return (
        <div className="loss-chart" ref={containerRef}>
            <div className="loss-chart__header">
                <h4 className="loss-chart__title">Loss Convergence</h4>
                {showLegend && trials.length > 0 && (
                    <div className="loss-chart__legend">
                        {trials.slice(0, 5).map((trial, i) => (
                            <div key={trial.trialId} className="loss-chart__legend-item">
                                <span
                                    className="loss-chart__legend-color"
                                    style={{ backgroundColor: getTrialColor(trial, i) }}
                                />
                                <span className="loss-chart__legend-label">
                                    Trial {trial.trialId}
                                    {trial.trialId === bestTrialId && ' ★'}
                                </span>
                            </div>
                        ))}
                        {trials.length > 5 && (
                            <span className="loss-chart__legend-more">
                                +{trials.length - 5} more
                            </span>
                        )}
                    </div>
                )}
            </div>
            <div className="loss-chart__canvas-container">
                <canvas
                    ref={canvasRef}
                    className="loss-chart__canvas"
                    style={{ width: dimensions.width, height: dimensions.height }}
                    onMouseMove={handleMouseMove}
                    onMouseLeave={handleMouseLeave}
                />
                {hoveredPoint && (
                    <div
                        className="loss-chart__tooltip"
                        style={{
                            left: hoveredPoint.x,
                            top: hoveredPoint.y - 10,
                        }}
                    >
                        <div className="loss-chart__tooltip-content">
                            <span className="loss-chart__tooltip-trial">
                                Trial {hoveredPoint.trial.trialId}
                                {hoveredPoint.trial.trialId === bestTrialId && ' ★'}
                            </span>
                            <span className="loss-chart__tooltip-value">
                                Loss: {hoveredPoint.point.loss.toFixed(6)}
                            </span>
                            <span className="loss-chart__tooltip-step">
                                Step: {hoveredPoint.point.step}
                            </span>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default LossChart;
