// ActivationHeatmap.tsx - 2D Canvas Fallback for Low-Performance Devices
// No WebGL required - uses HTML5 Canvas for heatmap rendering

import { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { viridisColor, plasmaColor, infernoColor, type ColorscaleName } from './colorscales';
import './ActivationHeatmap.css';

/** Activation data structure */
interface HeatmapData {
    z: number[][];
    layer_name: string;
    stats: {
        min: number;
        max: number;
        mean: number;
        std: number;
    };
}

/** Props for ActivationHeatmap component */
interface ActivationHeatmapProps {
    /** Activation data */
    data: HeatmapData | null;
    /** Loading state */
    loading?: boolean;
    /** Container height */
    height?: number;
    /** Initial colorscale */
    colorscale?: ColorscaleName;
    /** Export callback */
    onExport?: (dataUrl: string) => void;
}

/** Get colorscale function by name */
function getColorFn(name: ColorscaleName) {
    switch (name) {
        case 'plasma': return plasmaColor;
        case 'inferno': return infernoColor;
        default: return viridisColor;
    }
}

/** Check if WebGL is available */
function checkWebGLSupport(): boolean {
    try {
        const canvas = document.createElement('canvas');
        return !!(
            window.WebGLRenderingContext &&
            (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
        );
    } catch {
        return false;
    }
}

/**
 * ActivationHeatmap - 2D Canvas-based Heatmap Visualization
 *
 * Fallback for devices without WebGL support:
 * - Pure Canvas 2D rendering (no Three.js)
 * - Interactive hover tooltips
 * - Colorscale selector
 * - PNG export functionality
 */
export function ActivationHeatmap({
    data,
    loading = false,
    height = 300,
    colorscale: initialColorscale = 'viridis',
    onExport,
}: ActivationHeatmapProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [colorscale, setColorscale] = useState<ColorscaleName>(initialColorscale);
    const [hoveredCell, setHoveredCell] = useState<{ x: number; y: number; value: number } | null>(null);
    const [isExporting, setIsExporting] = useState(false);
    const [webglAvailable] = useState(() => checkWebGLSupport());

    // Render heatmap to canvas
    useEffect(() => {
        if (!canvasRef.current || !data) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const rows = data.z.length;
        const cols = data.z[0]?.length ?? 0;
        if (rows === 0 || cols === 0) return;

        // Set canvas size
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const cellWidth = rect.width / cols;
        const cellHeight = rect.height / rows;

        const { min, max } = data.stats;
        const range = max - min || 1;
        const colorFn = getColorFn(colorscale);

        // Clear canvas
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, rect.width, rect.height);

        // Draw cells
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const value = data.z[i]?.[j] ?? 0;
                const normalized = (value - min) / range;
                const [r, g, b] = colorFn(normalized);

                ctx.fillStyle = `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
                ctx.fillRect(j * cellWidth, i * cellHeight, cellWidth + 0.5, cellHeight + 0.5);
            }
        }
    }, [data, colorscale]);

    // Handle mouse move for hover tooltip
    const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!canvasRef.current || !data) return;

        const rect = canvasRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const rows = data.z.length;
        const cols = data.z[0]?.length ?? 0;

        const col = Math.floor(x / (rect.width / cols));
        const row = Math.floor(y / (rect.height / rows));

        if (row >= 0 && row < rows && col >= 0 && col < cols) {
            setHoveredCell({
                x: col,
                y: row,
                value: data.z[row][col],
            });
        }
    }, [data]);

    const handleMouseLeave = useCallback(() => {
        setHoveredCell(null);
    }, []);

    // Export to PNG
    const handleExport = useCallback(() => {
        if (!canvasRef.current || !onExport) return;
        setIsExporting(true);

        requestAnimationFrame(() => {
            const dataUrl = canvasRef.current?.toDataURL('image/png');
            if (dataUrl) onExport(dataUrl);
            setIsExporting(false);
        });
    }, [onExport]);

    // Generate colorscale legend gradient
    const gradientStyle = useMemo(() => {
        const colorFn = getColorFn(colorscale);
        const stops: string[] = [];
        for (let i = 0; i <= 10; i++) {
            const t = i / 10;
            const [r, g, b] = colorFn(t);
            stops.push(`rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)}) ${t * 100}%`);
        }
        return { background: `linear-gradient(to right, ${stops.join(', ')})` };
    }, [colorscale]);

    if (loading) {
        return (
            <div className="activation-heatmap activation-heatmap--loading" style={{ minHeight: height }}>
                <div className="heatmap-spinner" />
                <span>Loading activation data...</span>
            </div>
        );
    }

    if (!data) {
        return (
            <div className="activation-heatmap activation-heatmap--empty" style={{ minHeight: height }}>
                <span>No activation data available</span>
                <p>Start training to see tensor activations</p>
            </div>
        );
    }

    return (
        <div className="activation-heatmap" ref={containerRef}>
            <div className="activation-heatmap__header">
                <div className="activation-heatmap__title">
                    <span className="heatmap-badge">2D</span>
                    <span className="layer-name">{data.layer_name}</span>
                    Activation Heatmap
                </div>
                <div className="activation-heatmap__controls">
                    {!webglAvailable && (
                        <span className="webgl-warning" title="WebGL not available, using 2D fallback">
                            ⚠️ 2D Mode
                        </span>
                    )}
                    <select
                        value={colorscale}
                        onChange={(e) => setColorscale(e.target.value as ColorscaleName)}
                        className="colorscale-select"
                    >
                        <option value="viridis">Viridis</option>
                        <option value="plasma">Plasma</option>
                        <option value="inferno">Inferno</option>
                    </select>
                    {onExport && (
                        <button
                            onClick={handleExport}
                            disabled={isExporting}
                            className="heatmap-export-btn"
                        >
                            📷 {isExporting ? '...' : 'Export'}
                        </button>
                    )}
                </div>
            </div>

            <div className="activation-heatmap__canvas-container" style={{ height }}>
                <canvas
                    ref={canvasRef}
                    className="activation-heatmap__canvas"
                    onMouseMove={handleMouseMove}
                    onMouseLeave={handleMouseLeave}
                />

                {/* Hover tooltip */}
                {hoveredCell && (
                    <div className="heatmap-tooltip">
                        <span className="tooltip-coord">[{hoveredCell.y}, {hoveredCell.x}]</span>
                        <span className="tooltip-value">{hoveredCell.value.toFixed(4)}</span>
                    </div>
                )}
            </div>

            <div className="activation-heatmap__footer">
                <div className="heatmap-legend">
                    <span className="legend-label">{data.stats.min.toFixed(3)}</span>
                    <div className="legend-gradient" style={gradientStyle} />
                    <span className="legend-label">{data.stats.max.toFixed(3)}</span>
                </div>
                <div className="heatmap-stats">
                    <span>μ = {data.stats.mean.toFixed(4)}</span>
                    <span>σ = {data.stats.std.toFixed(4)}</span>
                    <span>{data.z.length}×{data.z[0]?.length || 0}</span>
                </div>
            </div>
        </div>
    );
}

export default ActivationHeatmap;
