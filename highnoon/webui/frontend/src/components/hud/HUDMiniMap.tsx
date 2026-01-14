// HUDMiniMap.tsx - Radar-style overview of training progress
// Displays trial positions in a circular mini-map format

import React, { useMemo, useRef, useEffect } from 'react';
import './HUDMiniMap.css';

interface MiniMapPoint {
    id: string | number;
    /** Normalized x position (0-1) */
    x: number;
    /** Normalized y position (0-1) */
    y: number;
    /** Point status */
    status: 'completed' | 'running' | 'pending' | 'best';
    /** Optional value for sizing */
    value?: number;
}

interface HUDMiniMapProps {
    /** Points to display on the map */
    points: MiniMapPoint[];
    /** Current user/focus position */
    currentPosition?: { x: number; y: number };
    /** Best position marker */
    bestPosition?: { x: number; y: number };
    /** Title label */
    title?: string;
    /** Size in pixels */
    size?: number;
    /** Show radar sweep animation */
    showRadarSweep?: boolean;
    /** Show grid lines */
    showGrid?: boolean;
    /** Click handler to expand to full view */
    onExpand?: () => void;
    /** Point click handler */
    onPointClick?: (point: MiniMapPoint) => void;
    /** Container className */
    className?: string;
}

const STATUS_COLORS: Record<string, string> = {
    completed: '#22c55e',
    running: '#00f5ff',
    pending: '#6b7280',
    best: '#fbbf24',
};

/**
 * HUDMiniMap - Radar-style circular display for trial overview
 *
 * Features:
 * - Circular radar display with grid
 * - Points color-coded by status
 * - Current position indicator
 * - Best position star marker
 * - Optional radar sweep animation
 * - Click to expand to full scatter plot
 */
export function HUDMiniMap({
    points,
    currentPosition,
    bestPosition,
    title = 'RADAR',
    size = 150,
    showRadarSweep = true,
    showGrid = true,
    onExpand,
    onPointClick,
    className = '',
}: HUDMiniMapProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationRef = useRef<number>(0);
    const sweepAngleRef = useRef(0);

    // Draw the mini-map
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const canvasSize = size * dpr;
        canvas.width = canvasSize;
        canvas.height = canvasSize;
        ctx.scale(dpr, dpr);

        const center = size / 2;
        const radius = (size / 2) - 10;

        const draw = () => {
            // Clear canvas
            ctx.clearRect(0, 0, size, size);

            // Draw background
            ctx.beginPath();
            ctx.arc(center, center, radius, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(10, 10, 15, 0.9)';
            ctx.fill();

            // Draw grid rings
            if (showGrid) {
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
                ctx.lineWidth = 1;

                [0.25, 0.5, 0.75, 1].forEach(scale => {
                    ctx.beginPath();
                    ctx.arc(center, center, radius * scale, 0, Math.PI * 2);
                    ctx.stroke();
                });

                // Draw cross lines
                ctx.beginPath();
                ctx.moveTo(center - radius, center);
                ctx.lineTo(center + radius, center);
                ctx.moveTo(center, center - radius);
                ctx.lineTo(center, center + radius);
                ctx.stroke();
            }

            // Draw radar sweep
            if (showRadarSweep) {
                sweepAngleRef.current = (sweepAngleRef.current + 0.02) % (Math.PI * 2);
                const gradient = ctx.createConicGradient(sweepAngleRef.current - Math.PI / 2, center, center);
                gradient.addColorStop(0, 'rgba(0, 245, 255, 0.3)');
                gradient.addColorStop(0.1, 'rgba(0, 245, 255, 0)');
                gradient.addColorStop(1, 'rgba(0, 245, 255, 0)');

                ctx.beginPath();
                ctx.arc(center, center, radius, 0, Math.PI * 2);
                ctx.fillStyle = gradient;
                ctx.fill();
            }

            // Draw outer ring glow
            ctx.beginPath();
            ctx.arc(center, center, radius, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(0, 245, 255, 0.3)';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Draw points
            points.forEach(point => {
                const px = center + (point.x - 0.5) * radius * 1.8;
                const py = center + (point.y - 0.5) * radius * 1.8;

                // Check if point is within radius
                const dist = Math.sqrt((px - center) ** 2 + (py - center) ** 2);
                if (dist > radius) return;

                const pointSize = point.status === 'best' ? 5 : 3;
                const color = STATUS_COLORS[point.status] || STATUS_COLORS.pending;

                ctx.beginPath();
                ctx.arc(px, py, pointSize, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();

                // Glow for running/best
                if (point.status === 'running' || point.status === 'best') {
                    ctx.beginPath();
                    ctx.arc(px, py, pointSize + 3, 0, Math.PI * 2);
                    ctx.fillStyle = color.replace(')', ', 0.3)').replace('rgb', 'rgba');
                    ctx.fill();
                }
            });

            // Draw current position
            if (currentPosition) {
                const cx = center + (currentPosition.x - 0.5) * radius * 1.8;
                const cy = center + (currentPosition.y - 0.5) * radius * 1.8;

                // Crosshair
                ctx.strokeStyle = '#00f5ff';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(cx - 6, cy);
                ctx.lineTo(cx + 6, cy);
                ctx.moveTo(cx, cy - 6);
                ctx.lineTo(cx, cy + 6);
                ctx.stroke();

                // Center dot
                ctx.beginPath();
                ctx.arc(cx, cy, 2, 0, Math.PI * 2);
                ctx.fillStyle = '#00f5ff';
                ctx.fill();
            }

            // Draw best position star
            if (bestPosition) {
                const bx = center + (bestPosition.x - 0.5) * radius * 1.8;
                const by = center + (bestPosition.y - 0.5) * radius * 1.8;

                // Draw star shape
                ctx.fillStyle = '#fbbf24';
                ctx.beginPath();
                for (let i = 0; i < 5; i++) {
                    const angle = (i * 4 * Math.PI) / 5 - Math.PI / 2;
                    const r = i % 2 === 0 ? 5 : 2.5;
                    const sx = bx + Math.cos(angle) * r;
                    const sy = by + Math.sin(angle) * r;
                    if (i === 0) ctx.moveTo(sx, sy);
                    else ctx.lineTo(sx, sy);
                }
                ctx.closePath();
                ctx.fill();
            }

            animationRef.current = requestAnimationFrame(draw);
        };

        draw();

        return () => {
            cancelAnimationFrame(animationRef.current);
        };
    }, [points, currentPosition, bestPosition, size, showGrid, showRadarSweep]);

    // Stats summary
    const stats = useMemo(() => {
        const completed = points.filter(p => p.status === 'completed').length;
        const running = points.filter(p => p.status === 'running').length;
        const pending = points.filter(p => p.status === 'pending').length;
        return { completed, running, pending, total: points.length };
    }, [points]);

    return (
        <div
            className={`hud-minimap ${className}`}
            style={{ width: size, height: size + 40 }}
            onClick={onExpand}
        >
            <div className="hud-minimap__header">
                <span className="hud-minimap__title">{title}</span>
                <span className="hud-minimap__count">{stats.total}</span>
            </div>

            <canvas
                ref={canvasRef}
                className="hud-minimap__canvas"
                style={{ width: size, height: size }}
            />

            <div className="hud-minimap__footer">
                <span className="hud-minimap__stat hud-minimap__stat--completed">
                    {stats.completed}
                </span>
                <span className="hud-minimap__stat hud-minimap__stat--running">
                    {stats.running}
                </span>
                <span className="hud-minimap__stat hud-minimap__stat--pending">
                    {stats.pending}
                </span>
            </div>

            {onExpand && (
                <div className="hud-minimap__expand-hint">
                    Click to expand
                </div>
            )}
        </div>
    );
}

export default HUDMiniMap;
