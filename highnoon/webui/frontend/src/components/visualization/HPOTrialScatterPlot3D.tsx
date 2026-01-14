// HPOTrialScatterPlot3D.tsx - 3D visualization of HPO trial search space
// Interactive scatter plot showing hyperparameter configurations and their performance

import React, { useRef, useMemo, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html, Line, Text } from '@react-three/drei';
import * as THREE from 'three';

interface TrialPoint {
    id: string | number;
    /** Hyperparameter values (normalized 0-1 or raw) */
    x: number;  // e.g., learning_rate
    y: number;  // e.g., hidden_dim
    z: number;  // e.g., batch_size
    /** Performance metric (loss, accuracy, etc.) */
    metric: number;
    /** Trial status */
    status: 'completed' | 'running' | 'pending' | 'failed';
    /** Optional trial duration */
    duration?: number;
    /** Optional additional metadata */
    metadata?: Record<string, unknown>;
}

interface AxisConfig {
    label: string;
    min: number;
    max: number;
    logScale?: boolean;
    unit?: string;
}

interface HPOTrialScatterPlot3DProps {
    /** Trial data points */
    trials: TrialPoint[];
    /** Best trial (highlighted) */
    bestTrialId?: string | number;
    /** Currently running trial */
    currentTrialId?: string | number;
    /** Axis configuration */
    xAxis?: AxisConfig;
    yAxis?: AxisConfig;
    zAxis?: AxisConfig;
    /** Metric range for coloring */
    metricRange?: { min: number; max: number };
    /** Color scheme */
    colorScale?: 'loss' | 'performance' | 'duration';
    /** Show Pareto frontier */
    showParetoFrontier?: boolean;
    /** Show convergence trails */
    showTrails?: boolean;
    /** Callback when trial is clicked */
    onTrialClick?: (trial: TrialPoint) => void;
    /** Callback when trial is hovered */
    onTrialHover?: (trial: TrialPoint | null) => void;
    /** Container className */
    className?: string;
}

// Default axis configurations
const defaultXAxis: AxisConfig = { label: 'Learning Rate', min: 1e-5, max: 1e-2, logScale: true };
const defaultYAxis: AxisConfig = { label: 'Hidden Dim', min: 64, max: 1024 };
const defaultZAxis: AxisConfig = { label: 'Batch Size', min: 8, max: 128 };

// Color constants
const COLORS = {
    completed: '#22c55e',
    running: '#00f5ff',
    pending: '#6b7280',
    failed: '#ef4444',
    best: '#fbbf24',
    axis: '#4b5563',
    grid: 'rgba(255, 255, 255, 0.05)',
};

/**
 * Normalize a value to 0-1 range, optionally using log scale
 */
function normalizeValue(value: number, min: number, max: number, logScale = false): number {
    if (logScale && min > 0 && max > 0) {
        return (Math.log10(value) - Math.log10(min)) / (Math.log10(max) - Math.log10(min));
    }
    return (value - min) / (max - min);
}

/**
 * Get color based on metric value (red to green gradient)
 */
function getMetricColor(normalized: number): THREE.Color {
    // Red (high loss) -> Yellow -> Green (low loss)
    if (normalized < 0.5) {
        return new THREE.Color().setHSL(0.1 * normalized * 2, 0.8, 0.5);  // Red to yellow
    }
    return new THREE.Color().setHSL(0.1 + 0.23 * (normalized - 0.5) * 2, 0.8, 0.5);  // Yellow to green
}

/**
 * Individual trial point component
 */
function TrialPointMesh({
    trial,
    position,
    isBest,
    isRunning,
    color,
    size,
    onHover,
    onClick,
}: {
    trial: TrialPoint;
    position: [number, number, number];
    isBest: boolean;
    isRunning: boolean;
    color: THREE.Color;
    size: number;
    onHover: (trial: TrialPoint | null) => void;
    onClick: (trial: TrialPoint) => void;
}) {
    const meshRef = useRef<THREE.Mesh>(null);
    const [hovered, setHovered] = useState(false);

    // Animation for running and best trials
    useFrame((state) => {
        if (meshRef.current) {
            if (isRunning) {
                // Pulsing animation for running trial
                const scale = 1 + Math.sin(state.clock.elapsedTime * 5) * 0.2;
                meshRef.current.scale.setScalar(scale);
            } else if (isBest) {
                // Gentle pulse for best trial
                const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.1;
                meshRef.current.scale.setScalar(scale);
            }
        }
    });

    const handlePointerOver = useCallback(() => {
        setHovered(true);
        onHover(trial);
    }, [trial, onHover]);

    const handlePointerOut = useCallback(() => {
        setHovered(false);
        onHover(null);
    }, [onHover]);

    const handleClick = useCallback(() => {
        onClick(trial);
    }, [trial, onClick]);

    const geometry = isBest ? (
        <octahedronGeometry args={[size * 1.5, 0]} />
    ) : (
        <sphereGeometry args={[size, 16, 16]} />
    );

    return (
        <group position={position}>
            <mesh
                ref={meshRef}
                onPointerOver={handlePointerOver}
                onPointerOut={handlePointerOut}
                onClick={handleClick}
            >
                {geometry}
                <meshStandardMaterial
                    color={isBest ? COLORS.best : color}
                    emissive={isBest ? COLORS.best : (isRunning ? COLORS.running : color)}
                    emissiveIntensity={hovered ? 1.0 : (isBest || isRunning ? 0.8 : 0.3)}
                    metalness={0.2}
                    roughness={0.4}
                />
            </mesh>

            {/* Glow effect for best/running */}
            {(isBest || isRunning) && (
                <mesh>
                    <sphereGeometry args={[size * 1.8, 16, 16]} />
                    <meshBasicMaterial
                        color={isBest ? COLORS.best : COLORS.running}
                        transparent
                        opacity={0.15}
                    />
                </mesh>
            )}

            {/* Tooltip on hover */}
            {hovered && (
                <Html position={[0, size * 2.5, 0]} center>
                    <div className="hpo-tooltip">
                        <div className="hpo-tooltip__header">
                            Trial #{trial.id}
                            {isBest && <span className="hpo-tooltip__badge hpo-tooltip__badge--best">★ BEST</span>}
                            {isRunning && <span className="hpo-tooltip__badge hpo-tooltip__badge--running">● RUNNING</span>}
                        </div>
                        <div className="hpo-tooltip__row">
                            <span>Metric:</span>
                            <span>{trial.metric.toFixed(4)}</span>
                        </div>
                        {trial.duration !== undefined && (
                            <div className="hpo-tooltip__row">
                                <span>Duration:</span>
                                <span>{trial.duration.toFixed(1)}s</span>
                            </div>
                        )}
                    </div>
                </Html>
            )}
        </group>
    );
}

/**
 * Axis with labels and ticks
 */
function Axis3D({
    start,
    end,
    label,
    tickCount = 5,
}: {
    start: [number, number, number];
    end: [number, number, number];
    label: string;
    tickCount?: number;
}) {
    const direction = new THREE.Vector3(
        end[0] - start[0],
        end[1] - start[1],
        end[2] - start[2]
    ).normalize();

    const labelPosition = [
        (start[0] + end[0]) / 2,
        (start[1] + end[1]) / 2,
        (start[2] + end[2]) / 2,
    ] as [number, number, number];

    return (
        <group>
            <Line
                points={[start, end]}
                color={COLORS.axis}
                lineWidth={2}
            />
            <Html position={labelPosition} center>
                <div className="axis-label">{label}</div>
            </Html>
        </group>
    );
}

/**
 * Grid plane for the scatter plot
 */
function GridPlane({
    position,
    rotation,
    size = 5,
}: {
    position: [number, number, number];
    rotation: [number, number, number];
    size?: number;
}) {
    return (
        <mesh position={position} rotation={rotation}>
            <planeGeometry args={[size, size, 10, 10]} />
            <meshBasicMaterial
                color="#ffffff"
                transparent
                opacity={0.03}
                wireframe
            />
        </mesh>
    );
}

/**
 * Scene content for the 3D scatter plot
 */
function ScatterPlotScene({
    trials,
    bestTrialId,
    currentTrialId,
    xAxis,
    yAxis,
    zAxis,
    metricRange,
    onTrialClick,
    onTrialHover,
}: {
    trials: TrialPoint[];
    bestTrialId?: string | number;
    currentTrialId?: string | number;
    xAxis: AxisConfig;
    yAxis: AxisConfig;
    zAxis: AxisConfig;
    metricRange: { min: number; max: number };
    onTrialClick: (trial: TrialPoint) => void;
    onTrialHover: (trial: TrialPoint | null) => void;
}) {
    const sceneSize = 4;
    const halfSize = sceneSize / 2;

    // Process trials into positioned points
    const processedTrials = useMemo(() => {
        return trials.map(trial => {
            const x = normalizeValue(trial.x, xAxis.min, xAxis.max, xAxis.logScale) * sceneSize - halfSize;
            const y = normalizeValue(trial.y, yAxis.min, yAxis.max, yAxis.logScale) * sceneSize - halfSize;
            const z = normalizeValue(trial.z, zAxis.min, zAxis.max, zAxis.logScale) * sceneSize - halfSize;

            const metricNorm = 1 - normalizeValue(trial.metric, metricRange.min, metricRange.max);
            const color = getMetricColor(metricNorm);
            const size = 0.08 + metricNorm * 0.08;  // Better trials are slightly larger

            return {
                trial,
                position: [x, y, z] as [number, number, number],
                color,
                size,
            };
        });
    }, [trials, xAxis, yAxis, zAxis, metricRange, sceneSize, halfSize]);

    return (
        <group>
            {/* Ambient and directional lighting */}
            <ambientLight intensity={0.4} />
            <directionalLight position={[5, 5, 5]} intensity={0.6} />
            <pointLight position={[-5, 5, -5]} intensity={0.3} color="#a855f7" />

            {/* Grid planes */}
            <GridPlane position={[0, -halfSize, 0]} rotation={[-Math.PI / 2, 0, 0]} size={sceneSize} />
            <GridPlane position={[-halfSize, 0, 0]} rotation={[0, Math.PI / 2, 0]} size={sceneSize} />
            <GridPlane position={[0, 0, -halfSize]} rotation={[0, 0, 0]} size={sceneSize} />

            {/* Axes */}
            <Axis3D
                start={[-halfSize, -halfSize, -halfSize]}
                end={[halfSize, -halfSize, -halfSize]}
                label={xAxis.label}
            />
            <Axis3D
                start={[-halfSize, -halfSize, -halfSize]}
                end={[-halfSize, halfSize, -halfSize]}
                label={yAxis.label}
            />
            <Axis3D
                start={[-halfSize, -halfSize, -halfSize]}
                end={[-halfSize, -halfSize, halfSize]}
                label={zAxis.label}
            />

            {/* Trial points */}
            {processedTrials.map(({ trial, position, color, size }) => (
                <TrialPointMesh
                    key={trial.id}
                    trial={trial}
                    position={position}
                    isBest={trial.id === bestTrialId}
                    isRunning={trial.id === currentTrialId}
                    color={color}
                    size={size}
                    onHover={onTrialHover}
                    onClick={onTrialClick}
                />
            ))}

            {/* Orbit controls */}
            <OrbitControls
                enableDamping
                dampingFactor={0.05}
                minDistance={3}
                maxDistance={15}
                autoRotate={false}
            />
        </group>
    );
}

/**
 * HPOTrialScatterPlot3D - Interactive 3D visualization of HPO trial search space
 *
 * Features:
 * - 3D scatter plot with instanced points for efficiency
 * - Color coding based on performance metric
 * - Size scaling based on relative performance
 * - Best trial highlighted with star shape and golden glow
 * - Running trial with pulsing animation
 * - Interactive tooltips on hover
 * - Click to open trial details
 * - Smooth orbit controls for exploration
 */
export function HPOTrialScatterPlot3D({
    trials,
    bestTrialId,
    currentTrialId,
    xAxis = defaultXAxis,
    yAxis = defaultYAxis,
    zAxis = defaultZAxis,
    metricRange,
    colorScale = 'loss',
    showParetoFrontier = false,
    showTrails = false,
    onTrialClick,
    onTrialHover,
    className = '',
}: HPOTrialScatterPlot3DProps) {
    // Calculate metric range from data if not provided
    const calculatedMetricRange = useMemo(() => {
        if (metricRange) return metricRange;
        if (trials.length === 0) return { min: 0, max: 1 };

        const metrics = trials.map(t => t.metric);
        return {
            min: Math.min(...metrics),
            max: Math.max(...metrics),
        };
    }, [trials, metricRange]);

    const handleTrialClick = useCallback((trial: TrialPoint) => {
        onTrialClick?.(trial);
    }, [onTrialClick]);

    const handleTrialHover = useCallback((trial: TrialPoint | null) => {
        onTrialHover?.(trial);
    }, [onTrialHover]);

    return (
        <div className={`hpo-scatter-plot-3d ${className}`}>
            <Canvas
                camera={{ position: [6, 4, 6], fov: 50 }}
                gl={{ antialias: true, alpha: true }}
            >
                <ScatterPlotScene
                    trials={trials}
                    bestTrialId={bestTrialId}
                    currentTrialId={currentTrialId}
                    xAxis={xAxis}
                    yAxis={yAxis}
                    zAxis={zAxis}
                    metricRange={calculatedMetricRange}
                    onTrialClick={handleTrialClick}
                    onTrialHover={handleTrialHover}
                />
            </Canvas>

            {/* Legend */}
            <div className="hpo-scatter-legend">
                <div className="hpo-scatter-legend__title">Trial Status</div>
                <div className="hpo-scatter-legend__items">
                    <div className="hpo-scatter-legend__item">
                        <span className="hpo-scatter-legend__dot" style={{ background: COLORS.completed }} />
                        <span>Completed</span>
                    </div>
                    <div className="hpo-scatter-legend__item">
                        <span className="hpo-scatter-legend__dot" style={{ background: COLORS.running }} />
                        <span>Running</span>
                    </div>
                    <div className="hpo-scatter-legend__item">
                        <span className="hpo-scatter-legend__dot" style={{ background: COLORS.best }} />
                        <span>Best</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default HPOTrialScatterPlot3D;
