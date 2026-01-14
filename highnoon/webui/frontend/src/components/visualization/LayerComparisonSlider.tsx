// LayerComparisonSlider.tsx - Side-by-side/overlay comparison UI
// Split-screen view with draggable divider for comparing layers

import { useState, useCallback, useRef, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { Layers, GitCompare, Minus, Eye, EyeOff } from 'lucide-react';
import { colorscales, type ColorscaleName } from './colorscales';
import type { SurfaceData } from './TensorSurfaceViz';

import './LayerComparisonSlider.css';

type ComparisonMode = 'split' | 'overlay' | 'difference';

interface LayerComparisonSliderProps {
    layerA: SurfaceData | null;
    layerB: SurfaceData | null;
    colorscale?: ColorscaleName;
    height?: number;
    enabled?: boolean;
}

/**
 * LayerComparisonSlider - Layer comparison visualization
 *
 * Features:
 * - Split-screen view with draggable divider
 * - Synchronized camera controls
 * - Overlay mode for direct comparison
 * - Difference visualization (subtract layer B from A)
 */
export function LayerComparisonSlider({
    layerA,
    layerB,
    colorscale = 'viridis',
    height = 400,
    enabled = true,
}: LayerComparisonSliderProps) {
    const [mode, setMode] = useState<ComparisonMode>('split');
    const [splitPosition, setSplitPosition] = useState(50); // percentage
    const [isDragging, setIsDragging] = useState(false);
    const [showLayerA, setShowLayerA] = useState(true);
    const [showLayerB, setShowLayerB] = useState(true);
    const containerRef = useRef<HTMLDivElement>(null);

    const colorFn = colorscales[colorscale] ?? colorscales.viridis;

    // Handle split divider drag
    const handleMouseDown = useCallback(() => {
        setIsDragging(true);
    }, []);

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
        if (!isDragging || !containerRef.current) return;

        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percentage = Math.max(10, Math.min(90, (x / rect.width) * 100));
        setSplitPosition(percentage);
    }, [isDragging]);

    const handleMouseUp = useCallback(() => {
        setIsDragging(false);
    }, []);

    // Build surface geometry from data
    const buildGeometry = useCallback((data: SurfaceData) => {
        const size = data.x.length;
        const geom = new THREE.PlaneGeometry(10, 10, size - 1, size - 1);
        const positions = geom.attributes.position;
        const colors = new Float32Array(positions.count * 3);

        const { min, max } = data.stats;
        const range = max - min || 1;

        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const idx = i * size + j;
                const z = data.z[i]?.[j] ?? 0;

                positions.setZ(idx, ((z - min) / range) * 3);

                const normalized = (z - min) / range;
                const [r, g, b] = colorFn(normalized);
                colors[idx * 3] = r;
                colors[idx * 3 + 1] = g;
                colors[idx * 3 + 2] = b;
            }
        }

        geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geom.computeVertexNormals();
        return geom;
    }, [colorFn]);

    // Compute difference surface (A - B)
    const differenceData = useMemo(() => {
        if (!layerA || !layerB) return null;

        const sizeA = layerA.z.length;
        const sizeB = layerB.z.length;
        const size = Math.min(sizeA, sizeB);

        const diffZ: number[][] = [];
        let diffMin = Infinity;
        let diffMax = -Infinity;
        let diffSum = 0;

        for (let i = 0; i < size; i++) {
            diffZ[i] = [];
            for (let j = 0; j < size; j++) {
                const valA = layerA.z[i]?.[j] ?? 0;
                const valB = layerB.z[i]?.[j] ?? 0;
                const diff = valA - valB;

                diffZ[i][j] = diff;
                diffMin = Math.min(diffMin, diff);
                diffMax = Math.max(diffMax, diff);
                diffSum += diff;
            }
        }

        const mean = diffSum / (size * size);
        let variance = 0;
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                variance += Math.pow(diffZ[i][j] - mean, 2);
            }
        }
        const std = Math.sqrt(variance / (size * size));

        return {
            x: Array.from({ length: size }, (_, i) => i),
            y: Array.from({ length: size }, (_, i) => i),
            z: diffZ,
            colorscale: 'plasma' as ColorscaleName,
            layer_name: `${layerA.layer_name} - ${layerB.layer_name}`,
            stats: { min: diffMin, max: diffMax, mean, std },
        } as SurfaceData;
    }, [layerA, layerB]);

    // Build difference geometry with diverging colorscale
    const buildDifferenceGeometry = useCallback((data: SurfaceData) => {
        const size = data.x.length;
        const geom = new THREE.PlaneGeometry(10, 10, size - 1, size - 1);
        const positions = geom.attributes.position;
        const colors = new Float32Array(positions.count * 3);

        const { min, max } = data.stats;
        const absMax = Math.max(Math.abs(min), Math.abs(max)) || 1;

        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const idx = i * size + j;
                const z = data.z[i]?.[j] ?? 0;

                // Height based on absolute value
                positions.setZ(idx, (Math.abs(z) / absMax) * 3);

                // Diverging colorscale: blue (negative) -> white (zero) -> red (positive)
                const normalized = z / absMax; // -1 to 1

                if (normalized < 0) {
                    // Blue for negative
                    colors[idx * 3] = 0.2 + Math.abs(normalized) * 0.1;
                    colors[idx * 3 + 1] = 0.4 + Math.abs(normalized) * 0.2;
                    colors[idx * 3 + 2] = 0.8 + Math.abs(normalized) * 0.2;
                } else {
                    // Red for positive
                    colors[idx * 3] = 0.8 + normalized * 0.2;
                    colors[idx * 3 + 1] = 0.3 - normalized * 0.2;
                    colors[idx * 3 + 2] = 0.2;
                }
            }
        }

        geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geom.computeVertexNormals();
        return geom;
    }, []);

    const geometryA = useMemo(() => layerA ? buildGeometry(layerA) : null, [layerA, buildGeometry]);
    const geometryB = useMemo(() => layerB ? buildGeometry(layerB) : null, [layerB, buildGeometry]);
    const geometryDiff = useMemo(
        () => differenceData ? buildDifferenceGeometry(differenceData) : null,
        [differenceData, buildDifferenceGeometry]
    );

    if (!enabled || (!layerA && !layerB)) {
        return (
            <div className="layer-comparison layer-comparison--empty" style={{ height }}>
                <span>Select two layers to compare</span>
            </div>
        );
    }

    const modes: { id: ComparisonMode; label: string; icon: React.ReactNode }[] = [
        { id: 'split', label: 'Split', icon: <GitCompare size={14} /> },
        { id: 'overlay', label: 'Overlay', icon: <Layers size={14} /> },
        { id: 'difference', label: 'Diff', icon: <Minus size={14} /> },
    ];

    return (
        <div className="layer-comparison">
            {/* Header */}
            <div className="layer-comparison__header">
                <div className="layer-comparison__title">
                    <GitCompare size={16} />
                    <span>Layer Comparison</span>
                </div>

                {/* Mode selector */}
                <div className="layer-comparison__modes">
                    {modes.map(m => (
                        <button
                            key={m.id}
                            className={`comparison-mode-btn ${mode === m.id ? 'comparison-mode-btn--active' : ''}`}
                            onClick={() => setMode(m.id)}
                        >
                            {m.icon}
                            <span>{m.label}</span>
                        </button>
                    ))}
                </div>

                {/* Layer visibility toggles (for overlay mode) */}
                {mode === 'overlay' && (
                    <div className="layer-comparison__toggles">
                        <button
                            className={`layer-toggle ${showLayerA ? 'layer-toggle--active' : ''}`}
                            onClick={() => setShowLayerA(!showLayerA)}
                        >
                            {showLayerA ? <Eye size={14} /> : <EyeOff size={14} />}
                            <span>A</span>
                        </button>
                        <button
                            className={`layer-toggle ${showLayerB ? 'layer-toggle--active' : ''}`}
                            onClick={() => setShowLayerB(!showLayerB)}
                        >
                            {showLayerB ? <Eye size={14} /> : <EyeOff size={14} />}
                            <span>B</span>
                        </button>
                    </div>
                )}
            </div>

            {/* Canvas area */}
            <div
                ref={containerRef}
                className="layer-comparison__canvas"
                style={{ height }}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
            >
                {mode === 'split' ? (
                    <>
                        {/* Layer A (left side) */}
                        <div
                            className="layer-comparison__pane layer-comparison__pane--left"
                            style={{ width: `${splitPosition}%` }}
                        >
                            <Canvas
                                gl={{ preserveDrawingBuffer: true, antialias: true }}
                                camera={{ position: [8, 8, 8], fov: 45 }}
                            >
                                <ambientLight intensity={0.4} />
                                <directionalLight position={[10, 10, 5]} intensity={0.8} />
                                {geometryA && (
                                    <mesh geometry={geometryA} rotation={[-Math.PI / 3, 0, 0]}>
                                        <meshStandardMaterial
                                            vertexColors
                                            side={THREE.DoubleSide}
                                        />
                                    </mesh>
                                )}
                                <OrbitControls enableDamping />
                            </Canvas>
                            <div className="layer-comparison__label">
                                {layerA?.layer_name || 'Layer A'}
                            </div>
                        </div>

                        {/* Divider */}
                        <div
                            className="layer-comparison__divider"
                            style={{ left: `${splitPosition}%` }}
                            onMouseDown={handleMouseDown}
                        >
                            <div className="layer-comparison__divider-handle" />
                        </div>

                        {/* Layer B (right side) */}
                        <div
                            className="layer-comparison__pane layer-comparison__pane--right"
                            style={{ width: `${100 - splitPosition}%` }}
                        >
                            <Canvas
                                gl={{ preserveDrawingBuffer: true, antialias: true }}
                                camera={{ position: [8, 8, 8], fov: 45 }}
                            >
                                <ambientLight intensity={0.4} />
                                <directionalLight position={[10, 10, 5]} intensity={0.8} />
                                {geometryB && (
                                    <mesh geometry={geometryB} rotation={[-Math.PI / 3, 0, 0]}>
                                        <meshStandardMaterial
                                            vertexColors
                                            side={THREE.DoubleSide}
                                        />
                                    </mesh>
                                )}
                                <OrbitControls enableDamping />
                            </Canvas>
                            <div className="layer-comparison__label">
                                {layerB?.layer_name || 'Layer B'}
                            </div>
                        </div>
                    </>
                ) : mode === 'overlay' ? (
                    <Canvas
                        gl={{ preserveDrawingBuffer: true, antialias: true }}
                        camera={{ position: [8, 8, 8], fov: 45 }}
                    >
                        <ambientLight intensity={0.4} />
                        <directionalLight position={[10, 10, 5]} intensity={0.8} />

                        {/* Layer A */}
                        {showLayerA && geometryA && (
                            <mesh geometry={geometryA} rotation={[-Math.PI / 3, 0, 0]}>
                                <meshStandardMaterial
                                    vertexColors
                                    side={THREE.DoubleSide}
                                    transparent
                                    opacity={0.7}
                                />
                            </mesh>
                        )}

                        {/* Layer B (offset slightly) */}
                        {showLayerB && geometryB && (
                            <mesh
                                geometry={geometryB}
                                rotation={[-Math.PI / 3, 0, 0]}
                                position={[0, 0.1, 0]}
                            >
                                <meshStandardMaterial
                                    vertexColors
                                    side={THREE.DoubleSide}
                                    transparent
                                    opacity={0.5}
                                    wireframe
                                />
                            </mesh>
                        )}

                        <OrbitControls enableDamping />
                    </Canvas>
                ) : (
                    /* Difference mode */
                    <Canvas
                        gl={{ preserveDrawingBuffer: true, antialias: true }}
                        camera={{ position: [8, 8, 8], fov: 45 }}
                    >
                        <ambientLight intensity={0.4} />
                        <directionalLight position={[10, 10, 5]} intensity={0.8} />

                        {geometryDiff && (
                            <mesh geometry={geometryDiff} rotation={[-Math.PI / 3, 0, 0]}>
                                <meshStandardMaterial
                                    vertexColors
                                    side={THREE.DoubleSide}
                                />
                            </mesh>
                        )}

                        <OrbitControls enableDamping />
                    </Canvas>
                )}
            </div>

            {/* Footer with stats */}
            <div className="layer-comparison__footer">
                {mode === 'difference' && differenceData && (
                    <>
                        <span className="stat">
                            <strong>Diff Range:</strong> [{differenceData.stats.min.toFixed(4)}, {differenceData.stats.max.toFixed(4)}]
                        </span>
                        <span className="stat">
                            <strong>Mean Diff:</strong> {differenceData.stats.mean.toFixed(4)}
                        </span>
                        <span className="stat">
                            <strong>Std:</strong> {differenceData.stats.std.toFixed(4)}
                        </span>
                    </>
                )}
                {mode !== 'difference' && (
                    <>
                        <span className="stat">
                            <strong>A:</strong> {layerA?.layer_name || 'None'}
                        </span>
                        <span className="stat">
                            <strong>B:</strong> {layerB?.layer_name || 'None'}
                        </span>
                    </>
                )}
            </div>
        </div>
    );
}

export default LayerComparisonSlider;
