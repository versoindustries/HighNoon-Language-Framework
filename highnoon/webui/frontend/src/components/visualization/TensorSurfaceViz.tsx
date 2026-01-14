// TensorSurfaceViz.tsx - 3D Surface Visualization for Tensor Activations
// Uses react-three-fiber with Three.js for WebGL rendering
// Enhanced with camera presets, video recording, and marketing mode

import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import { useMemo, useRef, useState, useCallback, useEffect } from 'react';
import * as THREE from 'three';
import { viridisColor, type ColorscaleName, colorscales } from './colorscales';
import { CAMERA_PRESETS, getPresetNames, getPreset, type CameraPreset } from './CameraPresets';
import { useVideoRecorder } from './useVideoRecorder';
import './TensorSurfaceViz.css';

/** Surface data structure returned by backend API */
export interface SurfaceData {
    x: number[];
    y: number[];
    z: number[][];
    colorscale: ColorscaleName;
    layer_name: string;
    original_shape?: [number, number];
    stats: {
        min: number;
        max: number;
        mean: number;
        std: number;
    };
}

interface TensorSurfaceVizProps {
    /** Activation surface data from API */
    data: SurfaceData | null;
    /** Loading state */
    loading?: boolean;
    /** Callback when PNG export is triggered */
    onExport?: (dataUrl: string) => void;
    /** Enable auto-rotation animation */
    autoRotate?: boolean;
    /** Height of the visualization container */
    height?: number;
    /** Colorscale to use */
    colorscale?: ColorscaleName;
    /** Show marketing mode controls */
    showMarketingControls?: boolean;
}

/** Internal component that renders the 3D surface mesh */
function Surface({
    data,
    autoRotate,
    colorscale = 'viridis',
}: {
    data: SurfaceData;
    autoRotate: boolean;
    colorscale: ColorscaleName;
}) {
    const meshRef = useRef<THREE.Mesh>(null);
    const colorFn = colorscales[colorscale] ?? viridisColor;

    // Auto-rotation effect
    useFrame((_state, delta) => {
        if (autoRotate && meshRef.current) {
            meshRef.current.rotation.z += delta * 0.1;
        }
    });

    const geometry = useMemo(() => {
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

                // Set height (z-position becomes y in Three.js orientation)
                positions.setZ(idx, ((z - min) / range) * 3);

                // Apply colorscale
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
    }, [data, colorFn]);

    return (
        <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 3, 0, 0]}>
            <meshStandardMaterial
                vertexColors
                side={THREE.DoubleSide}
                metalness={0.1}
                roughness={0.8}
            />
        </mesh>
    );
}

/**
 * 3D Tensor Surface Visualization Component
 *
 * Renders tensor activations as an interactive 3D surface plot using
 * react-three-fiber. Supports rotation, zoom, pan via OrbitControls
 * and PNG export for marketing materials.
 *
 * Enhanced with:
 * - Camera preset selector for dramatic angles
 * - Video recording via MediaRecorder API
 * - Marketing mode with auto-orbit
 */
export function TensorSurfaceViz({
    data,
    loading = false,
    onExport,
    autoRotate = false,
    height = 350,
    colorscale = 'viridis',
    showMarketingControls = true,
}: TensorSurfaceVizProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isExporting, setIsExporting] = useState(false);
    const [selectedPreset, setSelectedPreset] = useState<string>('hero');
    const [marketingMode, setMarketingMode] = useState(false);

    // Video recording hook
    const { state: recordingState, startRecording, downloadRecording } = useVideoRecorder(canvasRef);

    const handleExport = useCallback(() => {
        if (!canvasRef.current || !onExport) return;
        setIsExporting(true);

        // Delay to allow render to complete
        requestAnimationFrame(() => {
            const dataUrl = canvasRef.current?.toDataURL('image/png');
            if (dataUrl) onExport(dataUrl);
            setIsExporting(false);
        });
    }, [onExport]);

    const handleRecordToggle = useCallback(() => {
        if (recordingState.isRecording) {
            downloadRecording(`tensor_${data?.layer_name || 'activation'}_${Date.now()}.webm`);
        } else {
            startRecording();
        }
    }, [recordingState.isRecording, downloadRecording, startRecording, data?.layer_name]);

    // Get current preset
    const currentPreset = getPreset(selectedPreset);

    // Effective auto-rotate (marketing mode overrides)
    const effectiveAutoRotate = marketingMode || autoRotate;

    if (loading) {
        return (
            <div className="tensor-surface tensor-surface--loading" style={{ minHeight: height }}>
                <div className="tensor-surface__spinner" />
                <span>Loading activation data...</span>
            </div>
        );
    }

    if (!data) {
        return (
            <div className="tensor-surface tensor-surface--empty" style={{ minHeight: height }}>
                <span>No activation data available</span>
                <p>Start training to see live tensor activations</p>
            </div>
        );
    }

    return (
        <div className={`tensor-surface ${marketingMode ? 'tensor-surface--marketing' : ''}`}>
            <div className="tensor-surface__header">
                <h4>
                    <span className="layer-badge">{data.layer_name}</span>
                    Tensor Activation Surface
                </h4>
                <div className="tensor-surface__actions">
                    {showMarketingControls && (
                        <>
                            {/* Camera Preset Selector */}
                            <select
                                value={selectedPreset}
                                onChange={(e) => setSelectedPreset(e.target.value)}
                                className="preset-select"
                                title="Camera Preset"
                            >
                                {getPresetNames().map(name => (
                                    <option key={name} value={name}>
                                        {CAMERA_PRESETS[name].label}
                                    </option>
                                ))}
                            </select>

                            {/* Marketing Mode Toggle */}
                            <button
                                onClick={() => setMarketingMode(!marketingMode)}
                                className={`marketing-btn ${marketingMode ? 'marketing-btn--active' : ''}`}
                                title="Marketing Mode - Auto orbit with dramatic lighting"
                            >
                                🎬 {marketingMode ? 'ON' : 'OFF'}
                            </button>

                            {/* Video Record Button */}
                            <button
                                onClick={handleRecordToggle}
                                className={`record-btn ${recordingState.isRecording ? 'record-btn--recording' : ''}`}
                                title={recordingState.isRecording ? 'Stop recording and download' : 'Start recording'}
                            >
                                {recordingState.isRecording ? (
                                    <>🔴 {recordingState.duration}s</>
                                ) : (
                                    '⏺️ Record'
                                )}
                            </button>
                        </>
                    )}

                    {onExport && (
                        <button
                            onClick={handleExport}
                            disabled={isExporting}
                            className="export-btn"
                        >
                            📷 {isExporting ? 'Exporting...' : 'Export PNG'}
                        </button>
                    )}
                </div>
            </div>

            <div className="tensor-surface__canvas" style={{ height }}>
                <Canvas
                    ref={canvasRef}
                    gl={{ preserveDrawingBuffer: true, antialias: true }}
                    camera={{
                        position: currentPreset.position,
                        fov: currentPreset.fov
                    }}
                    dpr={[1, 2]}
                >
                    {/* Enhanced lighting for marketing mode */}
                    <ambientLight intensity={marketingMode ? 0.3 : 0.4} />
                    <directionalLight
                        position={[10, 10, 5]}
                        intensity={marketingMode ? 1.0 : 0.8}
                    />
                    <pointLight
                        position={[-10, -10, -5]}
                        intensity={marketingMode ? 0.5 : 0.3}
                        color="#4f46e5"
                    />
                    {marketingMode && (
                        <pointLight position={[0, 5, 10]} intensity={0.4} color="#8b5cf6" />
                    )}

                    <Surface data={data} autoRotate={effectiveAutoRotate} colorscale={colorscale} />

                    <OrbitControls
                        enableDamping
                        dampingFactor={0.05}
                        minDistance={5}
                        maxDistance={25}
                        autoRotate={marketingMode}
                        autoRotateSpeed={0.5}
                    />

                    {/* Stats overlay in 3D space */}
                    <Html position={[5.5, 0, 0]} center>
                        <div className="surface-stats-3d">
                            <div>Max: {data.stats.max.toFixed(3)}</div>
                            <div className="viridis-gradient" />
                            <div>Min: {data.stats.min.toFixed(3)}</div>
                        </div>
                    </Html>
                </Canvas>

                {/* Recording indicator overlay */}
                {recordingState.isRecording && (
                    <div className="recording-indicator">
                        <span className="recording-dot" />
                        REC {recordingState.duration}s
                    </div>
                )}
            </div>

            <div className="tensor-surface__footer">
                <span>μ = {data.stats.mean.toFixed(4)}</span>
                <span>σ = {data.stats.std.toFixed(4)}</span>
                <span>Shape: {data.z.length}×{data.z[0]?.length || 0}</span>
                {marketingMode && <span className="marketing-indicator">🎬 MARKETING</span>}
            </div>
        </div>
    );
}

export default TensorSurfaceViz;
