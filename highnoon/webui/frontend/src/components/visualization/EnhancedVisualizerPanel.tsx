// EnhancedVisualizerPanel.tsx - Unified enhanced 3D visualization panel
// Combines all visualization modes into a single prominent component

import { useState, useCallback, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html, Environment } from '@react-three/drei';
import * as THREE from 'three';
import {
    Eye, Layers, Zap, Target, LineChart, Thermometer,
    Play, Pause, Camera, Video, Settings, Maximize2,
    GitCompare, Box
} from 'lucide-react';

// Import all visualization components
import { AnimatedSurface } from './AnimatedSurface';
import { ParticleFlowSystem } from './ParticleFlowSystem';
import { HolographicSurface } from './HolographicSurface';
import { PostProcessingEffects } from './PostProcessingEffects';
import { GradientArrows } from './GradientArrows';
import { ActivationAnnotations } from './ActivationAnnotations';
import { HistoryTrails } from './HistoryTrails';
import { LossLandscape3D } from './LossLandscape3D';
import { QuantumTunnelingViz } from './QuantumTunnelingViz';
import { TemperatureHeatmap } from './TemperatureHeatmap';
// Phase 3: Multi-layer visualization
import { StackedLayerViz, type LayerSurfaceData } from './StackedLayerViz';
import { AttentionFlowViz, type AttentionConnection } from './AttentionFlowViz';
import { useVideoRecorder } from './useVideoRecorder';
import { CAMERA_PRESETS, getPresetNames, getPreset } from './CameraPresets';
import type { SurfaceData } from './TensorSurfaceViz';

import './EnhancedVisualizerPanel.css';

type VisualizationMode = 'surface' | 'holographic' | 'stacked' | 'landscape' | 'quantum';

interface EnhancedVisualizerPanelProps {
    /** Activation surface data */
    surfaceData: SurfaceData | null;
    /** Multi-layer surface data for stacked visualization */
    multiLayerData?: LayerSurfaceData[];
    /** Attention connections for flow visualization */
    attentionConnections?: AttentionConnection[];
    /** Loading state */
    loading?: boolean;
    /** Is training/HPO running */
    isRunning?: boolean;
    /** Current training phase */
    trainingPhase?: 'warmup' | 'exploration' | 'exploitation' | 'emergency' | 'idle';
    /** QAHPO tunneling probability (0-1) */
    tunnelingProbability?: number;
    /** Annealing temperature */
    temperature?: number;
    /** Gradient norm for particle flow */
    gradientNorm?: number;
    /** Callback for screenshot export */
    onExport?: (dataUrl: string) => void;
    /** Height of the visualization panel */
    height?: number;
}

/**
 * EnhancedVisualizerPanel - Premium 3D visualization dashboard
 *
 * Features:
 * - Multiple visualization modes (Surface, Holographic, Landscape, Quantum)
 * - Real-time animation with phase-aware effects
 * - Configurable overlays (particles, arrows, annotations, history)
 * - Video recording and screenshot export
 * - Responsive and fullscreen capable
 */
export function EnhancedVisualizerPanel({
    surfaceData,
    multiLayerData = [],
    attentionConnections = [],
    loading = false,
    isRunning = false,
    trainingPhase = 'idle',
    tunnelingProbability = 0.1,
    temperature = 1.0,
    gradientNorm = 1.0,
    onExport,
    height = 450,
}: EnhancedVisualizerPanelProps) {
    // State
    const [mode, setMode] = useState<VisualizationMode>('surface');
    const [selectedPreset, setSelectedPreset] = useState('hero');
    const [showParticles, setShowParticles] = useState(true);
    const [showGradients, setShowGradients] = useState(false);
    const [showAnnotations, setShowAnnotations] = useState(true);
    const [showHistory, setShowHistory] = useState(false);
    const [showTemperature, setShowTemperature] = useState(true);
    const [showAttentionFlow, setShowAttentionFlow] = useState(false);
    const [postProcessing, setPostProcessing] = useState(true);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const canvasRef = { current: null as HTMLCanvasElement | null };

    // Video recording
    const { state: recordingState, startRecording, downloadRecording } = useVideoRecorder(canvasRef);

    // Camera preset
    const currentPreset = getPreset(selectedPreset);

    // Generate demo multi-layer data if not provided
    const effectiveMultiLayerData = useMemo(() => {
        if (multiLayerData.length > 0) return multiLayerData;
        if (!surfaceData) return [];

        // Create demo stacked layers from single surface
        return ['layer_0', 'layer_1', 'layer_2', 'layer_3'].map((name, idx) => ({
            layerName: name,
            z: surfaceData.z.map(row =>
                row.map(val => val * (1 - idx * 0.15) + Math.sin(idx) * 0.1)
            ),
            stats: {
                ...surfaceData.stats,
                mean: surfaceData.stats.mean * (1 - idx * 0.1),
            },
        }));
    }, [multiLayerData, surfaceData]);

    // Mode tabs
    const modes: { id: VisualizationMode; label: string; icon: React.ReactNode }[] = [
        { id: 'surface', label: 'Surface', icon: <Layers size={14} /> },
        { id: 'holographic', label: 'Holo', icon: <Eye size={14} /> },
        { id: 'stacked', label: 'Stacked', icon: <Box size={14} /> },
        { id: 'landscape', label: 'Loss', icon: <LineChart size={14} /> },
        { id: 'quantum', label: 'Quantum', icon: <Zap size={14} /> },
    ];

    // Toggle handlers
    const handleRecordToggle = useCallback(() => {
        if (recordingState.isRecording) {
            downloadRecording(`visualization_${Date.now()}.webm`);
        } else {
            startRecording();
        }
    }, [recordingState.isRecording, downloadRecording, startRecording]);

    const handleExport = useCallback(() => {
        if (!canvasRef.current || !onExport) return;
        requestAnimationFrame(() => {
            const dataUrl = canvasRef.current?.toDataURL('image/png');
            if (dataUrl) onExport(dataUrl);
        });
    }, [onExport]);

    // Loading state
    if (loading) {
        return (
            <div className="enhanced-viz-panel enhanced-viz-panel--loading" style={{ height }}>
                <div className="enhanced-viz-panel__spinner" />
                <span>Loading visualization data...</span>
            </div>
        );
    }

    return (
        <div className={`enhanced-viz-panel ${isFullscreen ? 'enhanced-viz-panel--fullscreen' : ''}`}>
            {/* Header */}
            <div className="enhanced-viz-panel__header">
                <div className="enhanced-viz-panel__title">
                    <Layers size={18} />
                    <span>3D Visualization</span>
                    {isRunning && (
                        <span className="enhanced-viz-panel__live-badge">
                            <span className="live-dot" /> LIVE
                        </span>
                    )}
                </div>

                {/* Mode tabs */}
                <div className="enhanced-viz-panel__modes">
                    {modes.map(m => (
                        <button
                            key={m.id}
                            className={`mode-tab ${mode === m.id ? 'mode-tab--active' : ''}`}
                            onClick={() => setMode(m.id)}
                        >
                            {m.icon}
                            <span>{m.label}</span>
                        </button>
                    ))}
                </div>

                {/* Actions */}
                <div className="enhanced-viz-panel__actions">
                    <select
                        value={selectedPreset}
                        onChange={(e) => setSelectedPreset(e.target.value)}
                        className="preset-select"
                    >
                        {getPresetNames().map(name => (
                            <option key={name} value={name}>
                                {CAMERA_PRESETS[name].label}
                            </option>
                        ))}
                    </select>
                    <button
                        className={`action-btn ${recordingState.isRecording ? 'action-btn--recording' : ''}`}
                        onClick={handleRecordToggle}
                        title={recordingState.isRecording ? 'Stop recording' : 'Record video'}
                    >
                        {recordingState.isRecording ? (
                            <>🔴 {recordingState.duration}s</>
                        ) : (
                            <Video size={16} />
                        )}
                    </button>
                    {onExport && (
                        <button className="action-btn" onClick={handleExport} title="Export PNG">
                            <Camera size={16} />
                        </button>
                    )}
                    <button
                        className="action-btn"
                        onClick={() => setIsFullscreen(!isFullscreen)}
                        title="Toggle fullscreen"
                    >
                        <Maximize2 size={16} />
                    </button>
                </div>
            </div>

            {/* Overlay toggles */}
            <div className="enhanced-viz-panel__overlays">
                <label className="overlay-toggle">
                    <input
                        type="checkbox"
                        checked={showParticles}
                        onChange={(e) => setShowParticles(e.target.checked)}
                    />
                    <span>Particles</span>
                </label>
                <label className="overlay-toggle">
                    <input
                        type="checkbox"
                        checked={showGradients}
                        onChange={(e) => setShowGradients(e.target.checked)}
                    />
                    <span>Gradients</span>
                </label>
                <label className="overlay-toggle">
                    <input
                        type="checkbox"
                        checked={showAnnotations}
                        onChange={(e) => setShowAnnotations(e.target.checked)}
                    />
                    <span>Annotations</span>
                </label>
                <label className="overlay-toggle">
                    <input
                        type="checkbox"
                        checked={showHistory}
                        onChange={(e) => setShowHistory(e.target.checked)}
                    />
                    <span>History</span>
                </label>
                <label className="overlay-toggle">
                    <input
                        type="checkbox"
                        checked={showTemperature}
                        onChange={(e) => setShowTemperature(e.target.checked)}
                    />
                    <span>Temperature</span>
                </label>
                <label className="overlay-toggle">
                    <input
                        type="checkbox"
                        checked={showAttentionFlow}
                        onChange={(e) => setShowAttentionFlow(e.target.checked)}
                    />
                    <span>Attention</span>
                </label>
                <label className="overlay-toggle">
                    <input
                        type="checkbox"
                        checked={postProcessing}
                        onChange={(e) => setPostProcessing(e.target.checked)}
                    />
                    <span>Effects</span>
                </label>
            </div>

            {/* 3D Canvas */}
            <div className="enhanced-viz-panel__canvas" style={{ height: isFullscreen ? '80vh' : height }}>
                <Canvas
                    gl={{ preserveDrawingBuffer: true, antialias: true }}
                    camera={{ position: currentPreset.position, fov: currentPreset.fov }}
                    dpr={[1, 2]}
                    onCreated={({ gl }) => { canvasRef.current = gl.domElement; }}
                >
                    {/* Lighting */}
                    <ambientLight intensity={0.4} />
                    <directionalLight position={[10, 10, 5]} intensity={0.8} />
                    <pointLight position={[-10, -10, -5]} intensity={0.3} color="#4f46e5" />
                    {trainingPhase === 'exploration' && (
                        <pointLight position={[0, 5, 10]} intensity={0.4} color="#8b5cf6" />
                    )}
                    {trainingPhase === 'emergency' && (
                        <pointLight position={[0, 5, 0]} intensity={0.6} color="#ef4444" />
                    )}

                    {/* Main visualization based on mode */}
                    {mode === 'surface' && surfaceData && (
                        <AnimatedSurface
                            data={surfaceData}
                            autoRotate={isRunning}
                            breathingEnabled={isRunning}
                            trainingPhase={trainingPhase}
                        />
                    )}

                    {mode === 'holographic' && surfaceData && (
                        <HolographicSurface
                            data={surfaceData}
                            autoRotate={isRunning}
                            primaryColor={trainingPhase === 'emergency' ? '#ff4444' : '#00ffff'}
                            secondaryColor={trainingPhase === 'emergency' ? '#ff8800' : '#ff00ff'}
                        />
                    )}

                    {mode === 'stacked' && effectiveMultiLayerData.length > 0 && (
                        <StackedLayerViz
                            layers={effectiveMultiLayerData}
                            layerSpacing={2.0}
                            showLabels={true}
                            enabled={true}
                        />
                    )}

                    {mode === 'landscape' && (
                        <LossLandscape3D
                            enabled={true}
                            autoGenerate={true}
                            showTrajectory={true}
                        />
                    )}

                    {mode === 'quantum' && (
                        <QuantumTunnelingViz
                            tunnelingProbability={tunnelingProbability}
                            temperature={temperature}
                            enabled={true}
                            showBarrier={true}
                        />
                    )}

                    {/* Overlay components */}
                    {showParticles && surfaceData && (
                        <ParticleFlowSystem
                            enabled={isRunning}
                            gradientNorm={gradientNorm}
                            trainingPhase={trainingPhase}
                            flowSpeed={1.5}
                        />
                    )}

                    {showGradients && surfaceData && (
                        <GradientArrows
                            surfaceData={surfaceData}
                            enabled={true}
                            gradientNorm={gradientNorm}
                        />
                    )}

                    {showAnnotations && surfaceData && (
                        <ActivationAnnotations
                            surfaceData={surfaceData}
                            enabled={true}
                            showMax={true}
                            showMin={true}
                        />
                    )}

                    {showHistory && surfaceData && (
                        <HistoryTrails
                            data={surfaceData}
                            enabled={isRunning}
                            maxHistory={5}
                        />
                    )}

                    {showTemperature && (mode === 'quantum' || mode === 'landscape') && (
                        <TemperatureHeatmap
                            temperature={temperature}
                            enabled={true}
                            showGauge={true}
                        />
                    )}

                    {/* Attention flow visualization (Phase 3) */}
                    {showAttentionFlow && attentionConnections.length > 0 && (
                        <AttentionFlowViz
                            connections={attentionConnections}
                            enabled={true}
                            animationSpeed={1.0}
                            showParticles={true}
                            colorByHead={true}
                        />
                    )}

                    {/* Demo attention flow when enabled but no data */}
                    {showAttentionFlow && attentionConnections.length === 0 && mode === 'stacked' && (
                        <AttentionFlowViz
                            connections={[
                                { fromPosition: [-2, -2, 0], toPosition: [2, 0, 2], weight: 0.9, headIndex: 0 },
                                { fromPosition: [-1, -1, 1], toPosition: [1, 1, 3], weight: 0.7, headIndex: 1 },
                                { fromPosition: [0, -2, 0], toPosition: [0, 2, 4], weight: 0.8, headIndex: 2 },
                                { fromPosition: [1, -1, 1], toPosition: [-1, 1, 3], weight: 0.6, headIndex: 3 },
                                { fromPosition: [2, 0, 0], toPosition: [-2, 0, 2], weight: 0.5, headIndex: 0 },
                            ]}
                            enabled={true}
                            animationSpeed={1.0}
                            showParticles={true}
                            colorByHead={true}
                        />
                    )}

                    {/* Post-processing effects */}
                    {postProcessing && (
                        <PostProcessingEffects
                            enabled={true}
                            bloomEnabled={true}
                            bloomIntensity={0.4}
                            trainingPhase={trainingPhase}
                        />
                    )}

                    {/* Controls */}
                    <OrbitControls
                        enableDamping
                        dampingFactor={0.05}
                        minDistance={5}
                        maxDistance={25}
                        autoRotate={isRunning && mode !== 'quantum'}
                        autoRotateSpeed={0.3}
                    />
                </Canvas>

                {/* Recording indicator */}
                {recordingState.isRecording && (
                    <div className="recording-indicator">
                        <span className="recording-dot" />
                        REC {recordingState.duration}s
                    </div>
                )}
            </div>

            {/* Footer stats */}
            {surfaceData && (
                <div className="enhanced-viz-panel__footer">
                    <span className="stat">
                        <strong>Layer:</strong> {surfaceData.layer_name}
                    </span>
                    <span className="stat">
                        <strong>Shape:</strong> {surfaceData.z.length}×{surfaceData.z[0]?.length || 0}
                    </span>
                    <span className="stat">
                        <strong>μ:</strong> {surfaceData.stats.mean.toFixed(4)}
                    </span>
                    <span className="stat">
                        <strong>σ:</strong> {surfaceData.stats.std.toFixed(4)}
                    </span>
                    <span className="stat">
                        <strong>Phase:</strong> {trainingPhase.toUpperCase()}
                    </span>
                </div>
            )}
        </div>
    );
}

export default EnhancedVisualizerPanel;
