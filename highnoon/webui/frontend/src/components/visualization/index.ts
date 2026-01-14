// Visualization components barrel export

// Core tensor visualization
export { TensorSurfaceViz, type SurfaceData } from './TensorSurfaceViz';

// Enhanced 3D visualization components (Phase 1-2)
export { AnimatedSurface } from './AnimatedSurface';
export { ParticleFlowSystem } from './ParticleFlowSystem';
export { HolographicSurface } from './HolographicSurface';
export { PostProcessingEffects } from './PostProcessingEffects';

// Multi-layer visualization components (Phase 3)
export { StackedLayerViz, type LayerSurfaceData } from './StackedLayerViz';
export { AttentionFlowViz, type AttentionConnection } from './AttentionFlowViz';
export { LayerComparisonSlider } from './LayerComparisonSlider';

// Contextual information overlays (Phase 4)
export { GradientArrows } from './GradientArrows';
export { ActivationAnnotations } from './ActivationAnnotations';
export { HistoryTrails } from './HistoryTrails';

// Loss landscape & quantum visualization (Phase 5)
export { LossLandscape3D } from './LossLandscape3D';
export { QuantumTunnelingViz } from './QuantumTunnelingViz';
export { TemperatureHeatmap } from './TemperatureHeatmap';

// Phase 4: Holographic & Marketing visualizations
export { HolographicBundleViz } from './HolographicBundleViz';
export { FloquetPhaseViz } from './FloquetPhaseViz';
export { ActivationHeatmap } from './ActivationHeatmap';

// Phase 5: HPO & Architecture visualization (NEW)
export { HPOTrialScatterPlot3D } from './HPOTrialScatterPlot3D';
export { NeuralArchitectureGraph3D } from './NeuralArchitectureGraph3D';

// Utilities
export { useVideoRecorder } from './useVideoRecorder';
export { useActivationStream } from './useActivationStream';
export {
    CAMERA_PRESETS,
    getPresetNames,
    getPreset,
    animateToPreset,
    createOrbitAnimation,
    DEFAULT_ORBIT_CONFIG,
    type CameraPreset,
    type OrbitConfig,
} from './CameraPresets';

// Colorscales
export {
    viridisColor,
    plasmaColor,
    infernoColor,
    colorscales,
    type ColorscaleName,
    type ColorscaleFunction,
} from './colorscales';
