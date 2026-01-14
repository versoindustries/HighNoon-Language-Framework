// CameraPresets.ts - Dramatic Camera Angle Presets for Marketing
// Smooth animated transitions between cinematic viewpoints

import * as THREE from 'three';

/** Camera preset configuration */
export interface CameraPreset {
    name: string;
    label: string;
    position: [number, number, number];
    target: [number, number, number];
    fov: number;
    description: string;
}

/** Available camera presets for marketing and export */
export const CAMERA_PRESETS: Record<string, CameraPreset> = {
    hero: {
        name: 'hero',
        label: 'Hero Shot',
        position: [8, 6, 10],
        target: [0, 0, 0],
        fov: 45,
        description: 'Dramatic 45° angle with depth and drama',
    },
    topDown: {
        name: 'topDown',
        label: 'Top Down',
        position: [0, 15, 0.01],
        target: [0, 0, 0],
        fov: 50,
        description: 'Bird\'s eye view showing full surface',
    },
    sideProfile: {
        name: 'sideProfile',
        label: 'Side Profile',
        position: [15, 2, 0],
        target: [0, 0, 0],
        fov: 40,
        description: 'Layer depth visualization from side',
    },
    frontView: {
        name: 'frontView',
        label: 'Front View',
        position: [0, 3, 12],
        target: [0, 0, 0],
        fov: 50,
        description: 'Direct frontal perspective',
    },
    dramatic: {
        name: 'dramatic',
        label: 'Dramatic',
        position: [5, 2, 8],
        target: [0, 1, 0],
        fov: 35,
        description: 'Low angle dramatic perspective',
    },
    macro: {
        name: 'macro',
        label: 'Macro',
        position: [3, 2, 3],
        target: [0, 0, 0],
        fov: 60,
        description: 'Close-up detail shot',
    },
    cinematic: {
        name: 'cinematic',
        label: 'Cinematic',
        position: [10, 4, 6],
        target: [-1, 0, 0],
        fov: 30,
        description: 'Film-style telephoto perspective',
    },
};

/** Get list of all preset names */
export function getPresetNames(): string[] {
    return Object.keys(CAMERA_PRESETS);
}

/** Get preset by name with default fallback */
export function getPreset(name: string): CameraPreset {
    return CAMERA_PRESETS[name] ?? CAMERA_PRESETS.hero;
}

/**
 * Animate camera to target preset over duration
 *
 * @param camera - Three.js camera to animate
 * @param controls - OrbitControls instance
 * @param preset - Target preset configuration
 * @param duration - Animation duration in seconds
 * @param onComplete - Callback when animation completes
 */
export function animateToPreset(
    camera: THREE.PerspectiveCamera,
    controls: { target: THREE.Vector3; update: () => void },
    preset: CameraPreset,
    duration: number = 1.5,
    onComplete?: () => void
): () => void {
    const startPosition = camera.position.clone();
    const startTarget = controls.target.clone();
    const startFov = camera.fov;

    const endPosition = new THREE.Vector3(...preset.position);
    const endTarget = new THREE.Vector3(...preset.target);
    const endFov = preset.fov;

    const startTime = performance.now();
    let animationId: number;

    function animate() {
        const elapsed = (performance.now() - startTime) / 1000;
        const progress = Math.min(elapsed / duration, 1);

        // Smooth easing function (ease-out-expo)
        const eased = 1 - Math.pow(1 - progress, 4);

        // Interpolate position
        camera.position.lerpVectors(startPosition, endPosition, eased);

        // Interpolate target
        controls.target.lerpVectors(startTarget, endTarget, eased);

        // Interpolate FOV
        camera.fov = startFov + (endFov - startFov) * eased;
        camera.updateProjectionMatrix();

        // Update controls
        controls.update();

        if (progress < 1) {
            animationId = requestAnimationFrame(animate);
        } else if (onComplete) {
            onComplete();
        }
    }

    animationId = requestAnimationFrame(animate);

    // Return cancel function
    return () => cancelAnimationFrame(animationId);
}

/** Marketing mode orbit animation parameters */
export interface OrbitConfig {
    /** Orbit radius */
    radius: number;
    /** Orbit speed (radians per second) */
    speed: number;
    /** Vertical oscillation amplitude */
    verticalAmplitude: number;
    /** Center position to orbit around */
    center: [number, number, number];
}

/** Default orbit configuration for marketing mode */
export const DEFAULT_ORBIT_CONFIG: OrbitConfig = {
    radius: 10,
    speed: 0.15,
    verticalAmplitude: 2,
    center: [0, 0, 0],
};

/**
 * Create orbit animation update function for marketing mode
 *
 * @param camera - Three.js camera to animate
 * @param config - Orbit configuration
 * @returns Update function to call in animation frame
 */
export function createOrbitAnimation(
    camera: THREE.PerspectiveCamera,
    config: OrbitConfig = DEFAULT_ORBIT_CONFIG
): (time: number) => void {
    const { radius, speed, verticalAmplitude, center } = config;
    const centerVec = new THREE.Vector3(...center);

    return (time: number) => {
        const angle = time * speed;
        camera.position.x = centerVec.x + Math.cos(angle) * radius;
        camera.position.z = centerVec.z + Math.sin(angle) * radius;
        camera.position.y = centerVec.y + 4 + Math.sin(angle * 0.5) * verticalAmplitude;
        camera.lookAt(centerVec);
    };
}

export default CAMERA_PRESETS;
