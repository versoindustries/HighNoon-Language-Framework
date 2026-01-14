// PostProcessingEffects.tsx - Post-processing pipeline for 3D visualizations
// Enhanced with bloom, chromatic aberration, film grain, and vignette effects

import React, { useMemo } from 'react';
import { EffectComposer, Bloom, Vignette, ChromaticAberration, Noise } from '@react-three/postprocessing';
import { BlendFunction } from 'postprocessing';
import * as THREE from 'three';

interface PostProcessingEffectsProps {
    enabled?: boolean;
    bloomEnabled?: boolean;
    bloomIntensity?: number;
    bloomThreshold?: number;
    vignetteEnabled?: boolean;
    vignetteIntensity?: number;
    chromaticAberrationEnabled?: boolean;
    chromaticAberrationOffset?: number;
    filmGrainEnabled?: boolean;
    filmGrainIntensity?: number;
    trainingPhase?: 'warmup' | 'exploration' | 'exploitation' | 'emergency' | 'idle';
}

/**
 * PostProcessingEffects - Cinematic-quality visual enhancement pipeline
 *
 * Features:
 * - Bloom for high-activation glow
 * - Chromatic aberration for high-tech optics feel
 * - Film grain for analog texture
 * - Vignette to focus attention
 * - Phase-aware intensity adjustments
 */
export function PostProcessingEffects({
    enabled = true,
    bloomEnabled = true,
    bloomIntensity = 0.5,
    bloomThreshold = 0.8,
    vignetteEnabled = true,
    vignetteIntensity = 0.4,
    chromaticAberrationEnabled = true,
    chromaticAberrationOffset = 0.002,
    filmGrainEnabled = false,
    filmGrainIntensity = 0.03,
    trainingPhase = 'idle',
}: PostProcessingEffectsProps) {
    // Phase-based intensity adjustments
    const phaseSettings = useMemo(() => {
        switch (trainingPhase) {
            case 'emergency':
                return {
                    bloomIntensity: bloomIntensity * 1.5,
                    vignetteIntensity: 0.6,
                    chromaticAberrationOffset: chromaticAberrationOffset * 2,
                    filmGrainIntensity: filmGrainIntensity * 1.5,
                };
            case 'exploration':
                return {
                    bloomIntensity: bloomIntensity * 1.2,
                    vignetteIntensity: vignetteIntensity * 0.8,
                    chromaticAberrationOffset: chromaticAberrationOffset * 1.2,
                    filmGrainIntensity,
                };
            case 'exploitation':
                return {
                    bloomIntensity: bloomIntensity * 0.8,
                    vignetteIntensity,
                    chromaticAberrationOffset: chromaticAberrationOffset * 0.5,
                    filmGrainIntensity: filmGrainIntensity * 0.5,
                };
            case 'warmup':
                return {
                    bloomIntensity: bloomIntensity * 0.6,
                    vignetteIntensity: vignetteIntensity * 0.5,
                    chromaticAberrationOffset,
                    filmGrainIntensity,
                };
            default:
                return {
                    bloomIntensity,
                    vignetteIntensity,
                    chromaticAberrationOffset,
                    filmGrainIntensity,
                };
        }
    }, [trainingPhase, bloomIntensity, vignetteIntensity, chromaticAberrationOffset, filmGrainIntensity]);

    // Chromatic aberration offset as Vector2
    const caOffset = useMemo(() => {
        const offset = phaseSettings.chromaticAberrationOffset;
        return new THREE.Vector2(offset, offset);
    }, [phaseSettings.chromaticAberrationOffset]);

    if (!enabled) return null;

    // Build effects array based on enabled flags
    const effects: React.ReactElement[] = [];

    if (bloomEnabled) {
        effects.push(
            <Bloom
                key="bloom"
                intensity={phaseSettings.bloomIntensity}
                luminanceThreshold={bloomThreshold}
                luminanceSmoothing={0.4}
                mipmapBlur
            />
        );
    }

    if (chromaticAberrationEnabled) {
        effects.push(
            <ChromaticAberration
                key="chromatic"
                offset={caOffset}
                radialModulation={true}
                modulationOffset={0.5}
            />
        );
    }

    if (filmGrainEnabled) {
        effects.push(
            <Noise
                key="noise"
                opacity={phaseSettings.filmGrainIntensity}
                blendFunction={BlendFunction.OVERLAY}
            />
        );
    }

    if (vignetteEnabled) {
        effects.push(
            <Vignette
                key="vignette"
                offset={0.3}
                darkness={phaseSettings.vignetteIntensity}
                blendFunction={BlendFunction.NORMAL}
            />
        );
    }

    if (effects.length === 0) return null;

    return (
        <EffectComposer>
            {effects}
        </EffectComposer>
    );
}

export default PostProcessingEffects;
