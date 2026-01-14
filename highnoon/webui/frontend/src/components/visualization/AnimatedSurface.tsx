// AnimatedSurface.tsx - Smooth morphing 3D surface with vertex animation
// Uses lerp interpolation for fluid transitions between WebSocket updates

import { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { SurfaceData } from './TensorSurfaceViz';
import { colorscales, type ColorscaleName } from './colorscales';

interface AnimatedSurfaceProps {
    data: SurfaceData;
    colorscale?: ColorscaleName;
    autoRotate?: boolean;
    breathingEnabled?: boolean;
    breathingIntensity?: number;
    morphSpeed?: number;
    trainingPhase?: 'warmup' | 'exploration' | 'exploitation' | 'emergency' | 'idle';
}

/**
 * AnimatedSurface - Enhanced 3D surface mesh with smooth interpolation
 *
 * Features:
 * - Smooth lerp-based morphing between data updates
 * - Breathing/pulsing vertex displacement effect
 * - Phase-aware colorscale transitions
 * - Frame-rate independent animation
 */
export function AnimatedSurface({
    data,
    colorscale = 'viridis',
    autoRotate = false,
    breathingEnabled = true,
    breathingIntensity = 0.015,
    morphSpeed = 5,
    trainingPhase = 'idle',
}: AnimatedSurfaceProps) {
    const meshRef = useRef<THREE.Mesh>(null);
    const geometryRef = useRef<THREE.PlaneGeometry | null>(null);

    // Store target positions for lerp animation
    const targetPositions = useRef<Float32Array | null>(null);
    const currentPositions = useRef<Float32Array | null>(null);
    const basePositions = useRef<Float32Array | null>(null);

    // Phase-based colorscale mapping
    const phaseColorscale = useMemo(() => {
        switch (trainingPhase) {
            case 'exploration': return 'plasma';
            case 'exploitation': return 'viridis';
            case 'emergency': return 'inferno';
            default: return colorscale;
        }
    }, [trainingPhase, colorscale]);

    const colorFn = colorscales[phaseColorscale] ?? colorscales.viridis;

    // Initialize geometry and compute target positions when data changes
    const geometry = useMemo(() => {
        const size = data.x.length;
        const geom = new THREE.PlaneGeometry(10, 10, size - 1, size - 1);
        const positions = geom.attributes.position;
        const colors = new Float32Array(positions.count * 3);

        const { min, max } = data.stats;
        const range = max - min || 1;

        // Compute vertex positions and colors
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const idx = i * size + j;
                const z = data.z[i]?.[j] ?? 0;

                // Set height
                const normalizedZ = ((z - min) / range) * 3;
                positions.setZ(idx, normalizedZ);

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

        // Store reference for animation
        geometryRef.current = geom;

        // Initialize position arrays for lerp animation
        const posArray = new Float32Array(positions.array);
        targetPositions.current = posArray.slice();

        if (!currentPositions.current || currentPositions.current.length !== posArray.length) {
            currentPositions.current = posArray.slice();
            basePositions.current = posArray.slice();
        } else {
            // Update targets, let currentPositions lerp toward them
            targetPositions.current = posArray.slice();
        }

        return geom;
    }, [data, colorFn]);

    // Animation loop
    useFrame((state, delta) => {
        if (!meshRef.current || !geometryRef.current) return;

        const positions = geometryRef.current.attributes.position;
        const posArray = positions.array as Float32Array;

        // Lerp current positions toward targets
        if (targetPositions.current && currentPositions.current) {
            let needsUpdate = false;

            for (let i = 0; i < posArray.length; i++) {
                const target = targetPositions.current[i];
                const current = currentPositions.current[i];
                const diff = target - current;

                if (Math.abs(diff) > 0.0001) {
                    currentPositions.current[i] = current + diff * Math.min(1, delta * morphSpeed);
                    needsUpdate = true;
                }
            }

            if (needsUpdate) {
                // Apply lerped positions
                for (let i = 0; i < posArray.length; i++) {
                    posArray[i] = currentPositions.current[i];
                }
            }
        }

        // Breathing effect - sinusoidal z-displacement
        if (breathingEnabled && basePositions.current) {
            const breathe = Math.sin(state.clock.elapsedTime * 2) * breathingIntensity;
            const pulse = Math.sin(state.clock.elapsedTime * 0.5) * breathingIntensity * 0.5;

            const size = data.x.length;
            for (let i = 0; i < size; i++) {
                for (let j = 0; j < size; j++) {
                    const idx = (i * size + j) * 3 + 2; // Z component
                    // Wave effect: varies by position
                    const waveOffset = Math.sin(i * 0.3 + state.clock.elapsedTime) *
                        Math.cos(j * 0.3 + state.clock.elapsedTime * 0.7);
                    posArray[idx] += (breathe + pulse * waveOffset) * 0.5;
                }
            }
        }

        positions.needsUpdate = true;
        geometryRef.current.computeVertexNormals();

        // Auto-rotation
        if (autoRotate) {
            meshRef.current.rotation.z += delta * 0.1;
        }
    });

    // Phase-based material properties
    const materialProps = useMemo(() => {
        switch (trainingPhase) {
            case 'emergency':
                return { emissive: '#ff3300', emissiveIntensity: 0.3, metalness: 0.3 };
            case 'exploration':
                return { emissive: '#8b5cf6', emissiveIntensity: 0.1, metalness: 0.2 };
            case 'exploitation':
                return { emissive: '#00ff88', emissiveIntensity: 0.05, metalness: 0.15 };
            default:
                return { emissive: '#000000', emissiveIntensity: 0, metalness: 0.1 };
        }
    }, [trainingPhase]);

    return (
        <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 3, 0, 0]}>
            <meshStandardMaterial
                vertexColors
                side={THREE.DoubleSide}
                roughness={0.7}
                {...materialProps}
            />
        </mesh>
    );
}

export default AnimatedSurface;
