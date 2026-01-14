// ParticleFlowSystem.tsx - GPU-accelerated particles showing gradient flow
// Enhanced with InstancedMesh for better GPU performance

import { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface ParticleFlowSystemProps {
    surfaceWidth?: number;
    surfaceHeight?: number;
    particleCount?: number;
    gradientNorm?: number;
    flowSpeed?: number;
    enabled?: boolean;
    color?: string;
    useInstancing?: boolean;
    trainingPhase?: 'warmup' | 'exploration' | 'exploitation' | 'emergency' | 'idle';
}

// Shared geometry for instanced particles
const particleGeometry = new THREE.SphereGeometry(0.03, 8, 8);

/**
 * ParticleFlowSystem - GPU-accelerated particle field showing gradient flow
 *
 * Features:
 * - GPU instancing via InstancedMesh for high particle counts
 * - Per-instance color based on training phase
 * - Particles flow based on gradient information
 * - Color intensity reflects gradient magnitude
 * - Phase-aware particle behavior
 * - Automatic fallback to Points for simpler rendering
 */
export function ParticleFlowSystem({
    surfaceWidth = 10,
    surfaceHeight = 10,
    particleCount = 500,
    gradientNorm = 1.0,
    flowSpeed = 1.0,
    enabled = true,
    useInstancing = true,
    trainingPhase = 'idle',
}: ParticleFlowSystemProps) {
    const instancedRef = useRef<THREE.InstancedMesh>(null);
    const pointsRef = useRef<THREE.Points>(null);

    // Particle state refs
    const positionsRef = useRef<Float32Array | null>(null);
    const velocitiesRef = useRef<Float32Array | null>(null);
    const colorsRef = useRef<Float32Array | null>(null);
    const tempMatrix = useMemo(() => new THREE.Matrix4(), []);
    const tempColor = useMemo(() => new THREE.Color(), []);

    // Phase-based particle colors
    const phaseColor = useMemo(() => {
        switch (trainingPhase) {
            case 'exploration': return new THREE.Color('#a855f7'); // Purple
            case 'exploitation': return new THREE.Color('#22c55e'); // Green
            case 'emergency': return new THREE.Color('#ef4444'); // Red
            case 'warmup': return new THREE.Color('#3b82f6'); // Blue
            default: return new THREE.Color('#00f5ff'); // Quantum Cyan
        }
    }, [trainingPhase]);

    // Secondary color for gradient effect
    const secondaryColor = useMemo(() => {
        switch (trainingPhase) {
            case 'exploration': return new THREE.Color('#00f5ff');
            case 'exploitation': return new THREE.Color('#fbbf24');
            case 'emergency': return new THREE.Color('#fbbf24');
            case 'warmup': return new THREE.Color('#a855f7');
            default: return new THREE.Color('#a855f7');
        }
    }, [trainingPhase]);

    // Initialize particle data
    const particleData = useMemo(() => {
        const positions = new Float32Array(particleCount * 3);
        const velocities = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount; i++) {
            // Random position on surface plane
            positions[i * 3] = (Math.random() - 0.5) * surfaceWidth;
            positions[i * 3 + 1] = (Math.random() - 0.5) * surfaceHeight;
            positions[i * 3 + 2] = Math.random() * 3 + 0.5;

            // Initial color with gradient
            const t = Math.random();
            const color = phaseColor.clone().lerp(secondaryColor, t);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;

            // Random initial velocity
            velocities[i * 3] = (Math.random() - 0.5) * 0.5;
            velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.5;
            velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.2;
        }

        positionsRef.current = positions;
        velocitiesRef.current = velocities;
        colorsRef.current = colors;

        return { positions, velocities, colors };
    }, [particleCount, surfaceWidth, surfaceHeight, phaseColor, secondaryColor]);

    // Create instanced mesh color attribute
    useEffect(() => {
        if (instancedRef.current && useInstancing) {
            // Set initial instance matrices and colors
            for (let i = 0; i < particleCount; i++) {
                tempMatrix.setPosition(
                    particleData.positions[i * 3],
                    particleData.positions[i * 3 + 1],
                    particleData.positions[i * 3 + 2]
                );
                instancedRef.current.setMatrixAt(i, tempMatrix);

                tempColor.setRGB(
                    particleData.colors[i * 3],
                    particleData.colors[i * 3 + 1],
                    particleData.colors[i * 3 + 2]
                );
                instancedRef.current.setColorAt(i, tempColor);
            }
            instancedRef.current.instanceMatrix.needsUpdate = true;
            if (instancedRef.current.instanceColor) {
                instancedRef.current.instanceColor.needsUpdate = true;
            }
        }
    }, [particleData, particleCount, tempMatrix, tempColor, useInstancing]);

    // Update colors when phase changes
    useEffect(() => {
        if (!colorsRef.current) return;

        for (let i = 0; i < particleCount; i++) {
            const t = i / particleCount;
            const color = phaseColor.clone().lerp(secondaryColor, t);
            colorsRef.current[i * 3] = color.r;
            colorsRef.current[i * 3 + 1] = color.g;
            colorsRef.current[i * 3 + 2] = color.b;

            if (instancedRef.current && useInstancing) {
                instancedRef.current.setColorAt(i, color);
            }
        }

        if (instancedRef.current?.instanceColor) {
            instancedRef.current.instanceColor.needsUpdate = true;
        }

        if (pointsRef.current && !useInstancing) {
            const colorAttr = pointsRef.current.geometry.attributes.color;
            if (colorAttr) {
                colorAttr.needsUpdate = true;
            }
        }
    }, [phaseColor, secondaryColor, particleCount, useInstancing]);

    // Animation loop
    useFrame((state, delta) => {
        if (!enabled) return;

        const positions = positionsRef.current;
        const velocities = velocitiesRef.current;
        if (!positions || !velocities) return;

        const halfW = surfaceWidth / 2;
        const halfH = surfaceHeight / 2;
        const speed = flowSpeed * (0.5 + gradientNorm * 0.5);
        const time = state.clock.elapsedTime;

        for (let i = 0; i < particleCount; i++) {
            const idx = i * 3;

            // Apply velocity
            positions[idx] += velocities[idx] * delta * speed;
            positions[idx + 1] += velocities[idx + 1] * delta * speed;
            positions[idx + 2] += velocities[idx + 2] * delta * speed;

            // Add turbulence based on training phase
            if (trainingPhase === 'exploration') {
                velocities[idx] += (Math.random() - 0.5) * 0.1;
                velocities[idx + 1] += (Math.random() - 0.5) * 0.1;
            } else if (trainingPhase === 'emergency') {
                velocities[idx] += (Math.random() - 0.5) * 0.3;
                velocities[idx + 1] += (Math.random() - 0.5) * 0.3;
                velocities[idx + 2] += (Math.random() - 0.5) * 0.2;
            } else if (trainingPhase === 'exploitation') {
                // More organized flow during exploitation
                velocities[idx] *= 0.95;
                velocities[idx + 1] *= 0.95;
            }

            // Damping
            velocities[idx] *= 0.98;
            velocities[idx + 1] *= 0.98;
            velocities[idx + 2] *= 0.98;

            // Boundary respawn
            if (
                Math.abs(positions[idx]) > halfW ||
                Math.abs(positions[idx + 1]) > halfH ||
                positions[idx + 2] < 0 ||
                positions[idx + 2] > 4
            ) {
                // Respawn at random position
                positions[idx] = (Math.random() - 0.5) * surfaceWidth * 0.8;
                positions[idx + 1] = (Math.random() - 0.5) * surfaceHeight * 0.8;
                positions[idx + 2] = Math.random() * 2 + 0.5;

                // New random velocity
                velocities[idx] = (Math.random() - 0.5) * 0.5;
                velocities[idx + 1] = (Math.random() - 0.5) * 0.5;
                velocities[idx + 2] = (Math.random() - 0.5) * 0.2;
            }

            // Update instance matrix for instanced rendering
            if (useInstancing && instancedRef.current) {
                // Add subtle scale pulsing based on position
                const scale = 0.8 + Math.sin(time * 2 + i * 0.1) * 0.2;
                tempMatrix.makeScale(scale, scale, scale);
                tempMatrix.setPosition(
                    positions[idx],
                    positions[idx + 1],
                    positions[idx + 2]
                );
                instancedRef.current.setMatrixAt(i, tempMatrix);
            }
        }

        // Update GPU buffers
        if (useInstancing && instancedRef.current) {
            instancedRef.current.instanceMatrix.needsUpdate = true;
            // Gentle rotation of particle system
            instancedRef.current.rotation.z += delta * 0.02;
        }

        if (!useInstancing && pointsRef.current) {
            const posAttr = pointsRef.current.geometry.attributes.position;
            (posAttr.array as Float32Array).set(positions);
            posAttr.needsUpdate = true;
            pointsRef.current.rotation.z += delta * 0.02;
        }
    });

    if (!enabled) return null;

    // Use instanced mesh for better GPU performance
    if (useInstancing) {
        return (
            <group rotation={[-Math.PI / 3, 0, 0]}>
                <instancedMesh
                    ref={instancedRef}
                    args={[particleGeometry, undefined, particleCount]}
                    frustumCulled={false}
                >
                    <meshStandardMaterial
                        transparent
                        opacity={0.8}
                        emissive={phaseColor}
                        emissiveIntensity={0.5}
                        metalness={0.3}
                        roughness={0.5}
                    />
                </instancedMesh>
            </group>
        );
    }

    // Fallback to Points for simpler rendering
    return (
        <points ref={pointsRef} rotation={[-Math.PI / 3, 0, 0]}>
            <bufferGeometry>
                <bufferAttribute
                    attach="attributes-position"
                    args={[particleData.positions, 3]}
                />
                <bufferAttribute
                    attach="attributes-color"
                    args={[particleData.colors, 3]}
                />
            </bufferGeometry>
            <pointsMaterial
                size={0.08}
                vertexColors
                transparent
                opacity={0.7}
                sizeAttenuation
                depthWrite={false}
                blending={THREE.AdditiveBlending}
            />
        </points>
    );
}

export default ParticleFlowSystem;
