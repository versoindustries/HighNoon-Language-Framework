// QuantumTunnelingViz.tsx - Visualize QAHPO quantum tunneling events
// Enhanced with probability cloud, screen shake, and dramatic flash effects

import { useRef, useMemo, useEffect, useState, useCallback } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';

interface TunnelingEvent {
    timestamp: number;
    fromPosition: [number, number, number];
    toPosition: [number, number, number];
    probability: number;
}

interface QuantumTunnelingVizProps {
    tunnelingProbability?: number;
    temperature?: number;
    currentPosition?: { x: number; y: number } | null;
    enabled?: boolean;
    showBarrier?: boolean;
    showProbabilityCloud?: boolean;
    enableScreenShake?: boolean;
    onTunnelingEvent?: (event: TunnelingEvent) => void;
}

// Probability cloud particle system config
const CLOUD_PARTICLE_COUNT = 100;

/**
 * QuantumTunnelingViz - QAHPO tunneling visualization
 *
 * Features:
 * - Animated probability density cloud (3D noise field)
 * - Barrier with pulsing glow based on probability
 * - Screen shake on tunneling event (subtle camera jitter)
 * - Dramatic flash effect during phase-through
 * - Particle jump animation on tunneling events
 * - Wave function ripple visualization
 * - Tunneling counter and probability HUD
 */
export function QuantumTunnelingViz({
    tunnelingProbability = 0.1,
    temperature = 1.0,
    currentPosition,
    enabled = true,
    showBarrier = true,
    showProbabilityCloud = true,
    enableScreenShake = true,
    onTunnelingEvent,
}: QuantumTunnelingVizProps) {
    const { camera } = useThree();
    const particleRef = useRef<THREE.Mesh>(null);
    const barrierRef = useRef<THREE.Mesh>(null);
    const cloudRef = useRef<THREE.Points>(null);
    const flashRef = useRef<THREE.Mesh>(null);
    const afterimageRefs = useRef<THREE.Mesh[]>([]);

    const [isTunneling, setIsTunneling] = useState(false);
    const [tunnelingCount, setTunnelingCount] = useState(0);
    const [showFlash, setShowFlash] = useState(false);

    const lastTunnelTime = useRef(0);
    const tunnelingTarget = useRef<[number, number, number] | null>(null);
    const shakeIntensity = useRef(0);
    const originalCameraPos = useRef(new THREE.Vector3());

    // Create wave function barrier geometry (animated Gaussian curve)
    const barrierGeometry = useMemo(() => {
        const curve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(-3, 0, 0),
            new THREE.Vector3(-1.5, 2, 0),
            new THREE.Vector3(-0.5, 3.2, 0),
            new THREE.Vector3(0, 3.5, 0),
            new THREE.Vector3(0.5, 3.2, 0),
            new THREE.Vector3(1.5, 2, 0),
            new THREE.Vector3(3, 0, 0),
        ]);
        return new THREE.TubeGeometry(curve, 48, 0.15, 12, false);
    }, []);

    // Create probability cloud particles
    const cloudParticles = useMemo(() => {
        const positions = new Float32Array(CLOUD_PARTICLE_COUNT * 3);
        const velocities = new Float32Array(CLOUD_PARTICLE_COUNT * 3);
        const phases = new Float32Array(CLOUD_PARTICLE_COUNT);

        for (let i = 0; i < CLOUD_PARTICLE_COUNT; i++) {
            // Gaussian distribution around origin
            const r = Math.random() * 2;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;

            positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) + 1.5;
            positions[i * 3 + 2] = r * Math.cos(phi);

            velocities[i * 3] = (Math.random() - 0.5) * 0.02;
            velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.02;
            velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.02;

            phases[i] = Math.random() * Math.PI * 2;
        }

        return { positions, velocities, phases };
    }, []);

    // Create afterimage trail for tunneling
    const afterimagePositions = useMemo(() => {
        return Array.from({ length: 5 }, () => new THREE.Vector3());
    }, []);

    // Store original camera position for shake reset
    useEffect(() => {
        originalCameraPos.current.copy(camera.position);
    }, [camera]);

    // Trigger tunneling event
    const triggerTunneling = useCallback((targetPos: [number, number, number]) => {
        const now = Date.now();
        setIsTunneling(true);
        lastTunnelTime.current = now;
        setTunnelingCount(c => c + 1);
        tunnelingTarget.current = targetPos;

        // Trigger screen shake
        if (enableScreenShake) {
            shakeIntensity.current = 0.15;
        }

        // Trigger flash effect
        setShowFlash(true);
        setTimeout(() => setShowFlash(false), 150);

        // Store afterimage positions
        if (particleRef.current) {
            afterimagePositions.forEach((pos, i) => {
                const progress = i / afterimagePositions.length;
                pos.lerpVectors(
                    particleRef.current!.position,
                    new THREE.Vector3(...targetPos),
                    progress
                );
            });
        }

        if (onTunnelingEvent) {
            onTunnelingEvent({
                timestamp: now,
                fromPosition: currentPosition
                    ? [currentPosition.x, 0, currentPosition.y]
                    : [0, 0, 0],
                toPosition: targetPos,
                probability: tunnelingProbability,
            });
        }

        // End tunneling animation after delay
        setTimeout(() => {
            setIsTunneling(false);
            tunnelingTarget.current = null;
        }, 800);
    }, [currentPosition, enableScreenShake, onTunnelingEvent, tunnelingProbability, afterimagePositions]);

    // Check for tunneling events
    useEffect(() => {
        if (!enabled) return;

        const checkTunneling = () => {
            const now = Date.now();
            if (now - lastTunnelTime.current > 2000 && tunnelingProbability > 0.3) {
                if (Math.random() < tunnelingProbability * 0.1) {
                    const targetPos: [number, number, number] = [
                        (Math.random() - 0.5) * 4,
                        Math.random() * 2 + 1,
                        (Math.random() - 0.5) * 4,
                    ];
                    triggerTunneling(targetPos);
                }
            }
        };

        const interval = setInterval(checkTunneling, 500);
        return () => clearInterval(interval);
    }, [tunnelingProbability, enabled, triggerTunneling]);

    // Animation loop
    useFrame((state, delta) => {
        const time = state.clock.elapsedTime;

        // Screen shake decay
        if (shakeIntensity.current > 0) {
            const shake = shakeIntensity.current;
            camera.position.x = originalCameraPos.current.x + (Math.random() - 0.5) * shake;
            camera.position.y = originalCameraPos.current.y + (Math.random() - 0.5) * shake;
            shakeIntensity.current *= 0.9; // Decay

            if (shakeIntensity.current < 0.001) {
                camera.position.copy(originalCameraPos.current);
                shakeIntensity.current = 0;
            }
        }

        // Barrier pulsing based on probability
        if (barrierRef.current) {
            const basePulse = 1 + Math.sin(time * 4) * 0.1 * tunnelingProbability;
            const intensityPulse = isTunneling ? 1.3 : 1;
            barrierRef.current.scale.setScalar(basePulse * intensityPulse);
            barrierRef.current.rotation.y += delta * 0.3;

            // Update barrier material glow
            const mat = barrierRef.current.material as THREE.MeshStandardMaterial;
            mat.emissiveIntensity = 0.3 + tunnelingProbability * 0.7 + (isTunneling ? 1.5 : 0);
        }

        // Probability cloud animation
        if (cloudRef.current && showProbabilityCloud) {
            const positions = cloudRef.current.geometry.attributes.position;

            for (let i = 0; i < CLOUD_PARTICLE_COUNT; i++) {
                const phase = cloudParticles.phases[i];

                // Animated noise-like motion
                let x = positions.getX(i);
                let y = positions.getY(i);
                let z = positions.getZ(i);

                // Orbital motion with probability-based speed
                const speed = 0.5 + tunnelingProbability * 2;
                x += Math.sin(time * speed + phase) * 0.01;
                y += Math.cos(time * speed * 0.7 + phase) * 0.01;
                z += Math.sin(time * speed * 1.3 + phase) * 0.01;

                // Boundary check - keep particles in cloud region
                const dist = Math.sqrt(x * x + (y - 1.5) * (y - 1.5) + z * z);
                if (dist > 2.5) {
                    x *= 0.9;
                    y = (y - 1.5) * 0.9 + 1.5;
                    z *= 0.9;
                }

                positions.setXYZ(i, x, y, z);
            }

            positions.needsUpdate = true;

            // Cloud opacity based on probability
            const cloudMat = cloudRef.current.material as THREE.PointsMaterial;
            cloudMat.opacity = 0.2 + tunnelingProbability * 0.5;
        }

        // Particle tunneling animation
        if (particleRef.current) {
            if (isTunneling && tunnelingTarget.current) {
                // Quick jump to target with easing
                particleRef.current.position.lerp(
                    new THREE.Vector3(...tunnelingTarget.current),
                    delta * 8
                );
                // Intense glow during tunneling
                const mat = particleRef.current.material as THREE.MeshStandardMaterial;
                mat.emissiveIntensity = 3;
            } else {
                // Gentle oscillation at current position
                const baseY = 0.5;
                particleRef.current.position.y = baseY + Math.sin(time * 2) * 0.1;
                const mat = particleRef.current.material as THREE.MeshStandardMaterial;
                mat.emissiveIntensity = 0.5 + tunnelingProbability * 0.8;
            }
        }

        // Flash effect fade
        if (flashRef.current) {
            const mat = flashRef.current.material as THREE.MeshBasicMaterial;
            if (showFlash) {
                mat.opacity = 0.8;
            } else {
                mat.opacity *= 0.85;
            }
        }
    });

    if (!enabled) return null;

    return (
        <group>
            {/* Probability density cloud */}
            {showProbabilityCloud && (
                <points ref={cloudRef}>
                    <bufferGeometry>
                        <bufferAttribute
                            attach="attributes-position"
                            args={[cloudParticles.positions, 3]}
                        />
                    </bufferGeometry>
                    <pointsMaterial
                        color="#a855f7"
                        size={0.08}
                        transparent
                        opacity={0.3}
                        sizeAttenuation
                        blending={THREE.AdditiveBlending}
                    />
                </points>
            )}

            {/* Quantum barrier visualization with pulsing glow */}
            {showBarrier && (
                <mesh ref={barrierRef} geometry={barrierGeometry}>
                    <meshStandardMaterial
                        color="#8b5cf6"
                        transparent
                        opacity={0.5}
                        emissive="#a855f7"
                        emissiveIntensity={0.3 + tunnelingProbability * 0.5}
                        side={THREE.DoubleSide}
                    />
                </mesh>
            )}

            {/* Particle representing current state */}
            <mesh
                ref={particleRef}
                position={currentPosition ? [currentPosition.x * 2, 0.5, currentPosition.y * 2] : [0, 0.5, 0]}
            >
                <sphereGeometry args={[0.18, 24, 24]} />
                <meshStandardMaterial
                    color="#00f5ff"
                    emissive="#00f5ff"
                    emissiveIntensity={0.5}
                    transparent
                    opacity={0.95}
                    metalness={0.3}
                    roughness={0.2}
                />
            </mesh>

            {/* Afterimage trail during tunneling */}
            {isTunneling && afterimagePositions.map((pos, i) => (
                <mesh key={i} position={[pos.x, pos.y, pos.z]}>
                    <sphereGeometry args={[0.12 - i * 0.02, 12, 12]} />
                    <meshBasicMaterial
                        color="#00f5ff"
                        transparent
                        opacity={0.4 - i * 0.08}
                    />
                </mesh>
            ))}

            {/* Wave ripple during tunneling */}
            {isTunneling && (
                <>
                    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.1, 0]}>
                        <ringGeometry args={[0.5, 2.5, 48]} />
                        <meshBasicMaterial
                            color="#00f5ff"
                            transparent
                            opacity={0.4}
                            side={THREE.DoubleSide}
                        />
                    </mesh>
                    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.1, 0]}>
                        <ringGeometry args={[1.5, 3.5, 48]} />
                        <meshBasicMaterial
                            color="#a855f7"
                            transparent
                            opacity={0.25}
                            side={THREE.DoubleSide}
                        />
                    </mesh>
                </>
            )}

            {/* Flash effect on tunneling */}
            <mesh ref={flashRef} position={[0, 1, 0]}>
                <sphereGeometry args={[8, 16, 16]} />
                <meshBasicMaterial
                    color="#ffffff"
                    transparent
                    opacity={0}
                    side={THREE.BackSide}
                    blending={THREE.AdditiveBlending}
                />
            </mesh>

            {/* Tunneling counter HUD */}
            <Html position={[4.5, 3.5, 0]} center>
                <div className="tunneling-hud" style={{
                    background: 'rgba(10, 10, 15, 0.85)',
                    padding: '12px 16px',
                    borderRadius: '8px',
                    border: '1px solid rgba(168, 85, 247, 0.3)',
                    backdropFilter: 'blur(8px)',
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: '12px',
                    minWidth: '120px',
                }}>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginBottom: '8px',
                        color: '#a0a0b8',
                    }}>
                        <span>TUNNELING</span>
                        <span style={{
                            color: tunnelingProbability > 0.5 ? '#00f5ff' : '#a855f7',
                            fontWeight: 600,
                        }}>
                            {(tunnelingProbability * 100).toFixed(0)}%
                        </span>
                    </div>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        color: '#a0a0b8',
                    }}>
                        <span>EVENTS</span>
                        <span style={{ color: '#fbbf24', fontWeight: 600 }}>
                            {tunnelingCount}
                        </span>
                    </div>
                    {isTunneling && (
                        <div style={{
                            marginTop: '8px',
                            textAlign: 'center',
                            color: '#00f5ff',
                            fontWeight: 700,
                            animation: 'pulse 0.5s infinite',
                        }}>
                            ⚡ TUNNELING ⚡
                        </div>
                    )}
                </div>
            </Html>
        </group>
    );
}

export default QuantumTunnelingViz;
