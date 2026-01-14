// LossLandscape3D.tsx - 3D loss surface with current position marker
// Visualizes the loss landscape with animated optimization trajectory
// Enhanced with fog, terrain coloring, and ribbon trails

import { useRef, useMemo, useState, useEffect, useCallback } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Html, Trail, Line } from '@react-three/drei';
import * as THREE from 'three';
import { colorscales } from './colorscales';

interface LossLandscapeProps {
    landscapeData?: {
        x: number[];
        y: number[];
        z: number[][];
    } | null;
    currentPosition?: { x: number; y: number; z: number } | null;
    bestPosition?: { x: number; y: number; z: number } | null;
    trajectoryHistory?: Array<{ x: number; y: number; z: number }>;
    enabled?: boolean;
    showTrajectory?: boolean;
    showWireframe?: boolean;
    showFog?: boolean;
    autoGenerate?: boolean;
    terrainStyle?: 'plasma' | 'terrain' | 'viridis';
}

// Terrain color bands for elevation mapping
const terrainColorBands = [
    { threshold: 0.0, color: new THREE.Color('#1e3a5f') },  // Deep blue (valleys)
    { threshold: 0.2, color: new THREE.Color('#22c55e') },  // Green (low elevation)
    { threshold: 0.4, color: new THREE.Color('#84cc16') },  // Lime
    { threshold: 0.6, color: new THREE.Color('#eab308') },  // Yellow
    { threshold: 0.8, color: new THREE.Color('#f97316') },  // Orange
    { threshold: 1.0, color: new THREE.Color('#ef4444') },  // Red (peaks)
];

/**
 * Get terrain color based on normalized height value
 */
function getTerrainColor(normalized: number): [number, number, number] {
    // Find the two bands we're between
    for (let i = 1; i < terrainColorBands.length; i++) {
        if (normalized <= terrainColorBands[i].threshold) {
            const prev = terrainColorBands[i - 1];
            const curr = terrainColorBands[i];
            const t = (normalized - prev.threshold) / (curr.threshold - prev.threshold);
            const color = prev.color.clone().lerp(curr.color, t);
            return [color.r, color.g, color.b];
        }
    }
    const last = terrainColorBands[terrainColorBands.length - 1].color;
    return [last.r, last.g, last.b];
}

/**
 * LossLandscape3D - Interactive 3D loss landscape visualization
 *
 * Features:
 * - 3D surface representing loss function
 * - Atmospheric fog for depth perception
 * - Terrain-style height-based coloring
 * - Animated sphere marker at current hyperparameter position
 * - Ribbon trail showing optimization trajectory
 * - Best position marker with golden beacon glow
 * - Togglable wireframe overlay
 */
export function LossLandscape3D({
    landscapeData,
    currentPosition,
    bestPosition,
    trajectoryHistory = [],
    enabled = true,
    showTrajectory = true,
    showWireframe = true,
    showFog = true,
    autoGenerate = true,
    terrainStyle = 'terrain',
}: LossLandscapeProps) {
    const { scene } = useThree();
    const currentMarkerRef = useRef<THREE.Mesh>(null);
    const bestMarkerRef = useRef<THREE.Mesh>(null);
    const bestBeaconRef = useRef<THREE.Points>(null);
    const [wireframeVisible, setWireframeVisible] = useState(showWireframe);

    // Configure scene fog
    useEffect(() => {
        if (showFog && enabled) {
            scene.fog = new THREE.Fog('#0a0a0f', 10, 50);
        }
        return () => {
            scene.fog = null;
        };
    }, [scene, showFog, enabled]);

    // Generate synthetic landscape if no data provided
    const landscape = useMemo(() => {
        if (landscapeData) return landscapeData;
        if (!autoGenerate) return null;

        // Generate a synthetic Rastrigin-like loss landscape for visualization
        const gridSize = 50;
        const x = Array.from({ length: gridSize }, (_, i) => (i / gridSize - 0.5) * 4);
        const y = Array.from({ length: gridSize }, (_, i) => (i / gridSize - 0.5) * 4);
        const z: number[][] = [];

        for (let i = 0; i < gridSize; i++) {
            z[i] = [];
            for (let j = 0; j < gridSize; j++) {
                // Rastrigin-like function with multiple local minima
                const xi = x[i];
                const yj = y[j];
                z[i][j] = 20 + xi * xi + yj * yj -
                    10 * (Math.cos(2 * Math.PI * xi) + Math.cos(2 * Math.PI * yj));
            }
        }

        return { x, y, z };
    }, [landscapeData, autoGenerate]);

    // Build geometry from landscape data
    const { geometry, stats } = useMemo(() => {
        if (!landscape) {
            return { geometry: null, stats: { min: 0, max: 1 } };
        }

        const size = landscape.x.length;
        const geom = new THREE.PlaneGeometry(10, 10, size - 1, size - 1);
        const positions = geom.attributes.position;
        const colors = new Float32Array(positions.count * 3);

        // Find min/max for normalization
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const val = landscape.z[i]?.[j] ?? 0;
                min = Math.min(min, val);
                max = Math.max(max, val);
            }
        }
        const range = max - min || 1;

        // Choose color function based on style
        const getColor = terrainStyle === 'terrain'
            ? getTerrainColor
            : (n: number) => colorscales[terrainStyle === 'viridis' ? 'viridis' : 'plasma'](n);

        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const idx = i * size + j;
                const val = landscape.z[i]?.[j] ?? 0;
                const normalized = (val - min) / range;

                // Set height (inverted so lower loss is higher)
                positions.setZ(idx, (1 - normalized) * 4);

                // Color mapping based on style
                const [r, g, b] = getColor(normalized);
                colors[idx * 3] = r;
                colors[idx * 3 + 1] = g;
                colors[idx * 3 + 2] = b;
            }
        }

        geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geom.computeVertexNormals();

        return { geometry: geom, stats: { min, max } };
    }, [landscape, terrainStyle]);

    // Create beacon particles for best position
    const beaconParticles = useMemo(() => {
        const count = 20;
        const positions = new Float32Array(count * 3);
        const velocities = new Float32Array(count);

        for (let i = 0; i < count; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 0.3;
            positions[i * 3 + 1] = Math.random() * 0.5;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 0.3;
            velocities[i] = 0.2 + Math.random() * 0.3;
        }

        return { positions, velocities, count };
    }, []);

    // Animate markers and beacon
    useFrame((state) => {
        // Bounce animation for current position
        if (currentMarkerRef.current && currentPosition) {
            currentMarkerRef.current.position.y =
                currentPosition.z + Math.sin(state.clock.elapsedTime * 3) * 0.1 + 0.3;
        }

        // Pulsing glow for best position
        if (bestMarkerRef.current) {
            const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.1;
            bestMarkerRef.current.scale.setScalar(scale);
        }

        // Animate beacon particles
        if (bestBeaconRef.current && bestPosition) {
            const positions = bestBeaconRef.current.geometry.attributes.position;
            for (let i = 0; i < beaconParticles.count; i++) {
                let y = positions.getY(i);
                y += beaconParticles.velocities[i] * 0.02;
                if (y > 1.5) {
                    y = 0;
                    positions.setX(i, (Math.random() - 0.5) * 0.3);
                    positions.setZ(i, (Math.random() - 0.5) * 0.3);
                }
                positions.setY(i, y);
            }
            positions.needsUpdate = true;
        }
    });

    // Create trajectory ribbon curve
    const trajectoryCurve = useMemo(() => {
        if (!showTrajectory || trajectoryHistory.length < 2) return null;

        const points = trajectoryHistory.map(p =>
            new THREE.Vector3(p.x * 2.5, p.z + 0.2, p.y * 2.5)
        );

        return new THREE.CatmullRomCurve3(points);
    }, [trajectoryHistory, showTrajectory]);

    // Toggle wireframe with keyboard (handled by parent)
    const toggleWireframe = useCallback(() => {
        setWireframeVisible(prev => !prev);
    }, []);

    if (!enabled || !geometry) return null;

    return (
        <group rotation={[-Math.PI / 4, 0, 0]}>
            {/* Loss landscape surface with enhanced materials */}
            <mesh geometry={geometry}>
                <meshStandardMaterial
                    vertexColors
                    side={THREE.DoubleSide}
                    metalness={0.2}
                    roughness={0.6}
                    transparent
                    opacity={0.95}
                    envMapIntensity={0.3}
                />
            </mesh>

            {/* Wireframe overlay - toggleable */}
            {wireframeVisible && (
                <mesh geometry={geometry} position={[0, 0.01, 0]}>
                    <meshBasicMaterial
                        wireframe
                        color="#00f5ff"
                        transparent
                        opacity={0.12}
                    />
                </mesh>
            )}

            {/* Current position marker with trail */}
            {currentPosition && (
                <group position={[currentPosition.x * 2.5, 0, currentPosition.y * 2.5]}>
                    <Trail
                        width={0.4}
                        length={12}
                        color="#00f5ff"
                        attenuation={(t) => t * t}
                    >
                        <mesh ref={currentMarkerRef}>
                            <sphereGeometry args={[0.18, 24, 24]} />
                            <meshStandardMaterial
                                color="#00f5ff"
                                emissive="#00f5ff"
                                emissiveIntensity={0.8}
                                metalness={0.3}
                                roughness={0.2}
                            />
                        </mesh>
                    </Trail>
                    {/* Velocity indicator glow */}
                    <mesh position={[0, currentPosition.z + 0.3, 0]}>
                        <sphereGeometry args={[0.25, 16, 16]} />
                        <meshBasicMaterial
                            color="#00f5ff"
                            transparent
                            opacity={0.2}
                        />
                    </mesh>
                    <Html position={[0, 0.9, 0]} center>
                        <div className="landscape-marker landscape-marker--current">
                            CURRENT
                        </div>
                    </Html>
                </group>
            )}

            {/* Best position marker with golden beacon */}
            {bestPosition && (
                <group position={[bestPosition.x * 2.5, bestPosition.z + 0.3, bestPosition.y * 2.5]}>
                    {/* Main marker sphere */}
                    <mesh ref={bestMarkerRef}>
                        <sphereGeometry args={[0.22, 24, 24]} />
                        <meshStandardMaterial
                            color="#fbbf24"
                            emissive="#fbbf24"
                            emissiveIntensity={1.0}
                            metalness={0.5}
                            roughness={0.2}
                        />
                    </mesh>

                    {/* Glow ring */}
                    <mesh rotation={[Math.PI / 2, 0, 0]}>
                        <ringGeometry args={[0.28, 0.4, 32]} />
                        <meshBasicMaterial
                            color="#fbbf24"
                            transparent
                            opacity={0.4}
                            side={THREE.DoubleSide}
                        />
                    </mesh>

                    {/* Rising particle beacon */}
                    <points ref={bestBeaconRef}>
                        <bufferGeometry>
                            <bufferAttribute
                                attach="attributes-position"
                                args={[beaconParticles.positions, 3]}
                            />
                        </bufferGeometry>
                        <pointsMaterial
                            color="#fbbf24"
                            size={0.08}
                            transparent
                            opacity={0.8}
                            sizeAttenuation
                        />
                    </points>

                    <Html position={[0, 0.7, 0]} center>
                        <div className="landscape-marker landscape-marker--best">
                            ⭐ BEST
                        </div>
                    </Html>
                </group>
            )}

            {/* Enhanced ribbon trajectory using smooth curve */}
            {trajectoryCurve && (
                <group>
                    {/* Main trajectory line */}
                    <Line
                        points={trajectoryCurve.getPoints(Math.max(50, trajectoryHistory.length * 2))}
                        color="#a855f7"
                        lineWidth={3}
                        transparent
                        opacity={0.7}
                    />
                    {/* Glow line underneath */}
                    <Line
                        points={trajectoryCurve.getPoints(Math.max(50, trajectoryHistory.length * 2))}
                        color="#a855f7"
                        lineWidth={8}
                        transparent
                        opacity={0.2}
                    />
                </group>
            )}

            {/* Fallback simple trajectory if Line not available */}
            {showTrajectory && !trajectoryCurve && trajectoryHistory.length > 1 && (
                <line>
                    <bufferGeometry
                        onUpdate={(self) => {
                            self.setAttribute(
                                'position',
                                new THREE.BufferAttribute(
                                    new Float32Array(
                                        trajectoryHistory.flatMap(p => [p.x * 2.5, p.z + 0.2, p.y * 2.5])
                                    ),
                                    3
                                )
                            );
                        }}
                    />
                    <lineBasicMaterial
                        color="#a855f7"
                        transparent
                        opacity={0.6}
                    />
                </line>
            )}
        </group>
    );
}

export default LossLandscape3D;
