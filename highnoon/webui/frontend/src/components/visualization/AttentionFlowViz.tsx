// AttentionFlowViz.tsx - Animated connections showing attention/information flow
// Uses TubeGeometry for attention paths with animated dash offset

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export interface AttentionConnection {
    fromPosition: [number, number, number];
    toPosition: [number, number, number];
    weight: number; // 0-1 attention strength
    headIndex?: number;
}

interface AttentionFlowVizProps {
    connections: AttentionConnection[];
    enabled?: boolean;
    animationSpeed?: number;
    tubeRadius?: number;
    maxConnections?: number;
    colorByHead?: boolean;
    showParticles?: boolean;
}

// Head colors for multi-head attention visualization
const HEAD_COLORS = [
    '#8b5cf6', // Purple
    '#06b6d4', // Cyan
    '#22c55e', // Green
    '#eab308', // Yellow
    '#ef4444', // Red
    '#ec4899', // Pink
    '#f97316', // Orange
    '#3b82f6', // Blue
];

/**
 * AttentionFlowViz - Animated attention weight visualization
 *
 * Features:
 * - TubeGeometry paths for attention connections
 * - Animated dash offset for flow direction
 * - Color intensity reflects attention strength
 * - Optional particle flow along tubes
 */
export function AttentionFlowViz({
    connections,
    enabled = true,
    animationSpeed = 1.0,
    tubeRadius = 0.03,
    maxConnections = 50,
    colorByHead = true,
    showParticles = true,
}: AttentionFlowVizProps) {
    const groupRef = useRef<THREE.Group>(null);
    const particlesRef = useRef<THREE.Points>(null);

    // Filter and sort connections by weight, take top N
    const topConnections = useMemo(() => {
        return [...connections]
            .sort((a, b) => b.weight - a.weight)
            .slice(0, maxConnections);
    }, [connections, maxConnections]);

    // Build tube geometries for each connection
    const tubes = useMemo(() => {
        return topConnections.map((conn, idx) => {
            const from = new THREE.Vector3(...conn.fromPosition);
            const to = new THREE.Vector3(...conn.toPosition);

            // Create curved path (catmull-rom spline for smooth curves)
            const mid = new THREE.Vector3()
                .addVectors(from, to)
                .multiplyScalar(0.5);

            // Add curve offset based on connection index for visual separation
            const offset = (idx % 5 - 2) * 0.3;
            mid.y += Math.abs(to.z - from.z) * 0.3 + 0.5;
            mid.x += offset;

            const curve = new THREE.CatmullRomCurve3([from, mid, to]);
            const tubeGeom = new THREE.TubeGeometry(
                curve,
                20, // tubular segments
                tubeRadius * (0.5 + conn.weight * 0.5), // radius based on weight
                8, // radial segments
                false // closed
            );

            // Color based on head or weight
            let color: string;
            if (colorByHead && conn.headIndex !== undefined) {
                color = HEAD_COLORS[conn.headIndex % HEAD_COLORS.length];
            } else {
                // Interpolate from blue (weak) to purple (strong)
                const hue = 0.75 - conn.weight * 0.15; // 270deg to 220deg
                color = `hsl(${hue * 360}, 80%, 60%)`;
            }

            return {
                geometry: tubeGeom,
                curve,
                color,
                weight: conn.weight,
                opacity: 0.3 + conn.weight * 0.5,
            };
        });
    }, [topConnections, tubeRadius, colorByHead]);

    // Create particles that flow along the tubes
    const particles = useMemo(() => {
        if (!showParticles || tubes.length === 0) return null;

        const particleCount = Math.min(tubes.length * 3, 100);
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        const sizes = new Float32Array(particleCount);
        const progress = new Float32Array(particleCount); // 0-1 along curve

        for (let i = 0; i < particleCount; i++) {
            const tubeIdx = i % tubes.length;
            const tube = tubes[tubeIdx];
            const t = Math.random();

            const point = tube.curve.getPoint(t);
            positions[i * 3] = point.x;
            positions[i * 3 + 1] = point.y;
            positions[i * 3 + 2] = point.z;

            const color = new THREE.Color(tube.color);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;

            sizes[i] = 0.04 + tube.weight * 0.04;
            progress[i] = t;
        }

        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geom.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

        return { geometry: geom, progress };
    }, [tubes, showParticles]);

    // Animation loop
    useFrame((state, delta) => {
        if (!enabled) return;

        // Animate particles along curves
        if (particles && particlesRef.current) {
            const positions = particles.geometry.attributes.position.array as Float32Array;
            const progress = particles.progress;

            for (let i = 0; i < progress.length; i++) {
                // Advance progress
                progress[i] += delta * animationSpeed * 0.3;
                if (progress[i] > 1) progress[i] = 0;

                // Update position along curve
                const tubeIdx = i % tubes.length;
                const tube = tubes[tubeIdx];
                const point = tube.curve.getPoint(progress[i]);

                positions[i * 3] = point.x;
                positions[i * 3 + 1] = point.y;
                positions[i * 3 + 2] = point.z;
            }

            particles.geometry.attributes.position.needsUpdate = true;
        }
    });

    if (!enabled || tubes.length === 0) return null;

    return (
        <group ref={groupRef}>
            {/* Attention flow tubes */}
            {tubes.map((tube, idx) => (
                <mesh key={idx} geometry={tube.geometry}>
                    <meshStandardMaterial
                        color={tube.color}
                        emissive={tube.color}
                        emissiveIntensity={0.3}
                        transparent
                        opacity={tube.opacity}
                        side={THREE.DoubleSide}
                    />
                </mesh>
            ))}

            {/* Flow particles */}
            {particles && (
                <points ref={particlesRef} geometry={particles.geometry}>
                    <pointsMaterial
                        size={0.08}
                        vertexColors
                        transparent
                        opacity={0.9}
                        sizeAttenuation
                        depthWrite={false}
                        blending={THREE.AdditiveBlending}
                    />
                </points>
            )}

            {/* Connection endpoints (spheres) */}
            {topConnections.slice(0, 20).map((conn, idx) => (
                <group key={`endpoints-${idx}`}>
                    {/* Source point */}
                    <mesh position={conn.fromPosition}>
                        <sphereGeometry args={[0.06, 8, 8]} />
                        <meshStandardMaterial
                            color="#22c55e"
                            emissive="#22c55e"
                            emissiveIntensity={0.5}
                        />
                    </mesh>
                    {/* Target point */}
                    <mesh position={conn.toPosition}>
                        <sphereGeometry args={[0.06, 8, 8]} />
                        <meshStandardMaterial
                            color="#ef4444"
                            emissive="#ef4444"
                            emissiveIntensity={0.5}
                        />
                    </mesh>
                </group>
            ))}
        </group>
    );
}

export default AttentionFlowViz;
