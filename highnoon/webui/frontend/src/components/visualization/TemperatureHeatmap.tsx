// TemperatureHeatmap.tsx - Annealing temperature overlay visualization
// Heat color gradient on loss surface with cooling animation

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';

interface TemperatureHeatmapProps {
    temperature?: number;
    maxTemperature?: number;
    enabled?: boolean;
    showGauge?: boolean;
}

/**
 * TemperatureHeatmap - Simulated annealing temperature visualization
 *
 * Features:
 * - Heat gradient plane overlaying the scene
 * - "Cooling" particle effect as temperature decreases
 * - Temperature gauge display
 * - Color transitions from hot (red) to cold (blue)
 */
export function TemperatureHeatmap({
    temperature = 1.0,
    maxTemperature = 2.0,
    enabled = true,
    showGauge = true,
}: TemperatureHeatmapProps) {
    const heatPlaneRef = useRef<THREE.Mesh>(null);
    const particlesRef = useRef<THREE.Points>(null);

    // Normalized temperature (0 = cold, 1 = hot)
    const normalizedTemp = Math.min(temperature / maxTemperature, 1);

    // Temperature-based color
    const tempColor = useMemo(() => {
        const color = new THREE.Color();
        // Interpolate from blue (cold) through yellow to red (hot)
        if (normalizedTemp < 0.5) {
            color.setHSL(0.6 - normalizedTemp * 0.4, 0.9, 0.5); // Blue to cyan
        } else {
            color.setHSL(0.15 - (normalizedTemp - 0.5) * 0.3, 0.9, 0.5); // Yellow to red
        }
        return color;
    }, [normalizedTemp]);

    // Cooling particles
    const { particleGeometry, particlePositions } = useMemo(() => {
        const count = 100;
        const positions = new Float32Array(count * 3);

        for (let i = 0; i < count; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 10;
            positions[i * 3 + 1] = Math.random() * 4;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 10;
        }

        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        return { particleGeometry: geom, particlePositions: positions };
    }, []);

    // Animation loop
    useFrame((state, delta) => {
        // Heat plane opacity based on temperature
        if (heatPlaneRef.current) {
            const mat = heatPlaneRef.current.material as THREE.MeshBasicMaterial;
            mat.opacity = normalizedTemp * 0.15;
            mat.color = tempColor;
        }

        // Particle falling animation (only visible when hot)
        if (particlesRef.current && normalizedTemp > 0.3) {
            const positions = particleGeometry.attributes.position.array as Float32Array;

            for (let i = 0; i < positions.length / 3; i++) {
                const idx = i * 3;
                // Fall speed based on temperature
                positions[idx + 1] -= delta * normalizedTemp * 2;

                // Respawn at top when below ground
                if (positions[idx + 1] < 0) {
                    positions[idx] = (Math.random() - 0.5) * 10;
                    positions[idx + 1] = 4 + Math.random() * 2;
                    positions[idx + 2] = (Math.random() - 0.5) * 10;
                }
            }

            particleGeometry.attributes.position.needsUpdate = true;
        }
    });

    if (!enabled) return null;

    return (
        <group>
            {/* Heat overlay plane */}
            <mesh
                ref={heatPlaneRef}
                rotation={[-Math.PI / 2, 0, 0]}
                position={[0, 3.5, 0]}
            >
                <planeGeometry args={[12, 12]} />
                <meshBasicMaterial
                    color={tempColor}
                    transparent
                    opacity={normalizedTemp * 0.15}
                    side={THREE.DoubleSide}
                    depthWrite={false}
                    blending={THREE.AdditiveBlending}
                />
            </mesh>

            {/* Rising heat particles */}
            {normalizedTemp > 0.3 && (
                <points ref={particlesRef} geometry={particleGeometry}>
                    <pointsMaterial
                        size={0.06}
                        color={tempColor}
                        transparent
                        opacity={normalizedTemp * 0.5}
                        sizeAttenuation
                        depthWrite={false}
                        blending={THREE.AdditiveBlending}
                    />
                </points>
            )}

            {/* Temperature gauge */}
            {showGauge && (
                <Html position={[-5, 3, 0]} center>
                    <div className="temperature-gauge">
                        <div className="temperature-gauge__label">TEMP</div>
                        <div className="temperature-gauge__bar-container">
                            <div
                                className="temperature-gauge__bar"
                                style={{
                                    height: `${normalizedTemp * 100}%`,
                                    background: `linear-gradient(to top, #3b82f6, #eab308, #ef4444)`,
                                }}
                            />
                        </div>
                        <div className="temperature-gauge__value">
                            {temperature.toFixed(2)}
                        </div>
                        <div className="temperature-gauge__mode">
                            {normalizedTemp > 0.7 ? 'HOT' : normalizedTemp > 0.3 ? 'WARM' : 'COOL'}
                        </div>
                    </div>
                </Html>
            )}
        </group>
    );
}

export default TemperatureHeatmap;
