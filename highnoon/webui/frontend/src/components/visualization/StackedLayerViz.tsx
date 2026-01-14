// StackedLayerViz.tsx - Display multiple model layers as stacked 3D surfaces
// Vertical arrangement with configurable spacing and transparency

import { useMemo } from 'react';
import { Html } from '@react-three/drei';
import * as THREE from 'three';
import { colorscales, type ColorscaleName } from './colorscales';

export interface LayerSurfaceData {
    layerName: string;
    z: number[][];
    stats: { min: number; max: number; mean: number; std: number };
}

interface StackedLayerVizProps {
    layers: LayerSurfaceData[];
    layerSpacing?: number;
    gridSize?: number;
    colorscale?: ColorscaleName;
    showLabels?: boolean;
    baseOpacity?: number;
    enabled?: boolean;
}

/**
 * StackedLayerViz - Multi-layer 3D visualization
 *
 * Features:
 * - Vertical arrangement with configurable spacing
 * - Transparency gradient (front layers more opaque)
 * - Layer labels in 3D space using Html from drei
 * - Synchronized color mapping across all layers
 */
export function StackedLayerViz({
    layers,
    layerSpacing = 2.5,
    gridSize = 50,
    colorscale = 'viridis',
    showLabels = true,
    baseOpacity = 0.85,
    enabled = true,
}: StackedLayerVizProps) {
    const colorFn = colorscales[colorscale] ?? colorscales.viridis;

    // Compute global min/max for consistent color mapping
    const globalStats = useMemo(() => {
        let globalMin = Infinity;
        let globalMax = -Infinity;

        layers.forEach(layer => {
            globalMin = Math.min(globalMin, layer.stats.min);
            globalMax = Math.max(globalMax, layer.stats.max);
        });

        return { min: globalMin, max: globalMax, range: globalMax - globalMin || 1 };
    }, [layers]);

    // Build geometries for each layer
    const layerGeometries = useMemo(() => {
        return layers.map((layer, idx) => {
            const size = layer.z.length;
            const geom = new THREE.PlaneGeometry(8, 8, size - 1, size - 1);
            const positions = geom.attributes.position;
            const colors = new Float32Array(positions.count * 3);

            const { min, range } = globalStats;

            for (let i = 0; i < size; i++) {
                for (let j = 0; j < size; j++) {
                    const vertIdx = i * size + j;
                    const z = layer.z[i]?.[j] ?? 0;

                    // Set height normalized to global range
                    const normalizedZ = ((z - min) / range) * 2;
                    positions.setZ(vertIdx, normalizedZ);

                    // Apply colorscale
                    const normalized = (z - min) / range;
                    const [r, g, b] = colorFn(normalized);
                    colors[vertIdx * 3] = r;
                    colors[vertIdx * 3 + 1] = g;
                    colors[vertIdx * 3 + 2] = b;
                }
            }

            geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            geom.computeVertexNormals();

            // Calculate opacity: front layers more opaque
            const layerCount = layers.length;
            const opacityFactor = 0.5 + (idx / Math.max(1, layerCount - 1)) * 0.5;
            const opacity = baseOpacity * opacityFactor;

            // Vertical position
            const yOffset = idx * layerSpacing;

            return {
                geometry: geom,
                layerName: layer.layerName,
                opacity,
                yOffset,
                stats: layer.stats,
            };
        });
    }, [layers, globalStats, colorFn, layerSpacing, baseOpacity]);

    if (!enabled || layers.length === 0) return null;

    // Center the stack vertically
    const totalHeight = (layers.length - 1) * layerSpacing;
    const centerOffset = totalHeight / 2;

    return (
        <group rotation={[-Math.PI / 4, 0, 0]} position={[0, 0, 0]}>
            {layerGeometries.map((layer, idx) => (
                <group key={idx} position={[0, layer.yOffset - centerOffset, 0]}>
                    {/* Layer surface */}
                    <mesh geometry={layer.geometry}>
                        <meshStandardMaterial
                            vertexColors
                            side={THREE.DoubleSide}
                            transparent
                            opacity={layer.opacity}
                            metalness={0.1}
                            roughness={0.7}
                            depthWrite={idx === layers.length - 1}
                        />
                    </mesh>

                    {/* Wireframe overlay */}
                    <mesh geometry={layer.geometry} position={[0, 0.01, 0]}>
                        <meshBasicMaterial
                            wireframe
                            color="#ffffff"
                            transparent
                            opacity={0.05}
                        />
                    </mesh>

                    {/* Layer label */}
                    {showLabels && (
                        <Html
                            position={[-5, 0, 1]}
                            center
                            distanceFactor={10}
                            style={{ pointerEvents: 'none' }}
                        >
                            <div className="layer-stack-label">
                                <span className="layer-stack-label__name">{layer.layerName}</span>
                                <span className="layer-stack-label__stats">
                                    μ={layer.stats.mean.toFixed(3)}
                                </span>
                            </div>
                        </Html>
                    )}

                    {/* Connecting lines to next layer */}
                    {idx < layers.length - 1 && (
                        <group>
                            {/* Corner connection lines */}
                            {[[-4, -4], [4, -4], [-4, 4], [4, 4]].map(([x, z], cornerIdx) => (
                                <line key={cornerIdx}>
                                    <bufferGeometry
                                        onUpdate={(self) => {
                                            self.setAttribute(
                                                'position',
                                                new THREE.BufferAttribute(
                                                    new Float32Array([
                                                        x, 0, z,
                                                        x, layerSpacing, z,
                                                    ]),
                                                    3
                                                )
                                            );
                                        }}
                                    />
                                    <lineBasicMaterial
                                        color="#8b5cf6"
                                        transparent
                                        opacity={0.3}
                                    />
                                </line>
                            ))}
                        </group>
                    )}
                </group>
            ))}

            {/* Stack indicator */}
            {showLabels && (
                <Html position={[6, 0, 0]} center>
                    <div className="layer-stack-indicator">
                        <span className="layer-stack-indicator__count">{layers.length}</span>
                        <span className="layer-stack-indicator__label">LAYERS</span>
                    </div>
                </Html>
            )}
        </group>
    );
}

export default StackedLayerViz;
