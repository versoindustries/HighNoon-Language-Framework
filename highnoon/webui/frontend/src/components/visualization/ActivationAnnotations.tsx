// ActivationAnnotations.tsx - 3D labels for important activation regions
// Auto-detects max/min and renders billboarded labels

import { useMemo } from 'react';
import { Html } from '@react-three/drei';
import * as THREE from 'three';

interface ActivationAnnotationsProps {
    surfaceData: {
        z: number[][];
        stats: { min: number; max: number; mean: number };
    };
    enabled?: boolean;
    showMax?: boolean;
    showMin?: boolean;
    showMean?: boolean;
}

interface Annotation {
    position: [number, number, number];
    label: string;
    value: number;
    type: 'max' | 'min' | 'mean';
}

/**
 * ActivationAnnotations - 3D labels for activation hotspots
 *
 * Features:
 * - Auto-detects max/min activation coordinates
 * - Billboard labels that always face camera
 * - Leader lines connecting to surface
 */
export function ActivationAnnotations({
    surfaceData,
    enabled = true,
    showMax = true,
    showMin = true,
    showMean = false,
}: ActivationAnnotationsProps) {
    const annotations = useMemo(() => {
        if (!enabled || !surfaceData.z.length) return [];

        const { z, stats } = surfaceData;
        const size = z.length;
        const { min, max, mean } = stats;
        const range = max - min || 1;
        const result: Annotation[] = [];

        let maxPos = { i: 0, j: 0, val: -Infinity };
        let minPos = { i: 0, j: 0, val: Infinity };

        // Find max and min positions
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const val = z[i]?.[j] ?? 0;
                if (val > maxPos.val) {
                    maxPos = { i, j, val };
                }
                if (val < minPos.val) {
                    minPos = { i, j, val };
                }
            }
        }

        // Convert to 3D coordinates
        const toPos = (i: number, j: number, val: number): [number, number, number] => {
            const x = ((i / size) - 0.5) * 10;
            const y = ((j / size) - 0.5) * 10;
            const zPos = ((val - min) / range) * 3;
            return [x, y, zPos + 0.5];
        };

        if (showMax) {
            result.push({
                position: toPos(maxPos.i, maxPos.j, maxPos.val),
                label: 'MAX',
                value: maxPos.val,
                type: 'max',
            });
        }

        if (showMin) {
            result.push({
                position: toPos(minPos.i, minPos.j, minPos.val),
                label: 'MIN',
                value: minPos.val,
                type: 'min',
            });
        }

        if (showMean) {
            // Find position closest to mean
            let meanPos = { i: 0, j: 0, val: mean, diff: Infinity };
            for (let i = 0; i < size; i++) {
                for (let j = 0; j < size; j++) {
                    const val = z[i]?.[j] ?? 0;
                    const diff = Math.abs(val - mean);
                    if (diff < meanPos.diff) {
                        meanPos = { i, j, val, diff };
                    }
                }
            }
            result.push({
                position: toPos(meanPos.i, meanPos.j, meanPos.val),
                label: 'MEAN',
                value: meanPos.val,
                type: 'mean',
            });
        }

        return result;
    }, [surfaceData, enabled, showMax, showMin, showMean]);

    if (!enabled || annotations.length === 0) return null;

    return (
        <group rotation={[-Math.PI / 3, 0, 0]}>
            {annotations.map((ann, idx) => (
                <group key={idx} position={ann.position}>
                    {/* Leader line */}
                    <line>
                        <bufferGeometry
                            onUpdate={(self) => {
                                self.setAttribute(
                                    'position',
                                    new THREE.BufferAttribute(
                                        new Float32Array([0, 0, 0, 0, 0, 0.8]),
                                        3
                                    )
                                );
                            }}
                        />
                        <lineBasicMaterial
                            color={ann.type === 'max' ? '#22c55e' : ann.type === 'min' ? '#ef4444' : '#3b82f6'}
                            transparent
                            opacity={0.6}
                        />
                    </line>

                    {/* Billboard label */}
                    <Html
                        position={[0, 0, 1]}
                        center
                        distanceFactor={8}
                        style={{ pointerEvents: 'none' }}
                    >
                        <div className={`activation-annotation activation-annotation--${ann.type}`}>
                            <span className="activation-annotation__label">{ann.label}</span>
                            <span className="activation-annotation__value">{ann.value.toFixed(4)}</span>
                        </div>
                    </Html>
                </group>
            ))}
        </group>
    );
}

export default ActivationAnnotations;
