// HistoryTrails.tsx - Time-lapse ghost surfaces showing optimization history
// Ring buffer of past surface states with progressive transparency

import { useMemo, useRef, useEffect } from 'react';
import * as THREE from 'three';
import type { SurfaceData } from './TensorSurfaceViz';

interface HistoryTrailsProps {
    data: SurfaceData | null;
    maxHistory?: number;
    enabled?: boolean;
    baseOpacity?: number;
}

/**
 * HistoryTrails - Time-lapse visualization of surface evolution
 *
 * Features:
 * - Ring buffer of last N surface states
 * - Progressive transparency (older = more transparent)
 * - Shows optimization trajectory over time
 */
export function HistoryTrails({
    data,
    maxHistory = 5,
    enabled = true,
    baseOpacity = 0.15,
}: HistoryTrailsProps) {
    const historyRef = useRef<SurfaceData[]>([]);
    const updateCountRef = useRef(0);

    // Update history when data changes
    useEffect(() => {
        if (!enabled || !data) return;

        updateCountRef.current += 1;

        // Only store every 3rd update to avoid too many trails
        if (updateCountRef.current % 3 !== 0) return;

        historyRef.current.push(structuredClone(data));

        // Maintain ring buffer
        if (historyRef.current.length > maxHistory) {
            historyRef.current.shift();
        }
    }, [data, enabled, maxHistory]);

    const historyGeometries = useMemo(() => {
        return historyRef.current.map((histData, idx) => {
            const size = histData.x.length;
            const geom = new THREE.PlaneGeometry(10, 10, size - 1, size - 1);
            const positions = geom.attributes.position;

            const { min, max } = histData.stats;
            const range = max - min || 1;

            for (let i = 0; i < size; i++) {
                for (let j = 0; j < size; j++) {
                    const vertIdx = i * size + j;
                    const z = histData.z[i]?.[j] ?? 0;
                    positions.setZ(vertIdx, ((z - min) / range) * 3);
                }
            }

            geom.computeVertexNormals();

            // Calculate opacity based on age (older = more transparent)
            const age = (idx / maxHistory);
            const opacity = baseOpacity * (0.3 + age * 0.7);

            return { geometry: geom, opacity, offset: (maxHistory - idx) * 0.05 };
        });
    }, [historyRef.current.length, maxHistory, baseOpacity]);

    if (!enabled || historyGeometries.length === 0) return null;

    return (
        <group rotation={[-Math.PI / 3, 0, 0]}>
            {historyGeometries.map((hist, idx) => (
                <mesh
                    key={idx}
                    geometry={hist.geometry}
                    position={[0, 0, -hist.offset]}
                >
                    <meshStandardMaterial
                        color="#8b5cf6"
                        transparent
                        opacity={hist.opacity}
                        side={THREE.DoubleSide}
                        depthWrite={false}
                        wireframe
                    />
                </mesh>
            ))}
        </group>
    );
}

export default HistoryTrails;
