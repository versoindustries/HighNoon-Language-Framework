// GradientArrows.tsx - 3D arrow field showing gradient direction
// Visualizes gradient flow as arrows on the surface

import { useMemo } from 'react';
import * as THREE from 'three';

interface GradientArrowsProps {
    surfaceData: {
        z: number[][];
        stats: { min: number; max: number };
    };
    gridSpacing?: number;
    arrowScale?: number;
    gradientNorm?: number;
    enabled?: boolean;
    color?: string;
}

/**
 * GradientArrows - 3D gradient direction visualization
 *
 * Features:
 * - Computes local gradient from surface height
 * - Arrow length proportional to gradient magnitude
 * - Color indicates gradient strength
 */
export function GradientArrows({
    surfaceData,
    gridSpacing = 3,
    arrowScale = 0.5,
    gradientNorm = 1.0,
    enabled = true,
    color = '#00ff88',
}: GradientArrowsProps) {
    const arrows = useMemo(() => {
        if (!enabled || !surfaceData.z.length) return [];

        const { z, stats } = surfaceData;
        const size = z.length;
        const { min, max } = stats;
        const range = max - min || 1;
        const arrowList: { position: [number, number, number]; direction: [number, number, number]; magnitude: number }[] = [];

        // Sample gradient at grid points
        for (let i = gridSpacing; i < size - gridSpacing; i += gridSpacing) {
            for (let j = gridSpacing; j < size - gridSpacing; j += gridSpacing) {
                // Compute local gradient via finite differences
                const dx = (z[i + 1]?.[j] ?? 0) - (z[i - 1]?.[j] ?? 0);
                const dy = (z[i]?.[j + 1] ?? 0) - (z[i]?.[j - 1] ?? 0);

                const gradMag = Math.sqrt(dx * dx + dy * dy);
                if (gradMag < 0.01) continue; // Skip flat regions

                // Normalize direction
                const nx = dx / gradMag;
                const ny = dy / gradMag;

                // Map grid indices to 3D coordinates
                const x = ((i / size) - 0.5) * 10;
                const y = ((j / size) - 0.5) * 10;
                const zPos = ((z[i]?.[j] ?? 0) - min) / range * 3;

                arrowList.push({
                    position: [x, y, zPos + 0.2],
                    direction: [nx * arrowScale, ny * arrowScale, 0],
                    magnitude: Math.min(gradMag * gradientNorm, 1),
                });
            }
        }

        return arrowList;
    }, [surfaceData, gridSpacing, arrowScale, gradientNorm, enabled]);

    if (!enabled || arrows.length === 0) return null;

    return (
        <group rotation={[-Math.PI / 3, 0, 0]}>
            {arrows.map((arrow, idx) => (
                <Arrow
                    key={idx}
                    position={arrow.position}
                    direction={arrow.direction}
                    magnitude={arrow.magnitude}
                    color={color}
                />
            ))}
        </group>
    );
}

interface ArrowProps {
    position: [number, number, number];
    direction: [number, number, number];
    magnitude: number;
    color: string;
}

function Arrow({ position, direction, magnitude, color }: ArrowProps) {
    const arrowHelper = useMemo(() => {
        const dir = new THREE.Vector3(...direction).normalize();
        const length = magnitude * 0.8;
        const arrowColor = new THREE.Color(color);

        // Blend toward red for high magnitude
        if (magnitude > 0.7) {
            arrowColor.lerp(new THREE.Color('#ff4444'), (magnitude - 0.7) / 0.3);
        }

        return { dir, length, color: arrowColor };
    }, [direction, magnitude, color]);

    return (
        <arrowHelper
            args={[
                arrowHelper.dir,
                new THREE.Vector3(...position),
                arrowHelper.length,
                arrowHelper.color.getHex(),
                arrowHelper.length * 0.3,
                arrowHelper.length * 0.15,
            ]}
        />
    );
}

export default GradientArrows;
