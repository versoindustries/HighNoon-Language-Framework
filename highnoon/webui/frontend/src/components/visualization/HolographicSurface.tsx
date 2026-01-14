// HolographicSurface.tsx - Sci-fi holographic wireframe rendering mode
// Neon glow, Fresnel edges, animated scanlines

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { SurfaceData } from './TensorSurfaceViz';

interface HolographicSurfaceProps {
    data: SurfaceData;
    primaryColor?: string;
    secondaryColor?: string;
    scanlineSpeed?: number;
    glowIntensity?: number;
    wireframeOpacity?: number;
    autoRotate?: boolean;
}

/**
 * HolographicSurface - Sci-fi holographic visualization mode
 *
 * Features:
 * - Wireframe mesh with neon glow
 * - Fresnel edge highlighting
 * - Animated horizontal scanlines
 * - Customizable colors and intensity
 */
export function HolographicSurface({
    data,
    primaryColor = '#00ffff',
    secondaryColor = '#ff00ff',
    scanlineSpeed = 1.0,
    glowIntensity = 0.8,
    wireframeOpacity = 0.9,
    autoRotate = false,
}: HolographicSurfaceProps) {
    const meshRef = useRef<THREE.Mesh>(null);
    const wireframeMeshRef = useRef<THREE.LineSegments>(null);
    const scanlineRef = useRef<THREE.Mesh>(null);

    // Build geometry from surface data
    const geometry = useMemo(() => {
        const size = data.x.length;
        const geom = new THREE.PlaneGeometry(10, 10, size - 1, size - 1);
        const positions = geom.attributes.position;

        const { min, max } = data.stats;
        const range = max - min || 1;

        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const idx = i * size + j;
                const z = data.z[i]?.[j] ?? 0;
                positions.setZ(idx, ((z - min) / range) * 3);
            }
        }

        geom.computeVertexNormals();
        return geom;
    }, [data]);

    // Wireframe geometry
    const wireframeGeometry = useMemo(() => {
        return new THREE.WireframeGeometry(geometry);
    }, [geometry]);

    // Scanline plane
    const scanlineGeometry = useMemo(() => {
        return new THREE.PlaneGeometry(12, 0.1);
    }, []);

    // Animation loop
    useFrame((state, delta) => {
        // Scanline animation
        if (scanlineRef.current) {
            scanlineRef.current.position.y = Math.sin(state.clock.elapsedTime * scanlineSpeed) * 5;
        }

        // Auto-rotation
        if (autoRotate) {
            if (meshRef.current) meshRef.current.rotation.z += delta * 0.05;
            if (wireframeMeshRef.current) wireframeMeshRef.current.rotation.z += delta * 0.05;
        }

        // Pulsing glow effect
        if (wireframeMeshRef.current) {
            const material = wireframeMeshRef.current.material as THREE.LineBasicMaterial;
            const pulse = 0.7 + Math.sin(state.clock.elapsedTime * 3) * 0.3;
            material.opacity = wireframeOpacity * pulse;
        }
    });

    return (
        <group rotation={[-Math.PI / 3, 0, 0]}>
            {/* Base holographic surface - semi-transparent */}
            <mesh ref={meshRef} geometry={geometry}>
                <meshStandardMaterial
                    color={primaryColor}
                    emissive={primaryColor}
                    emissiveIntensity={glowIntensity * 0.3}
                    transparent
                    opacity={0.15}
                    side={THREE.DoubleSide}
                    depthWrite={false}
                />
            </mesh>

            {/* Wireframe overlay */}
            <lineSegments ref={wireframeMeshRef} geometry={wireframeGeometry}>
                <lineBasicMaterial
                    color={primaryColor}
                    transparent
                    opacity={wireframeOpacity}
                    linewidth={1}
                />
            </lineSegments>

            {/* Edge highlight wireframe (secondary color) */}
            <lineSegments geometry={wireframeGeometry} position={[0, 0, 0.02]}>
                <lineBasicMaterial
                    color={secondaryColor}
                    transparent
                    opacity={wireframeOpacity * 0.3}
                    linewidth={1}
                />
            </lineSegments>

            {/* Horizontal scanline */}
            <mesh ref={scanlineRef} geometry={scanlineGeometry} position={[0, 0, 2]}>
                <meshBasicMaterial
                    color="#ffffff"
                    transparent
                    opacity={0.3}
                    side={THREE.DoubleSide}
                    depthWrite={false}
                    blending={THREE.AdditiveBlending}
                />
            </mesh>
        </group>
    );
}

export default HolographicSurface;
