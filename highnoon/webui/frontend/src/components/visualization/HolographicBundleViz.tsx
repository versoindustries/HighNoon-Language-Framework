// HolographicBundleViz.tsx - HD Encoding CTQW Visualization
// Premium 3D visualization of HSMN's Holographic Dense encoding process

import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Trail, Float } from '@react-three/drei';
import { useMemo, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import './HolographicBundleViz.css';

/** Props for the HolographicBundleViz component */
interface HolographicBundleVizProps {
    /** Number of tokens being encoded */
    numTokens?: number;
    /** HD dimension size */
    hdDim?: number;
    /** Embedding dimension */
    embedDim?: number;
    /** Enable auto-rotation */
    autoRotate?: boolean;
    /** Container height */
    height?: number;
    /** Export callback */
    onExport?: (dataUrl: string) => void;
}

/** Single particle in the CTQW quantum walk */
function QuantumParticle({
    initialPos,
    color,
    delay,
    spreadRadius
}: {
    initialPos: [number, number, number];
    color: string;
    delay: number;
    spreadRadius: number;
}) {
    const meshRef = useRef<THREE.Mesh>(null);
    const time = useRef(0);
    const phase = useRef(Math.random() * Math.PI * 2);

    useFrame((_, delta) => {
        if (!meshRef.current) return;
        time.current += delta;

        const t = Math.max(0, time.current - delay);
        const spreadFactor = 1 - Math.exp(-t * 0.5);

        // CTQW spreading pattern - interference-like motion
        const angle = phase.current + t * 0.8;
        const radius = spreadRadius * spreadFactor;

        meshRef.current.position.x = initialPos[0] + Math.cos(angle) * radius * Math.sin(t * 2);
        meshRef.current.position.y = initialPos[1] + Math.sin(angle) * radius * Math.cos(t * 1.5);
        meshRef.current.position.z = initialPos[2] + Math.sin(t * 0.7 + phase.current) * radius * 0.5;

        // Pulsing opacity based on interference
        const interference = Math.abs(Math.sin(t * 3 + phase.current));
        meshRef.current.scale.setScalar(0.08 + interference * 0.04);
    });

    return (
        <Trail
            width={0.3}
            length={6}
            color={new THREE.Color(color)}
            attenuation={(t) => t * t}
        >
            <mesh ref={meshRef}>
                <sphereGeometry args={[0.1, 16, 16]} />
                <meshStandardMaterial
                    color={color}
                    emissive={color}
                    emissiveIntensity={0.6}
                    transparent
                    opacity={0.9}
                />
            </mesh>
        </Trail>
    );
}

/** Phase rotation ring showing position encoding */
function PhaseRing({ radius, phase, color }: { radius: number; phase: number; color: string }) {
    const ringRef = useRef<THREE.Mesh>(null);

    useFrame((_, delta) => {
        if (ringRef.current) {
            ringRef.current.rotation.z += delta * 0.3;
        }
    });

    return (
        <mesh ref={ringRef} rotation={[Math.PI / 2, 0, phase]}>
            <torusGeometry args={[radius, 0.02, 8, 64]} />
            <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.4}
                transparent
                opacity={0.6}
            />
        </mesh>
    );
}

/** Central bundle node showing superposition */
function BundleCore({ intensity }: { intensity: number }) {
    const coreRef = useRef<THREE.Mesh>(null);

    useFrame((state) => {
        if (coreRef.current) {
            const pulse = Math.sin(state.clock.elapsedTime * 2) * 0.1 + 1;
            coreRef.current.scale.setScalar(0.5 * pulse);
        }
    });

    return (
        <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
            <mesh ref={coreRef}>
                <icosahedronGeometry args={[0.5, 2]} />
                <meshStandardMaterial
                    color="#6366f1"
                    emissive="#6366f1"
                    emissiveIntensity={intensity}
                    transparent
                    opacity={0.8}
                    wireframe
                />
            </mesh>
            <mesh>
                <icosahedronGeometry args={[0.35, 1]} />
                <meshStandardMaterial
                    color="#8b5cf6"
                    emissive="#8b5cf6"
                    emissiveIntensity={0.6}
                    transparent
                    opacity={0.6}
                />
            </mesh>
        </Float>
    );
}

/** Main scene containing all holographic elements */
function HolographicScene({
    numTokens,
    autoRotate
}: {
    numTokens: number;
    autoRotate: boolean;
}) {
    const groupRef = useRef<THREE.Group>(null);

    useFrame((_, delta) => {
        if (autoRotate && groupRef.current) {
            groupRef.current.rotation.y += delta * 0.1;
        }
    });

    // Generate particles for CTQW visualization
    const particles = useMemo(() => {
        const colors = ['#6366f1', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b'];
        const result = [];

        for (let i = 0; i < numTokens; i++) {
            const angle = (i / numTokens) * Math.PI * 2;
            const radius = 2;
            result.push({
                initialPos: [
                    Math.cos(angle) * radius,
                    Math.sin(angle) * radius,
                    0
                ] as [number, number, number],
                color: colors[i % colors.length],
                delay: i * 0.1,
                spreadRadius: 1.5 + Math.random() * 0.5,
            });
        }
        return result;
    }, [numTokens]);

    // Phase rings for position encoding visualization
    const phaseRings = useMemo(() => {
        return [
            { radius: 2.5, phase: 0, color: '#6366f1' },
            { radius: 3.0, phase: Math.PI / 4, color: '#8b5cf6' },
            { radius: 3.5, phase: Math.PI / 2, color: '#a78bfa' },
        ];
    }, []);

    return (
        <group ref={groupRef}>
            {/* Central bundle core */}
            <BundleCore intensity={0.8} />

            {/* CTQW particles */}
            {particles.map((p, i) => (
                <QuantumParticle key={i} {...p} />
            ))}

            {/* Phase rotation rings */}
            {phaseRings.map((ring, i) => (
                <PhaseRing key={i} {...ring} />
            ))}

            {/* Ambient grid for depth perception */}
            <gridHelper args={[10, 20, '#1a1a2e', '#0a0a12']} rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -3]} />
        </group>
    );
}

/**
 * HolographicBundleViz - Premium visualization of HD encoding process
 *
 * Displays the holographic dense encoding with:
 * - CTQW quantum walk particle spreading
 * - Phase rotation rings for position encoding
 * - Central bundle superposition node
 * - Compression ratio statistics
 */
export function HolographicBundleViz({
    numTokens = 8,
    hdDim = 10000,
    embedDim = 512,
    autoRotate = true,
    height = 350,
    onExport,
}: HolographicBundleVizProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isExporting, setIsExporting] = useState(false);

    const compressionRatio = ((numTokens * embedDim) / hdDim).toFixed(2);

    const handleExport = useCallback(() => {
        if (!canvasRef.current || !onExport) return;
        setIsExporting(true);

        requestAnimationFrame(() => {
            const dataUrl = canvasRef.current?.toDataURL('image/png');
            if (dataUrl) onExport(dataUrl);
            setIsExporting(false);
        });
    }, [onExport]);

    return (
        <div className="holographic-bundle">
            <div className="holographic-bundle__header">
                <div className="holographic-bundle__title">
                    <span className="holographic-badge">HD-CTQW</span>
                    Holographic Bundle Encoding
                </div>
                <div className="holographic-bundle__actions">
                    {onExport && (
                        <button
                            onClick={handleExport}
                            disabled={isExporting}
                            className="holographic-export-btn"
                        >
                            📷 {isExporting ? 'Exporting...' : 'Export'}
                        </button>
                    )}
                </div>
            </div>

            <div className="holographic-bundle__canvas" style={{ height }}>
                <Canvas
                    ref={canvasRef}
                    gl={{ preserveDrawingBuffer: true, antialias: true }}
                    camera={{ position: [0, 0, 8], fov: 50 }}
                    dpr={[1, 2]}
                >
                    <color attach="background" args={['#0a0a12']} />
                    <fog attach="fog" args={['#0a0a12', 8, 20]} />

                    <ambientLight intensity={0.3} />
                    <pointLight position={[5, 5, 5]} intensity={0.8} color="#6366f1" />
                    <pointLight position={[-5, -5, 5]} intensity={0.4} color="#8b5cf6" />

                    <HolographicScene numTokens={numTokens} autoRotate={autoRotate} />

                    <OrbitControls
                        enableDamping
                        dampingFactor={0.05}
                        minDistance={4}
                        maxDistance={15}
                    />

                    {/* Stats overlay */}
                    <Html position={[4.5, 2.5, 0]} center>
                        <div className="holographic-stats-3d">
                            <div className="stat-row">
                                <span className="stat-label">Tokens</span>
                                <span className="stat-value">{numTokens}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">HD Dim</span>
                                <span className="stat-value">{hdDim.toLocaleString()}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">Ratio</span>
                                <span className="stat-value">{compressionRatio}×</span>
                            </div>
                        </div>
                    </Html>
                </Canvas>

                {/* Scanline overlay */}
                <div className="holographic-bundle__scanlines" />
            </div>

            <div className="holographic-bundle__footer">
                <div className="encoding-flow">
                    <span className="flow-step">Input</span>
                    <span className="flow-arrow">→</span>
                    <span className="flow-step">Random Index</span>
                    <span className="flow-arrow">→</span>
                    <span className="flow-step active">Bundle</span>
                    <span className="flow-arrow">→</span>
                    <span className="flow-step">CTQW</span>
                    <span className="flow-arrow">→</span>
                    <span className="flow-step">Compressed</span>
                </div>
            </div>
        </div>
    );
}

export default HolographicBundleViz;
