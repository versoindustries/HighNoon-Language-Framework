// FloquetPhaseViz.tsx - Discrete Time Crystal Phase-Space Visualization
// Animated Floquet Hamiltonian evolution with sub-harmonic response

import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Line, Trail } from '@react-three/drei';
import { useMemo, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import './FloquetPhaseViz.css';

/** Props for FloquetPhaseViz component */
interface FloquetPhaseVizProps {
    /** Floquet period (T) */
    period?: number;
    /** Number of oscillators */
    numOscillators?: number;
    /** Sub-harmonic order (period doubling) */
    subharmonicOrder?: number;
    /** Enable auto-rotation */
    autoRotate?: boolean;
    /** Container height */
    height?: number;
    /** Export callback */
    onExport?: (dataUrl: string) => void;
}

/** Single oscillator in the time crystal lattice */
function TimeOscillator({
    index,
    total,
    period,
    subharmonicOrder,
    color
}: {
    index: number;
    total: number;
    period: number;
    subharmonicOrder: number;
    color: string;
}) {
    const meshRef = useRef<THREE.Mesh>(null);
    const phase = (index / total) * Math.PI * 2;
    const baseRadius = 2.5;

    useFrame((state) => {
        if (!meshRef.current) return;
        const t = state.clock.elapsedTime;

        // Floquet driving frequency
        const omega = (2 * Math.PI) / period;

        // Sub-harmonic response (period doubling/tripling)
        const subOmega = omega / subharmonicOrder;

        // Phase-space trajectory with Floquet evolution
        const x = baseRadius * Math.cos(phase + subOmega * t);
        const y = baseRadius * Math.sin(phase + subOmega * t);

        // Z oscillation with driving frequency
        const z = Math.sin(omega * t + phase) * 0.8;

        meshRef.current.position.set(x, y, z);

        // Scale pulsing synchronized with driving
        const pulse = 0.15 + Math.abs(Math.sin(omega * t + phase)) * 0.1;
        meshRef.current.scale.setScalar(pulse);
    });

    return (
        <Trail
            width={0.2}
            length={12}
            color={new THREE.Color(color)}
            attenuation={(t) => t * t}
        >
            <mesh ref={meshRef}>
                <octahedronGeometry args={[0.15, 0]} />
                <meshStandardMaterial
                    color={color}
                    emissive={color}
                    emissiveIntensity={0.7}
                    transparent
                    opacity={0.9}
                />
            </mesh>
        </Trail>
    );
}

/** Central phase indicator showing current Floquet phase */
function PhaseIndicator({ period }: { period: number }) {
    const arrowRef = useRef<THREE.Group>(null);
    const [currentPhase, setCurrentPhase] = useState(0);

    useFrame((state) => {
        if (!arrowRef.current) return;
        const t = state.clock.elapsedTime;
        const omega = (2 * Math.PI) / period;
        const phase = (t * omega) % (2 * Math.PI);
        setCurrentPhase(phase);
        arrowRef.current.rotation.z = -phase;
    });

    return (
        <group ref={arrowRef}>
            {/* Arrow indicator */}
            <mesh position={[0.8, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
                <coneGeometry args={[0.1, 0.3, 8]} />
                <meshStandardMaterial
                    color="#f59e0b"
                    emissive="#f59e0b"
                    emissiveIntensity={0.6}
                />
            </mesh>
            <mesh position={[0.4, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
                <cylinderGeometry args={[0.03, 0.03, 0.8, 8]} />
                <meshStandardMaterial
                    color="#f59e0b"
                    emissive="#f59e0b"
                    emissiveIntensity={0.4}
                />
            </mesh>

            {/* Phase value display */}
            <Html position={[0, -0.6, 0]} center>
                <div className="phase-value">
                    φ = {(currentPhase / Math.PI).toFixed(2)}π
                </div>
            </Html>
        </group>
    );
}

/** Lattice structure connecting oscillators */
function CrystalLattice({ numOscillators }: { numOscillators: number }) {
    const points = useMemo(() => {
        const pts: THREE.Vector3[] = [];
        const radius = 2.5;

        for (let i = 0; i <= numOscillators; i++) {
            const angle = (i / numOscillators) * Math.PI * 2;
            pts.push(new THREE.Vector3(
                Math.cos(angle) * radius,
                Math.sin(angle) * radius,
                0
            ));
        }
        return pts;
    }, [numOscillators]);

    return (
        <Line
            points={points}
            color="#6366f1"
            lineWidth={1}
            transparent
            opacity={0.3}
            dashed
            dashSize={0.1}
            gapSize={0.05}
        />
    );
}

/** Main Floquet scene */
function FloquetScene({
    numOscillators,
    period,
    subharmonicOrder,
    autoRotate
}: {
    numOscillators: number;
    period: number;
    subharmonicOrder: number;
    autoRotate: boolean;
}) {
    const groupRef = useRef<THREE.Group>(null);

    useFrame((_, delta) => {
        if (autoRotate && groupRef.current) {
            groupRef.current.rotation.x += delta * 0.05;
            groupRef.current.rotation.y += delta * 0.08;
        }
    });

    const oscillatorColors = useMemo(() => {
        const palette = ['#6366f1', '#8b5cf6', '#ec4899', '#10b981', '#3b82f6', '#f59e0b'];
        return Array.from({ length: numOscillators }, (_, i) => palette[i % palette.length]);
    }, [numOscillators]);

    return (
        <group ref={groupRef}>
            {/* Central phase indicator */}
            <PhaseIndicator period={period} />

            {/* Crystal lattice structure */}
            <CrystalLattice numOscillators={numOscillators} />

            {/* Time oscillators */}
            {oscillatorColors.map((color, i) => (
                <TimeOscillator
                    key={i}
                    index={i}
                    total={numOscillators}
                    period={period}
                    subharmonicOrder={subharmonicOrder}
                    color={color}
                />
            ))}

            {/* Central glow */}
            <mesh>
                <sphereGeometry args={[0.3, 32, 32]} />
                <meshStandardMaterial
                    color="#6366f1"
                    emissive="#6366f1"
                    emissiveIntensity={0.5}
                    transparent
                    opacity={0.4}
                />
            </mesh>

            {/* Z-axis guide */}
            <group>
                <mesh position={[0, 0, 1]} rotation={[Math.PI / 2, 0, 0]}>
                    <cylinderGeometry args={[0.01, 0.01, 2, 8]} />
                    <meshStandardMaterial color="#444" transparent opacity={0.5} />
                </mesh>
            </group>
        </group>
    );
}

/**
 * FloquetPhaseViz - Discrete Time Crystal Visualization
 *
 * Displays Floquet Hamiltonian dynamics with:
 * - Periodic driving visualization
 * - Sub-harmonic response (period doubling)
 * - Phase-space trajectory orbits
 * - Central phase indicator
 */
export function FloquetPhaseViz({
    period = 2,
    numOscillators = 6,
    subharmonicOrder = 2,
    autoRotate = true,
    height = 350,
    onExport,
}: FloquetPhaseVizProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isExporting, setIsExporting] = useState(false);

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
        <div className="floquet-phase">
            <div className="floquet-phase__header">
                <div className="floquet-phase__title">
                    <span className="floquet-badge">FLOQUET</span>
                    Discrete Time Crystal
                </div>
                <div className="floquet-phase__controls">
                    <span className="period-indicator">
                        T = {period}s
                    </span>
                    <span className="subharmonic-indicator">
                        ×{subharmonicOrder} Period
                    </span>
                    {onExport && (
                        <button
                            onClick={handleExport}
                            disabled={isExporting}
                            className="floquet-export-btn"
                        >
                            📷 {isExporting ? '...' : 'Export'}
                        </button>
                    )}
                </div>
            </div>

            <div className="floquet-phase__canvas" style={{ height }}>
                <Canvas
                    ref={canvasRef}
                    gl={{ preserveDrawingBuffer: true, antialias: true }}
                    camera={{ position: [4, 3, 5], fov: 50 }}
                    dpr={[1, 2]}
                >
                    <color attach="background" args={['#08080e']} />
                    <fog attach="fog" args={['#08080e', 10, 25]} />

                    <ambientLight intensity={0.25} />
                    <pointLight position={[5, 5, 5]} intensity={0.7} color="#6366f1" />
                    <pointLight position={[-5, -5, 3]} intensity={0.3} color="#f59e0b" />

                    <FloquetScene
                        numOscillators={numOscillators}
                        period={period}
                        subharmonicOrder={subharmonicOrder}
                        autoRotate={autoRotate}
                    />

                    <OrbitControls
                        enableDamping
                        dampingFactor={0.05}
                        minDistance={4}
                        maxDistance={15}
                    />
                </Canvas>

                {/* Ambient glow overlay */}
                <div className="floquet-phase__glow" />
            </div>

            <div className="floquet-phase__footer">
                <div className="floquet-legend">
                    <div className="legend-item">
                        <span className="legend-dot driving" />
                        <span>Driving ω = 2π/T</span>
                    </div>
                    <div className="legend-item">
                        <span className="legend-dot response" />
                        <span>Response ω/{subharmonicOrder}</span>
                    </div>
                    <div className="legend-item">
                        <span className="legend-dot phase" />
                        <span>Phase Trajectory</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default FloquetPhaseViz;
