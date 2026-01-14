// NeuralArchitectureGraph3D.tsx - 3D force-directed graph of neural architecture
// Visualizes HSMN architecture blocks with interactive exploration

import React, { useRef, useMemo, useState, useCallback, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Line } from '@react-three/drei';
import * as THREE from 'three';

// Node type definitions
type NodeType = 'embedding' | 'qhd_spatial' | 'time_crystal' | 'latent_reasoning' | 'wlam' | 'moe' | 'output';

interface ArchitectureNode {
    id: string;
    type: NodeType;
    label: string;
    /** Optional parameter count */
    params?: number;
    /** Current activation magnitude (for training visualization) */
    activation?: number;
    /** Position override (for hierarchical layout) */
    position?: [number, number, number];
}

interface ArchitectureEdge {
    source: string;
    target: string;
    /** Edge type */
    type: 'primary' | 'skip' | 'attention';
    /** Edge weight (parameter count or attention weight) */
    weight?: number;
}

interface NeuralArchitectureGraph3DProps {
    /** Architecture nodes */
    nodes: ArchitectureNode[];
    /** Architecture edges */
    edges: ArchitectureEdge[];
    /** Layout mode */
    layout?: 'force' | 'hierarchical' | 'circular' | 'layered';
    /** Enable real-time activation visualization */
    showActivations?: boolean;
    /** Animation speed multiplier */
    animationSpeed?: number;
    /** Node click callback */
    onNodeClick?: (node: ArchitectureNode) => void;
    /** Node hover callback */
    onNodeHover?: (node: ArchitectureNode | null) => void;
    /** Container className */
    className?: string;
}

// Node type visual configurations
const NODE_CONFIG: Record<NodeType, {
    color: string;
    emissive: string;
    shape: 'cube' | 'octahedron' | 'sphere' | 'torus' | 'cone';
    size: number;
}> = {
    embedding: {
        color: '#ffffff',
        emissive: '#ffffff',
        shape: 'cube',
        size: 0.25,
    },
    qhd_spatial: {
        color: '#3b82f6',
        emissive: '#3b82f6',
        shape: 'cube',
        size: 0.4,
    },
    time_crystal: {
        color: '#f97316',
        emissive: '#f97316',
        shape: 'octahedron',
        size: 0.35,
    },
    latent_reasoning: {
        color: '#a855f7',
        emissive: '#a855f7',
        shape: 'sphere',
        size: 0.35,
    },
    wlam: {
        color: '#22c55e',
        emissive: '#22c55e',
        shape: 'torus',
        size: 0.3,
    },
    moe: {
        color: '#ef4444',
        emissive: '#ef4444',
        shape: 'sphere',
        size: 0.4,
    },
    output: {
        color: '#1f2937',
        emissive: '#374151',
        shape: 'cube',
        size: 0.25,
    },
};

// Edge type visual configurations
const EDGE_CONFIG: Record<string, { color: string; dashSize?: number; opacity: number }> = {
    primary: { color: '#6b7280', opacity: 0.6 },
    skip: { color: '#a855f7', dashSize: 0.1, opacity: 0.4 },
    attention: { color: '#00f5ff', opacity: 0.5 },
};

/**
 * Calculate hierarchical layout positions
 */
function calculateHierarchicalLayout(nodes: ArchitectureNode[], edges: ArchitectureEdge[]): Map<string, [number, number, number]> {
    const positions = new Map<string, [number, number, number]>();

    // Build adjacency list
    const outEdges = new Map<string, string[]>();
    const inEdges = new Map<string, string[]>();

    nodes.forEach(n => {
        outEdges.set(n.id, []);
        inEdges.set(n.id, []);
    });

    edges.forEach(e => {
        outEdges.get(e.source)?.push(e.target);
        inEdges.get(e.target)?.push(e.source);
    });

    // Find layers using topological sort
    const layers: string[][] = [];
    const visited = new Set<string>();
    const remaining = new Set(nodes.map(n => n.id));

    while (remaining.size > 0) {
        const layer: string[] = [];

        remaining.forEach(nodeId => {
            const deps = inEdges.get(nodeId) || [];
            if (deps.every(d => visited.has(d))) {
                layer.push(nodeId);
            }
        });

        if (layer.length === 0) {
            // Handle cycles - just add remaining
            layer.push(...remaining);
        }

        layer.forEach(id => {
            visited.add(id);
            remaining.delete(id);
        });

        layers.push(layer);
    }

    // Position nodes in layers
    const layerSpacing = 2.5;
    const nodeSpacing = 1.5;

    layers.forEach((layer, layerIndex) => {
        const x = layerIndex * layerSpacing - (layers.length - 1) * layerSpacing / 2;
        const yOffset = (layer.length - 1) * nodeSpacing / 2;

        layer.forEach((nodeId, nodeIndex) => {
            const y = nodeIndex * nodeSpacing - yOffset;
            const z = (Math.random() - 0.5) * 0.5; // Slight z variation
            positions.set(nodeId, [x, y, z]);
        });
    });

    return positions;
}

/**
 * Individual node component
 */
function ArchitectureNodeMesh({
    node,
    position,
    config,
    onHover,
    onClick,
}: {
    node: ArchitectureNode;
    position: [number, number, number];
    config: typeof NODE_CONFIG[NodeType];
    onHover: (node: ArchitectureNode | null) => void;
    onClick: (node: ArchitectureNode) => void;
}) {
    const meshRef = useRef<THREE.Mesh>(null);
    const [hovered, setHovered] = useState(false);

    // Animation based on activation level
    useFrame((state) => {
        if (meshRef.current) {
            // Pulse based on activation
            const activationScale = node.activation
                ? 1 + node.activation * 0.2 * Math.sin(state.clock.elapsedTime * 3)
                : 1;
            meshRef.current.scale.setScalar(activationScale);

            // Gentle rotation for non-cube shapes
            if (config.shape !== 'cube') {
                meshRef.current.rotation.y += 0.005;
            }
        }
    });

    const handlePointerOver = useCallback(() => {
        setHovered(true);
        onHover(node);
    }, [node, onHover]);

    const handlePointerOut = useCallback(() => {
        setHovered(false);
        onHover(null);
    }, [onHover]);

    const handleClick = useCallback(() => {
        onClick(node);
    }, [node, onClick]);

    // Render appropriate geometry based on shape
    const geometry = useMemo(() => {
        switch (config.shape) {
            case 'cube':
                return <boxGeometry args={[config.size, config.size, config.size]} />;
            case 'octahedron':
                return <octahedronGeometry args={[config.size]} />;
            case 'sphere':
                return <sphereGeometry args={[config.size, 24, 24]} />;
            case 'torus':
                return <torusGeometry args={[config.size, config.size * 0.3, 16, 32]} />;
            case 'cone':
                return <coneGeometry args={[config.size, config.size * 1.5, 8]} />;
            default:
                return <sphereGeometry args={[config.size, 16, 16]} />;
        }
    }, [config]);

    const emissiveIntensity = useMemo(() => {
        const base = hovered ? 0.8 : 0.3;
        return base + (node.activation || 0) * 0.5;
    }, [hovered, node.activation]);

    return (
        <group position={position}>
            <mesh
                ref={meshRef}
                onPointerOver={handlePointerOver}
                onPointerOut={handlePointerOut}
                onClick={handleClick}
            >
                {geometry}
                <meshStandardMaterial
                    color={config.color}
                    emissive={config.emissive}
                    emissiveIntensity={emissiveIntensity}
                    metalness={0.3}
                    roughness={0.4}
                />
            </mesh>

            {/* Glow effect on hover */}
            {hovered && (
                <mesh>
                    <sphereGeometry args={[config.size * 1.5, 16, 16]} />
                    <meshBasicMaterial
                        color={config.emissive}
                        transparent
                        opacity={0.15}
                    />
                </mesh>
            )}

            {/* Label */}
            <Html position={[0, config.size + 0.3, 0]} center>
                <div className={`arch-node-label ${hovered ? 'arch-node-label--hover' : ''}`}>
                    {node.label}
                </div>
            </Html>

            {/* Detailed tooltip on hover */}
            {hovered && (
                <Html position={[config.size + 0.5, 0, 0]} center>
                    <div className="arch-node-tooltip">
                        <div className="arch-node-tooltip__type">{node.type.replace('_', ' ').toUpperCase()}</div>
                        {node.params !== undefined && (
                            <div className="arch-node-tooltip__params">
                                {(node.params / 1e6).toFixed(2)}M params
                            </div>
                        )}
                        {node.activation !== undefined && (
                            <div className="arch-node-tooltip__activation">
                                Activation: {(node.activation * 100).toFixed(1)}%
                            </div>
                        )}
                    </div>
                </Html>
            )}
        </group>
    );
}

/**
 * Edge connection with optional animated particles
 */
function ArchitectureEdgeLine({
    start,
    end,
    config,
    showAnimation,
}: {
    start: [number, number, number];
    end: [number, number, number];
    config: typeof EDGE_CONFIG[string];
    showAnimation: boolean;
}) {
    const particleRef = useRef<THREE.Mesh>(null);
    const [particleT, setParticleT] = useState(Math.random());

    // Animate particles along edge
    useFrame((state, delta) => {
        if (showAnimation && particleRef.current) {
            setParticleT(prev => (prev + delta * 0.5) % 1);

            const x = start[0] + (end[0] - start[0]) * particleT;
            const y = start[1] + (end[1] - start[1]) * particleT;
            const z = start[2] + (end[2] - start[2]) * particleT;

            particleRef.current.position.set(x, y, z);
        }
    });

    return (
        <group>
            <Line
                points={[start, end]}
                color={config.color}
                lineWidth={2}
                transparent
                opacity={config.opacity}
                dashed={config.dashSize !== undefined}
                dashSize={config.dashSize}
                gapSize={config.dashSize}
            />

            {/* Animated data flow particle */}
            {showAnimation && (
                <mesh ref={particleRef}>
                    <sphereGeometry args={[0.05, 8, 8]} />
                    <meshBasicMaterial color={config.color} />
                </mesh>
            )}
        </group>
    );
}

/**
 * Scene content for the neural architecture graph
 */
function ArchitectureScene({
    nodes,
    edges,
    layout,
    showActivations,
    animationSpeed,
    onNodeClick,
    onNodeHover,
}: {
    nodes: ArchitectureNode[];
    edges: ArchitectureEdge[];
    layout: 'force' | 'hierarchical' | 'circular' | 'layered';
    showActivations: boolean;
    animationSpeed: number;
    onNodeClick: (node: ArchitectureNode) => void;
    onNodeHover: (node: ArchitectureNode | null) => void;
}) {
    // Calculate node positions based on layout
    const positions = useMemo(() => {
        if (layout === 'hierarchical') {
            return calculateHierarchicalLayout(nodes, edges);
        }

        // Default: use provided positions or random
        const pos = new Map<string, [number, number, number]>();
        nodes.forEach((node, i) => {
            if (node.position) {
                pos.set(node.id, node.position);
            } else {
                // Circular layout as fallback
                const angle = (i / nodes.length) * Math.PI * 2;
                const radius = 3;
                pos.set(node.id, [
                    Math.cos(angle) * radius,
                    Math.sin(angle) * radius,
                    (Math.random() - 0.5) * 0.5,
                ]);
            }
        });
        return pos;
    }, [nodes, edges, layout]);

    return (
        <group>
            {/* Lighting */}
            <ambientLight intensity={0.4} />
            <directionalLight position={[5, 5, 5]} intensity={0.6} />
            <pointLight position={[-5, 5, -5]} intensity={0.3} color="#a855f7" />
            <pointLight position={[5, -5, 5]} intensity={0.2} color="#3b82f6" />

            {/* Edges (render first for proper depth) */}
            {edges.map((edge, i) => {
                const startPos = positions.get(edge.source);
                const endPos = positions.get(edge.target);
                if (!startPos || !endPos) return null;

                const config = EDGE_CONFIG[edge.type] || EDGE_CONFIG.primary;

                return (
                    <ArchitectureEdgeLine
                        key={`edge-${i}`}
                        start={startPos}
                        end={endPos}
                        config={config}
                        showAnimation={showActivations && edge.type === 'attention'}
                    />
                );
            })}

            {/* Nodes */}
            {nodes.map(node => {
                const pos = positions.get(node.id);
                if (!pos) return null;

                const config = NODE_CONFIG[node.type] || NODE_CONFIG.embedding;

                return (
                    <ArchitectureNodeMesh
                        key={node.id}
                        node={node}
                        position={pos}
                        config={config}
                        onHover={onNodeHover}
                        onClick={onNodeClick}
                    />
                );
            })}

            {/* Orbit controls */}
            <OrbitControls
                enableDamping
                dampingFactor={0.05}
                minDistance={3}
                maxDistance={20}
                autoRotate
                autoRotateSpeed={0.5 * animationSpeed}
            />
        </group>
    );
}

/**
 * NeuralArchitectureGraph3D - 3D force-directed graph visualization
 *
 * Features:
 * - Distinct shapes/colors for each block type
 * - Force-directed or hierarchical layout
 * - Animated data flow along edges
 * - Activation-based node pulsing during training
 * - Interactive hover/click for block details
 * - Auto-rotation for visual interest
 */
export function NeuralArchitectureGraph3D({
    nodes,
    edges,
    layout = 'hierarchical',
    showActivations = false,
    animationSpeed = 1,
    onNodeClick,
    onNodeHover,
    className = '',
}: NeuralArchitectureGraph3DProps) {
    const handleNodeClick = useCallback((node: ArchitectureNode) => {
        onNodeClick?.(node);
    }, [onNodeClick]);

    const handleNodeHover = useCallback((node: ArchitectureNode | null) => {
        onNodeHover?.(node);
    }, [onNodeHover]);

    return (
        <div className={`neural-arch-graph-3d ${className}`}>
            <Canvas
                camera={{ position: [0, 0, 8], fov: 50 }}
                gl={{ antialias: true, alpha: true }}
            >
                <ArchitectureScene
                    nodes={nodes}
                    edges={edges}
                    layout={layout}
                    showActivations={showActivations}
                    animationSpeed={animationSpeed}
                    onNodeClick={handleNodeClick}
                    onNodeHover={handleNodeHover}
                />
            </Canvas>

            {/* Legend */}
            <div className="neural-arch-legend">
                <div className="neural-arch-legend__title">Block Types</div>
                <div className="neural-arch-legend__items">
                    {Object.entries(NODE_CONFIG).map(([type, config]) => (
                        <div key={type} className="neural-arch-legend__item">
                            <span
                                className="neural-arch-legend__shape"
                                style={{ background: config.color }}
                            />
                            <span>{type.replace('_', ' ')}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

export default NeuralArchitectureGraph3D;
