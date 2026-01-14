// QuantumPopulationViz.tsx - QAHPO Quantum Population Visualization
// Shows quantum-inspired population state with amplitudes and tunneling

import { useState, useEffect, useMemo } from 'react';
import { Atom, Zap, Thermometer, TrendingUp, Info } from 'lucide-react';
import './QuantumPopulationViz.css';

interface PopulationMember {
    config_hash: string;
    amplitude: number;
    loss: number;
    generation: number;
    trial_id?: string | number;
}

interface QuantumPopulationVizProps {
    jobId: string | null;
    isRunning: boolean;
    temperature?: number;
    tunnelingProbability?: number;
    generation?: number;
    explorationMode?: string;
}

// Generate simulated population data for demo (in production, comes from WebSocket)
function generateDemoPopulation(size: number = 8): PopulationMember[] {
    return Array.from({ length: size }, (_, i) => ({
        config_hash: `config_${i}`,
        amplitude: Math.random() * 0.8 + 0.2,
        loss: Math.random() * 0.5 + 0.1,
        generation: Math.floor(Math.random() * 5),
        trial_id: i,
    }));
}

// Color based on loss (lower is better = more green)
function lossToColor(loss: number, minLoss: number, maxLoss: number): string {
    const range = maxLoss - minLoss || 1;
    const normalized = 1 - (loss - minLoss) / range;
    const hue = normalized * 120; // 0 = red, 120 = green
    return `hsl(${hue}, 70%, 55%)`;
}

export function QuantumPopulationViz({
    jobId,
    isRunning,
    temperature = 1.0,
    tunnelingProbability = 0.1,
    generation = 0,
    explorationMode = 'exploit',
}: QuantumPopulationVizProps) {
    const [population, setPopulation] = useState<PopulationMember[]>([]);
    const [tunnelFlash, setTunnelFlash] = useState(false);

    // Demo mode: generate sample population
    useEffect(() => {
        if (isRunning && population.length === 0) {
            setPopulation(generateDemoPopulation(8));
        }
    }, [isRunning, population.length]);

    // Simulate tunneling flash animation
    useEffect(() => {
        if (tunnelingProbability > 0.5) {
            setTunnelFlash(true);
            const timer = setTimeout(() => setTunnelFlash(false), 500);
            return () => clearTimeout(timer);
        }
    }, [tunnelingProbability, generation]);

    const { sortedPopulation, lossRange } = useMemo(() => {
        if (population.length === 0) {
            return { sortedPopulation: [], lossRange: { min: 0, max: 1 } };
        }

        const sorted = [...population].sort((a, b) => b.amplitude - a.amplitude);
        const losses = population.map(p => p.loss);
        return {
            sortedPopulation: sorted,
            lossRange: {
                min: Math.min(...losses),
                max: Math.max(...losses),
            },
        };
    }, [population]);

    // Temperature gradient background
    const tempGradient = useMemo(() => {
        // High temp = more red/orange, low temp = more blue/purple
        const warmth = Math.min(temperature, 1.0);
        if (warmth > 0.7) {
            return 'linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(249, 115, 22, 0.1))';
        } else if (warmth > 0.4) {
            return 'linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(16, 185, 129, 0.08))';
        } else {
            return 'linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(99, 102, 241, 0.1))';
        }
    }, [temperature]);

    if (!jobId) {
        return (
            <div className="quantum-pop quantum-pop--empty">
                <Atom size={24} />
                <p>Start a sweep to see population state</p>
            </div>
        );
    }

    return (
        <div
            className={`quantum-pop ${tunnelFlash ? 'quantum-pop--tunnel-flash' : ''}`}
            style={{ background: tempGradient }}
        >
            <div className="quantum-pop__header">
                <h4 className="quantum-pop__title">
                    <Atom size={16} />
                    Quantum Population
                </h4>
                <div className="quantum-pop__stats">
                    <span className="quantum-pop__stat">
                        <Thermometer size={12} />
                        T={temperature.toFixed(2)}
                    </span>
                    <span className="quantum-pop__stat">
                        <Zap size={12} />
                        P<sub>tunnel</sub>={tunnelingProbability.toFixed(2)}
                    </span>
                    <span className={`quantum-pop__mode quantum-pop__mode--${explorationMode}`}>
                        {explorationMode}
                    </span>
                </div>
            </div>

            {sortedPopulation.length === 0 ? (
                <div className="quantum-pop__waiting">
                    <Info size={16} />
                    <span>Waiting for population data...</span>
                </div>
            ) : (
                <div className="quantum-pop__grid">
                    {sortedPopulation.map((member, i) => {
                        const size = 30 + member.amplitude * 40; // 30-70px
                        const color = lossToColor(member.loss, lossRange.min, lossRange.max);

                        return (
                            <div
                                key={member.config_hash}
                                className="quantum-pop__node"
                                style={{
                                    width: size,
                                    height: size,
                                    backgroundColor: color,
                                    boxShadow: `0 0 ${member.amplitude * 20}px ${color}`,
                                    opacity: 0.4 + member.amplitude * 0.6,
                                }}
                                title={`Config ${i + 1}\nAmplitude: ${member.amplitude.toFixed(3)}\nLoss: ${member.loss.toFixed(4)}\nGen: ${member.generation}`}
                            >
                                <span className="quantum-pop__node-rank">
                                    {i + 1}
                                </span>
                            </div>
                        );
                    })}
                </div>
            )}

            <div className="quantum-pop__legend">
                <div className="quantum-pop__legend-item">
                    <span>Size = Amplitude</span>
                </div>
                <div className="quantum-pop__legend-item">
                    <div className="quantum-pop__legend-gradient" />
                    <span>Loss</span>
                </div>
                <div className="quantum-pop__legend-item">
                    <span>Gen: {generation}</span>
                </div>
            </div>
        </div>
    );
}

export default QuantumPopulationViz;
