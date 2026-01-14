// FidelityLadder.tsx - BOHB Multi-Fidelity Bracket Visualization
// Shows trial progression through fidelity levels with promotion arrows

import { useState, useMemo } from 'react';
import { Layers, ArrowRight, Check, X, Loader } from 'lucide-react';
import type { HPOTrialInfo } from '../api/types';
import './FidelityLadder.css';

interface FidelityLadderProps {
    trials: HPOTrialInfo[];
    fidelityLevels?: number;
    height?: number;
}

// Infer fidelity level from trial step count or hyperparams
function inferFidelity(trial: HPOTrialInfo, maxSteps: number): number {
    const step = trial.step || 0;
    const ratio = step / maxSteps;
    if (ratio > 0.8) return 3;
    if (ratio > 0.4) return 2;
    return 1;
}

export function FidelityLadder({
    trials,
    fidelityLevels = 3,
    height = 200,
}: FidelityLadderProps) {
    const [hoveredTrialId, setHoveredTrialId] = useState<string | number | null>(null);

    const { trialsByFidelity, maxTrialsPerLevel, maxSteps } = useMemo(() => {
        if (trials.length === 0) {
            return { trialsByFidelity: {}, maxTrialsPerLevel: 0, maxSteps: 1000 };
        }

        const maxSteps = Math.max(...trials.map(t => t.step || 0), 1000);
        const byFidelity: Record<number, HPOTrialInfo[]> = {};

        for (let i = 1; i <= fidelityLevels; i++) {
            byFidelity[i] = [];
        }

        trials.forEach(trial => {
            const fidelity = inferFidelity(trial, maxSteps);
            if (fidelity >= 1 && fidelity <= fidelityLevels) {
                byFidelity[fidelity].push(trial);
            }
        });

        const maxTrialsPerLevel = Math.max(
            ...Object.values(byFidelity).map(t => t.length),
            1
        );

        return { trialsByFidelity: byFidelity, maxTrialsPerLevel, maxSteps };
    }, [trials, fidelityLevels]);

    const getStatusColor = (status: HPOTrialInfo['status']) => {
        switch (status) {
            case 'completed': return '#10b981';
            case 'running': return '#6366f1';
            case 'pruned': return '#f59e0b';
            case 'failed': return '#ef4444';
            default: return '#64748b';
        }
    };

    const getStatusIcon = (status: HPOTrialInfo['status']) => {
        switch (status) {
            case 'completed': return <Check size={10} />;
            case 'running': return <Loader size={10} className="spinning" />;
            case 'pruned': return <X size={10} />;
            case 'failed': return <X size={10} />;
            default: return null;
        }
    };

    if (trials.length === 0) {
        return (
            <div className="fidelity-ladder fidelity-ladder--empty">
                <Layers size={24} />
                <p>No trials to display</p>
            </div>
        );
    }

    return (
        <div className="fidelity-ladder" style={{ minHeight: height }}>
            <div className="fidelity-ladder__header">
                <h4 className="fidelity-ladder__title">
                    <Layers size={16} />
                    BOHB Fidelity Ladder
                </h4>
                <div className="fidelity-ladder__legend">
                    <span className="fidelity-ladder__legend-item">
                        <span className="fidelity-ladder__legend-dot" style={{ background: '#10b981' }} />
                        Completed
                    </span>
                    <span className="fidelity-ladder__legend-item">
                        <span className="fidelity-ladder__legend-dot" style={{ background: '#6366f1' }} />
                        Running
                    </span>
                    <span className="fidelity-ladder__legend-item">
                        <span className="fidelity-ladder__legend-dot" style={{ background: '#f59e0b' }} />
                        Pruned
                    </span>
                </div>
            </div>

            <div className="fidelity-ladder__levels">
                {Array.from({ length: fidelityLevels }, (_, i) => {
                    const level = fidelityLevels - i; // Top level first
                    const trialsAtLevel = trialsByFidelity[level] || [];
                    const budget = Math.pow(3, level - 1); // Typical η=3 scaling

                    return (
                        <div key={level} className="fidelity-ladder__level">
                            <div className="fidelity-ladder__level-header">
                                <span className="fidelity-ladder__level-label">
                                    Level {level}
                                </span>
                                <span className="fidelity-ladder__level-budget">
                                    {budget}× budget
                                </span>
                            </div>

                            <div className="fidelity-ladder__level-bar">
                                <div className="fidelity-ladder__trials">
                                    {trialsAtLevel.map((trial, idx) => (
                                        <div
                                            key={String(trial.trial_id)}
                                            className={`fidelity-ladder__trial ${trial.trial_id === hoveredTrialId ? 'fidelity-ladder__trial--hovered' : ''
                                                }`}
                                            style={{
                                                backgroundColor: getStatusColor(trial.status),
                                            }}
                                            onMouseEnter={() => setHoveredTrialId(trial.trial_id)}
                                            onMouseLeave={() => setHoveredTrialId(null)}
                                            title={`Trial ${trial.trial_id}\nStatus: ${trial.status}\nLoss: ${trial.loss?.toFixed(4) || '—'}`}
                                        >
                                            {getStatusIcon(trial.status)}
                                        </div>
                                    ))}

                                    {/* Fill remaining space */}
                                    {trialsAtLevel.length === 0 && (
                                        <span className="fidelity-ladder__empty-level">
                                            No trials at this level
                                        </span>
                                    )}
                                </div>

                                {/* Promotion arrows */}
                                {level < fidelityLevels && trialsAtLevel.length > 0 && (
                                    <div className="fidelity-ladder__promotion">
                                        <ArrowRight size={14} />
                                    </div>
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Summary */}
            <div className="fidelity-ladder__summary">
                {Array.from({ length: fidelityLevels }, (_, i) => {
                    const level = i + 1;
                    const count = (trialsByFidelity[level] || []).length;
                    return (
                        <span key={level} className="fidelity-ladder__summary-item">
                            L{level}: {count}
                        </span>
                    );
                })}
            </div>
        </div>
    );
}

export default FidelityLadder;
