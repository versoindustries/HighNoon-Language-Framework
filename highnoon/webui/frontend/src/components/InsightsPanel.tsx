// InsightsPanel.tsx - Automated HPO insights and recommendations
// Generates actionable text summaries from trial data

import { useMemo } from 'react';
import { Lightbulb, AlertTriangle, TrendingUp, Award, MemoryStick, Zap } from 'lucide-react';
import type { HPOTrialInfo, HPOSweepInfo } from '../api/types';
import './InsightsPanel.css';

interface InsightsPanelProps {
    trials: HPOTrialInfo[];
    sweepStatus: HPOSweepInfo | null;
    importance?: Record<string, number>;
}

interface Insight {
    id: string;
    type: 'success' | 'warning' | 'info' | 'tip';
    icon: React.ReactNode;
    title: string;
    message: string;
}

export function InsightsPanel({
    trials,
    sweepStatus,
    importance,
}: InsightsPanelProps) {
    const insights = useMemo<Insight[]>(() => {
        const result: Insight[] = [];

        const completedTrials = trials.filter(t => t.status === 'completed');
        const failedTrials = trials.filter(t => t.status === 'failed');
        const prunedTrials = trials.filter(t => t.status === 'pruned');

        if (completedTrials.length === 0) {
            return [{
                id: 'no-trials',
                type: 'info',
                icon: <Lightbulb size={16} />,
                title: 'Waiting for Trials',
                message: 'Insights will appear as trials complete.',
            }];
        }

        // Best trial insight
        const bestTrial = completedTrials.reduce((best, t) => {
            const score = t.composite_score ?? (t.loss !== null ? 1 - t.loss : 0);
            const bestScore = best.composite_score ?? (best.loss !== null ? 1 - best.loss : 0);
            return score > bestScore ? t : best;
        }, completedTrials[0]);

        if (bestTrial) {
            const lr = bestTrial.hyperparams?.learning_rate as number | undefined;
            result.push({
                id: 'best-trial',
                type: 'success',
                icon: <Award size={16} />,
                title: 'Best Configuration Found',
                message: lr
                    ? `Trial ${bestTrial.trial_id} achieved loss ${bestTrial.loss?.toFixed(4)} with LR=${lr.toExponential(2)}`
                    : `Trial ${bestTrial.trial_id} achieved lowest loss: ${bestTrial.loss?.toFixed(4)}`,
            });
        }

        // Importance insights
        if (importance && Object.keys(importance).length > 0) {
            const sorted = Object.entries(importance).sort((a, b) => b[1] - a[1]);
            const topParam = sorted[0];
            if (topParam && topParam[1] > 0.3) {
                result.push({
                    id: 'top-importance',
                    type: 'tip',
                    icon: <TrendingUp size={16} />,
                    title: 'Key Hyperparameter',
                    message: `${topParam[0].replace(/_/g, ' ')} has ${(topParam[1] * 100).toFixed(0)}% importance—focus tuning here.`,
                });
            }
        }

        // OOM/failure analysis
        if (failedTrials.length > 0) {
            const oomCount = failedTrials.filter(t =>
                (t.peak_memory_mb || 0) > 50000 // Likely OOM if peak >50GB
            ).length;

            if (oomCount > 0) {
                result.push({
                    id: 'oom-warning',
                    type: 'warning',
                    icon: <MemoryStick size={16} />,
                    title: 'Memory Pressure Detected',
                    message: `${oomCount} trial(s) likely failed due to OOM. Consider reducing param budget or embedding dim.`,
                });
            } else if (failedTrials.length >= 3) {
                result.push({
                    id: 'failures',
                    type: 'warning',
                    icon: <AlertTriangle size={16} />,
                    title: 'Multiple Failures',
                    message: `${failedTrials.length} trials failed. Check logs for common error patterns.`,
                });
            }
        }

        // Pruning efficiency
        if (prunedTrials.length > 0 && completedTrials.length > 0) {
            const pruneRate = prunedTrials.length / (prunedTrials.length + completedTrials.length);
            if (pruneRate > 0.6) {
                result.push({
                    id: 'high-prune',
                    type: 'info',
                    icon: <Zap size={16} />,
                    title: 'Aggressive Pruning',
                    message: `${(pruneRate * 100).toFixed(0)}% pruned—early stopping is working well.`,
                });
            }
        }

        // Convergence analysis
        if (completedTrials.length >= 5) {
            const losses = completedTrials
                .map(t => t.loss)
                .filter((l): l is number => l !== null)
                .sort((a, b) => a - b);

            if (losses.length >= 5) {
                const topN = losses.slice(0, 5);
                const range = topN[topN.length - 1] - topN[0];
                if (range < 0.01) {
                    result.push({
                        id: 'convergence',
                        type: 'info',
                        icon: <TrendingUp size={16} />,
                        title: 'Converging',
                        message: `Top 5 trials within ${(range * 100).toFixed(2)}% of each other—consider stopping.`,
                    });
                }
            }
        }

        // Pareto insight
        const paretoTrials = completedTrials.filter(t =>
            (t as unknown as { is_pareto_optimal?: boolean }).is_pareto_optimal
        );
        if (paretoTrials.length > 1) {
            result.push({
                id: 'pareto',
                type: 'tip',
                icon: <Award size={16} />,
                title: 'Pareto Frontier',
                message: `${paretoTrials.length} trials on Pareto frontier for loss-memory tradeoff.`,
            });
        }

        return result;
    }, [trials, sweepStatus, importance]);

    if (insights.length === 0) {
        return null;
    }

    return (
        <div className="insights-panel">
            <div className="insights-panel__header">
                <h4 className="insights-panel__title">
                    <Lightbulb size={16} />
                    Insights
                </h4>
            </div>
            <div className="insights-panel__list">
                {insights.map(insight => (
                    <div
                        key={insight.id}
                        className={`insights-panel__item insights-panel__item--${insight.type}`}
                    >
                        <div className="insights-panel__icon">
                            {insight.icon}
                        </div>
                        <div className="insights-panel__content">
                            <span className="insights-panel__item-title">{insight.title}</span>
                            <span className="insights-panel__item-message">{insight.message}</span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default InsightsPanel;
