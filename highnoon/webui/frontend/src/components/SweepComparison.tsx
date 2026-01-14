// HighNoon Dashboard - Sweep Comparison Component
// Side-by-side comparison of HPO sweeps for performance analysis

import { useState, useEffect, useMemo } from 'react';
import {
    GitCompare,
    Trophy,
    Clock,
    Target,
    Layers,
    ChevronDown,
    X,
    Download,
    TrendingUp,
    TrendingDown,
} from 'lucide-react';
import { hpoApi } from '../api/client';
import type { HPOSweepInfo } from '../api/types';
import { Sparkline } from './charts/Sparkline';
import './SweepComparison.css';

// =============================================================================
// TYPES
// =============================================================================

interface SweepComparisonProps {
    /** Called when user closes the comparison view */
    onClose: () => void;
    /** Initial sweep IDs to compare (optional) */
    initialSweepIds?: [string, string];
}

interface ComparisonMetric {
    label: string;
    key: keyof HPOSweepInfo | string;
    format: (value: unknown, sweep: HPOSweepInfo) => string;
    compare: (a: unknown, b: unknown) => 'better' | 'worse' | 'equal';
    lowerIsBetter?: boolean;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function formatDuration(isoString: string | null, endString?: string | null): string {
    if (!isoString) return '—';
    const start = new Date(isoString).getTime();
    const end = endString ? new Date(endString).getTime() : Date.now();
    const seconds = Math.floor((end - start) / 1000);

    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function formatNumber(num: number | null | undefined, decimals = 4): string {
    if (num === null || num === undefined) return '—';
    if (Math.abs(num) >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
    if (Math.abs(num) >= 1_000) return `${(num / 1_000).toFixed(1)}K`;
    return num.toFixed(decimals);
}

function formatParams(num: number | null | undefined): string {
    if (num === null || num === undefined) return '—';
    if (num >= 1_000_000_000) return `${(num / 1_000_000_000).toFixed(1)}B`;
    if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(0)}M`;
    return formatNumber(num, 0);
}

// =============================================================================
// COMPARISON METRICS
// =============================================================================

const COMPARISON_METRICS: ComparisonMetric[] = [
    {
        label: 'Best Loss',
        key: 'best_loss',
        format: (v) => formatNumber(v as number, 6),
        compare: (a, b) => {
            if (a === null || a === undefined) return 'worse';
            if (b === null || b === undefined) return 'better';
            return (a as number) < (b as number) ? 'better' : (a as number) > (b as number) ? 'worse' : 'equal';
        },
        lowerIsBetter: true,
    },
    {
        label: 'Best Composite',
        key: 'best_composite_score',
        format: (v) => formatNumber(v as number, 4),
        compare: (a, b) => {
            if (a === null || a === undefined) return 'worse';
            if (b === null || b === undefined) return 'better';
            return (a as number) > (b as number) ? 'better' : (a as number) < (b as number) ? 'worse' : 'equal';
        },
    },
    {
        label: 'Perplexity',
        key: 'best_perplexity',
        format: (v) => formatNumber(v as number, 2),
        compare: (a, b) => {
            if (a === null || a === undefined) return 'worse';
            if (b === null || b === undefined) return 'better';
            return (a as number) < (b as number) ? 'better' : (a as number) > (b as number) ? 'worse' : 'equal';
        },
        lowerIsBetter: true,
    },
    {
        label: 'Confidence',
        key: 'best_confidence',
        format: (v) => v != null ? `${((v as number) * 100).toFixed(1)}%` : '—',
        compare: (a, b) => {
            if (a === null || a === undefined) return 'worse';
            if (b === null || b === undefined) return 'better';
            return (a as number) > (b as number) ? 'better' : (a as number) < (b as number) ? 'worse' : 'equal';
        },
    },
    {
        label: 'Trials Completed',
        key: 'completed_trials',
        format: (v, sweep) => `${v}/${sweep.max_trials}`,
        compare: () => 'equal', // Not really comparable
    },
    {
        label: 'Pruned Trials',
        key: 'pruned_trials',
        format: (v) => `${v ?? 0}`,
        compare: () => 'equal',
    },
    {
        label: 'Peak Memory',
        key: 'best_memory_mb',
        format: (v) => v != null ? `${formatNumber(v as number, 0)} MB` : '—',
        compare: (a, b) => {
            if (a === null) return 'better'; // Less memory is better
            if (b === null) return 'worse';
            return (a as number) < (b as number) ? 'better' : (a as number) > (b as number) ? 'worse' : 'equal';
        },
        lowerIsBetter: true,
    },
];

// =============================================================================
// SWEEP SELECTOR
// =============================================================================

function SweepSelector({
    sweeps,
    selectedId,
    onSelect,
    excludeId,
    label,
}: {
    sweeps: HPOSweepInfo[];
    selectedId: string | null;
    onSelect: (id: string) => void;
    excludeId?: string;
    label: string;
}) {
    const availableSweeps = sweeps.filter(s => s.sweep_id !== excludeId);

    return (
        <div className="sweep-selector">
            <label className="sweep-selector__label">{label}</label>
            <div className="sweep-selector__dropdown">
                <select
                    value={selectedId || ''}
                    onChange={(e) => onSelect(e.target.value)}
                    className="sweep-selector__select"
                >
                    <option value="">Select a sweep...</option>
                    {availableSweeps.map(sweep => (
                        <option key={sweep.sweep_id} value={sweep.sweep_id}>
                            {sweep.sweep_id.slice(0, 8)} — {sweep.state}
                            {sweep.best_loss != null && ` (Loss: ${formatNumber(sweep.best_loss, 4)})`}
                        </option>
                    ))}
                </select>
                <ChevronDown size={16} className="sweep-selector__icon" />
            </div>
        </div>
    );
}

// =============================================================================
// SWEEP CARD
// =============================================================================

function SweepCard({ sweep, position }: { sweep: HPOSweepInfo | null; position: 'left' | 'right' }) {
    if (!sweep) {
        return (
            <div className="sweep-card sweep-card--empty">
                <div className="sweep-card__placeholder">
                    <GitCompare size={32} />
                    <span>Select a sweep to compare</span>
                </div>
            </div>
        );
    }

    return (
        <div className={`sweep-card sweep-card--${position}`}>
            <div className="sweep-card__header">
                <div className="sweep-card__id">{sweep.sweep_id.slice(0, 12)}...</div>
                <span className={`sweep-card__status sweep-card__status--${sweep.state}`}>
                    {sweep.state}
                </span>
            </div>

            <div className="sweep-card__stats">
                <div className="sweep-card__stat">
                    <Target size={14} />
                    <span>{sweep.completed_trials} trials</span>
                </div>
                <div className="sweep-card__stat">
                    <Clock size={14} />
                    <span>{formatDuration(sweep.started_at)}</span>
                </div>
                {sweep.model_config?.param_budget && (
                    <div className="sweep-card__stat">
                        <Layers size={14} />
                        <span>{formatParams(sweep.model_config.param_budget)}</span>
                    </div>
                )}
            </div>

            {sweep.config?.curriculum_id && (
                <div className="sweep-card__config">
                    <span className="sweep-card__config-label">Curriculum:</span>
                    <span className="sweep-card__config-value">{sweep.config.curriculum_id}</span>
                </div>
            )}
        </div>
    );
}

// =============================================================================
// COMPARISON TABLE
// =============================================================================

function ComparisonTable({
    sweepA,
    sweepB,
    metrics,
}: {
    sweepA: HPOSweepInfo | null;
    sweepB: HPOSweepInfo | null;
    metrics: ComparisonMetric[];
}) {
    if (!sweepA || !sweepB) {
        return null;
    }

    return (
        <div className="comparison-table">
            <div className="comparison-table__header">
                <div className="comparison-table__cell comparison-table__cell--metric">Metric</div>
                <div className="comparison-table__cell comparison-table__cell--value">Sweep A</div>
                <div className="comparison-table__cell comparison-table__cell--value">Sweep B</div>
                <div className="comparison-table__cell comparison-table__cell--delta">Δ</div>
            </div>

            {metrics.map(metric => {
                const valueA = (sweepA as any)[metric.key];
                const valueB = (sweepB as any)[metric.key];
                const comparisonA = metric.compare(valueA, valueB);
                const comparisonB = metric.compare(valueB, valueA);

                // Calculate delta
                let delta = '—';
                let deltaClass = '';
                if (typeof valueA === 'number' && typeof valueB === 'number') {
                    const diff = valueA - valueB;
                    const pctDiff = valueB !== 0 ? (diff / Math.abs(valueB)) * 100 : 0;
                    delta = `${diff >= 0 ? '+' : ''}${pctDiff.toFixed(1)}%`;
                    deltaClass = metric.lowerIsBetter
                        ? diff < 0 ? 'positive' : diff > 0 ? 'negative' : ''
                        : diff > 0 ? 'positive' : diff < 0 ? 'negative' : '';
                }

                return (
                    <div key={metric.key} className="comparison-table__row">
                        <div className="comparison-table__cell comparison-table__cell--metric">
                            {metric.label}
                        </div>
                        <div className={`comparison-table__cell comparison-table__cell--value comparison-table__cell--${comparisonA}`}>
                            {metric.format(valueA, sweepA)}
                            {comparisonA === 'better' && <Trophy size={12} className="comparison-trophy" />}
                        </div>
                        <div className={`comparison-table__cell comparison-table__cell--value comparison-table__cell--${comparisonB}`}>
                            {metric.format(valueB, sweepB)}
                            {comparisonB === 'better' && <Trophy size={12} className="comparison-trophy" />}
                        </div>
                        <div className={`comparison-table__cell comparison-table__cell--delta comparison-table__cell--${deltaClass}`}>
                            {deltaClass === 'positive' && <TrendingUp size={12} />}
                            {deltaClass === 'negative' && <TrendingDown size={12} />}
                            {delta}
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function SweepComparison({ onClose, initialSweepIds }: SweepComparisonProps) {
    const [sweeps, setSweeps] = useState<HPOSweepInfo[]>([]);
    const [loading, setLoading] = useState(true);
    const [sweepIdA, setSweepIdA] = useState<string | null>(initialSweepIds?.[0] || null);
    const [sweepIdB, setSweepIdB] = useState<string | null>(initialSweepIds?.[1] || null);

    // Fetch all sweeps
    useEffect(() => {
        async function fetchSweeps() {
            try {
                const data = await hpoApi.listSweeps();
                setSweeps(data);

                // Auto-select most recent completed sweeps if not pre-selected
                if (!sweepIdA && !sweepIdB && data.length >= 2) {
                    const completed = data.filter(s => s.state === 'completed');
                    if (completed.length >= 2) {
                        setSweepIdA(completed[0].sweep_id);
                        setSweepIdB(completed[1].sweep_id);
                    } else if (data.length >= 2) {
                        setSweepIdA(data[0].sweep_id);
                        setSweepIdB(data[1].sweep_id);
                    }
                }
            } catch (err) {
                console.error('Failed to fetch sweeps:', err);
            } finally {
                setLoading(false);
            }
        }
        fetchSweeps();
    }, []);

    const sweepA = useMemo(() => sweeps.find(s => s.sweep_id === sweepIdA) || null, [sweeps, sweepIdA]);
    const sweepB = useMemo(() => sweeps.find(s => s.sweep_id === sweepIdB) || null, [sweeps, sweepIdB]);

    // Extract trial losses for sparklines
    const lossHistoryA = useMemo(() => {
        if (!sweepA?.trials) return [];
        return sweepA.trials
            .filter(t => t.loss != null)
            .sort((a, b) => {
                const idA = typeof a.trial_id === 'number' ? a.trial_id : parseInt(String(a.trial_id).replace(/\D/g, '') || '0', 10);
                const idB = typeof b.trial_id === 'number' ? b.trial_id : parseInt(String(b.trial_id).replace(/\D/g, '') || '0', 10);
                return idA - idB;
            })
            .map(t => t.loss as number);
    }, [sweepA]);

    const lossHistoryB = useMemo(() => {
        if (!sweepB?.trials) return [];
        return sweepB.trials
            .filter(t => t.loss != null)
            .sort((a, b) => {
                const idA = typeof a.trial_id === 'number' ? a.trial_id : parseInt(String(a.trial_id).replace(/\D/g, '') || '0', 10);
                const idB = typeof b.trial_id === 'number' ? b.trial_id : parseInt(String(b.trial_id).replace(/\D/g, '') || '0', 10);
                return idA - idB;
            })
            .map(t => t.loss as number);
    }, [sweepB]);

    if (loading) {
        return (
            <div className="sweep-comparison sweep-comparison--loading">
                <div className="skeleton skeleton-header" />
                <div className="sweep-comparison__grid">
                    <div className="skeleton skeleton-card" />
                    <div className="skeleton skeleton-card" />
                </div>
            </div>
        );
    }

    if (sweeps.length < 2) {
        return (
            <div className="sweep-comparison sweep-comparison--empty">
                <GitCompare size={48} />
                <h3>Not Enough Sweeps</h3>
                <p>Run at least two HPO sweeps to enable comparison mode.</p>
                <button className="btn btn-secondary" onClick={onClose}>Close</button>
            </div>
        );
    }

    return (
        <div className="sweep-comparison">
            {/* Header */}
            <div className="sweep-comparison__header">
                <div className="sweep-comparison__title">
                    <GitCompare size={20} />
                    <h2>Sweep Comparison</h2>
                </div>
                <button className="sweep-comparison__close" onClick={onClose} aria-label="Close">
                    <X size={20} />
                </button>
            </div>

            {/* Selectors */}
            <div className="sweep-comparison__selectors">
                <SweepSelector
                    sweeps={sweeps}
                    selectedId={sweepIdA}
                    onSelect={setSweepIdA}
                    excludeId={sweepIdB || undefined}
                    label="Sweep A"
                />
                <div className="sweep-comparison__vs">VS</div>
                <SweepSelector
                    sweeps={sweeps}
                    selectedId={sweepIdB}
                    onSelect={setSweepIdB}
                    excludeId={sweepIdA || undefined}
                    label="Sweep B"
                />
            </div>

            {/* Sweep Cards */}
            <div className="sweep-comparison__grid">
                <SweepCard sweep={sweepA} position="left" />
                <SweepCard sweep={sweepB} position="right" />
            </div>

            {/* Sparkline Comparison */}
            {sweepA && sweepB && (lossHistoryA.length > 1 || lossHistoryB.length > 1) && (
                <div className="sweep-comparison__sparklines">
                    <h4>Loss Progression</h4>
                    <div className="sweep-comparison__sparkline-row">
                        <div className="sweep-comparison__sparkline">
                            <span className="sweep-comparison__sparkline-label">A</span>
                            <Sparkline
                                data={lossHistoryA}
                                width={200}
                                height={40}
                                decreaseIsGood={true}
                                showExtremes={true}
                            />
                        </div>
                        <div className="sweep-comparison__sparkline">
                            <span className="sweep-comparison__sparkline-label">B</span>
                            <Sparkline
                                data={lossHistoryB}
                                width={200}
                                height={40}
                                decreaseIsGood={true}
                                showExtremes={true}
                            />
                        </div>
                    </div>
                </div>
            )}

            {/* Comparison Table */}
            <ComparisonTable sweepA={sweepA} sweepB={sweepB} metrics={COMPARISON_METRICS} />
        </div>
    );
}
