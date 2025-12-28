// HPODashboard.tsx - Unified live dashboard for HPO optimization
// Enterprise-grade monitoring interface with real-time metrics

import { useState, useMemo, useCallback, useEffect } from 'react';
import {
    Pause,
    Square,
    Settings,
    ChevronDown,
    ChevronUp,
    Zap,
    Clock,
    Cpu,
    Target,
    TrendingDown,
    Activity,
    BarChart3,
    Brain,
    AlertTriangle,
    Compass,
    Battery,
    Database
} from 'lucide-react';
import { Card, CardContent, Button, ProgressBar } from './ui';
import { LossChart } from './charts/LossChart';
import { MetricGauge, ConfidenceGauge } from './charts/MetricGauge';
import { TrialResultsTable } from './TrialResultsTable';
import { TrainingConsole } from './TrainingConsole';
import HPOImportanceChart from './HPOImportanceChart';
import type { HPOSweepInfo, HPOTrialInfo, SmartTunerStatus, TunerMemoryStats } from '../api/types';
import './HPODashboard.css';

interface HPODashboardProps {
    sweepStatus: HPOSweepInfo | null;
    trials: HPOTrialInfo[];
    isRunning: boolean;
    devMode?: boolean;
    onPause?: () => void;
    onStop?: () => void;
    onShowSettings?: () => void;
    config?: {
        curriculumName?: string;
        paramBudget?: number;
        optimizer?: string;
        optimizationMode?: string;
        maxTrials?: number;
    };
}

// Helper to format parameters
function formatParams(num: number): string {
    if (num >= 1_000_000_000) return `${(num / 1_000_000_000).toFixed(1)}B`;
    if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(0)}M`;
    if (num >= 1_000) return `${(num / 1_000).toFixed(0)}K`;
    return num.toString();
}

// Helper to format duration
function formatDuration(seconds: number): string {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
}

export function HPODashboard({
    sweepStatus,
    trials,
    isRunning,
    devMode = false,
    onPause,
    onStop,
    onShowSettings,
    config,
}: HPODashboardProps) {
    const [configExpanded, setConfigExpanded] = useState(false);
    const [consoleExpanded, setConsoleExpanded] = useState(false);
    const [smartTunerExpanded, setSmartTunerExpanded] = useState(false);

    // Smart Tuner status state
    const [smartTunerStatus, setSmartTunerStatus] = useState<SmartTunerStatus | null>(null);
    const [tunerMemoryStats, setTunerMemoryStats] = useState<TunerMemoryStats | null>(null);

    // Fetch Smart Tuner status when sweep is running
    useEffect(() => {
        if (!sweepStatus?.sweep_id || !isRunning) {
            return;
        }

        const fetchSmartTunerStatus = async () => {
            try {
                const response = await fetch(`/api/smart-tuner/status?sweep_id=${sweepStatus.sweep_id}`);
                if (response.ok) {
                    const data = await response.json();
                    setSmartTunerStatus(data);
                }
            } catch (err) {
                console.debug('[SmartTuner] Failed to fetch status:', err);
            }
        };

        const fetchMemoryStats = async () => {
            try {
                const response = await fetch('/api/smart-tuner/memory/stats');
                if (response.ok) {
                    const data = await response.json();
                    setTunerMemoryStats(data.stats);
                }
            } catch (err) {
                console.debug('[SmartTuner] Failed to fetch memory stats:', err);
            }
        };

        // Initial fetch
        fetchSmartTunerStatus();
        fetchMemoryStats();

        // Poll every 5 seconds
        const interval = setInterval(() => {
            fetchSmartTunerStatus();
            fetchMemoryStats();
        }, 5000);

        return () => clearInterval(interval);
    }, [sweepStatus?.sweep_id, isRunning]);

    // Helper for phase styling
    const getPhaseInfo = (phase: string) => {
        switch (phase) {
            case 'warmup':
                return { icon: <Battery size={14} />, color: 'var(--color-warning)', label: 'Warmup' };
            case 'exploration':
                return { icon: <Compass size={14} />, color: 'var(--color-primary)', label: 'Exploration' };
            case 'exploitation':
                return { icon: <Target size={14} />, color: 'var(--color-success)', label: 'Exploitation' };
            case 'emergency':
                return { icon: <AlertTriangle size={14} />, color: 'var(--color-danger)', label: 'Emergency' };
            default:
                return { icon: <Brain size={14} />, color: 'var(--color-text-muted)', label: 'Idle' };
        }
    };

    // Compute current metrics from latest trial
    const currentMetrics = useMemo(() => {
        const runningTrial = trials.find(t => t.status === 'running');
        const completedTrials = trials.filter(t => t.status === 'completed');
        const bestTrial = completedTrials.reduce((best, t) => {
            if (!best) return t;
            return (t.composite_score ?? 0) > (best.composite_score ?? 0) ? t : best;
        }, null as HPOTrialInfo | null);

        // Check if running in convergence mode (max_trials >= 999 or explicitly set)
        const maxTrials = sweepStatus?.max_trials ?? config?.maxTrials ?? 0;
        const isConvergenceMode = maxTrials >= 999;

        return {
            currentTrial: runningTrial,
            bestTrial,
            completedCount: completedTrials.length,
            totalTrials: maxTrials,
            isConvergenceMode,
            bestLoss: sweepStatus?.best_loss ?? bestTrial?.loss,
            bestPerplexity: sweepStatus?.best_perplexity ?? bestTrial?.perplexity,
            bestConfidence: sweepStatus?.best_confidence ?? bestTrial?.mean_confidence,
            bestComposite: sweepStatus?.best_composite_score ?? bestTrial?.composite_score,
        };
    }, [trials, sweepStatus, config]);

    // Transform trials to chart data
    const chartData = useMemo(() => {
        // Group trials and create points
        // For now, we'll use loss progression per trial
        return trials.map(trial => ({
            trialId: trial.trial_id,
            status: trial.status,
            isBest: trial.trial_id === sweepStatus?.best_trial_id,
            points: trial.loss !== null && trial.loss !== undefined
                ? [{ step: trial.step ?? 0, loss: trial.loss, trialId: trial.trial_id }]
                : [],
        }));
    }, [trials, sweepStatus]);

    // Get elapsed time
    const elapsedTime = useMemo(() => {
        if (!sweepStatus?.started_at) return null;
        const start = new Date(sweepStatus.started_at).getTime();
        const now = Date.now();
        return Math.floor((now - start) / 1000);
    }, [sweepStatus?.started_at]);

    // Get sweep state label
    const stateLabel = useMemo(() => {
        if (!sweepStatus) return 'Initializing...';
        switch (sweepStatus.state) {
            case 'running': return 'Optimizing';
            case 'completed': return 'Completed';
            case 'failed': return 'Failed';
            case 'cancelled': return 'Cancelled';
            case 'paused': return 'Paused';
            default: return sweepStatus.state;
        }
    }, [sweepStatus?.state]);

    return (
        <div className="hpo-dashboard">
            {/* Hero Section */}
            <div className="hpo-dashboard__hero">
                <div className="hpo-dashboard__hero-left">
                    <div className="hpo-dashboard__status">
                        <span className={`hpo-dashboard__status-dot ${isRunning ? 'hpo-dashboard__status-dot--running' : ''}`} />
                        <span className="hpo-dashboard__status-text">{stateLabel}</span>
                        <span className="hpo-dashboard__status-badge">QAHPO</span>
                    </div>
                    <div className="hpo-dashboard__progress-info">
                        <span className="hpo-dashboard__trial-count">
                            {currentMetrics.isConvergenceMode
                                ? `Trial ${currentMetrics.completedCount + (currentMetrics.currentTrial ? 1 : 0)}`
                                : `Trial ${currentMetrics.completedCount + (currentMetrics.currentTrial ? 1 : 0)} / ${currentMetrics.totalTrials}`
                            }
                        </span>
                        {elapsedTime !== null && (
                            <span className="hpo-dashboard__elapsed">
                                <Clock size={14} />
                                {formatDuration(elapsedTime)}
                            </span>
                        )}
                    </div>
                    {currentMetrics.isConvergenceMode ? (
                        <div className="hpo-dashboard__convergence-indicator">
                            <span className="hpo-dashboard__convergence-text">Running until convergence...</span>
                        </div>
                    ) : (
                        <ProgressBar
                            value={currentMetrics.completedCount}
                            max={currentMetrics.totalTrials}
                            variant="gradient"
                            size="lg"
                            animated={isRunning}
                            showLabel
                        />
                    )}
                </div>

                <div className="hpo-dashboard__hero-right">
                    <div className="hpo-dashboard__actions">
                        {isRunning && (
                            <>
                                <Button
                                    variant="secondary"
                                    size="sm"
                                    leftIcon={<Pause size={16} />}
                                    onClick={onPause}
                                >
                                    Pause
                                </Button>
                                <Button
                                    variant="danger"
                                    size="sm"
                                    leftIcon={<Square size={16} />}
                                    onClick={onStop}
                                >
                                    Stop
                                </Button>
                            </>
                        )}
                        <Button
                            variant="ghost"
                            size="sm"
                            leftIcon={<Settings size={16} />}
                            onClick={onShowSettings}
                        >
                            Settings
                        </Button>
                    </div>
                </div>
            </div>

            {/* Configuration Summary (collapsible) */}
            <div className={`hpo-dashboard__config ${configExpanded ? 'hpo-dashboard__config--expanded' : ''}`}>
                <button
                    className="hpo-dashboard__config-toggle"
                    onClick={() => setConfigExpanded(!configExpanded)}
                >
                    <div className="hpo-dashboard__config-summary">
                        {config?.paramBudget && (
                            <span className="hpo-dashboard__config-item">
                                <Cpu size={14} />
                                {formatParams(config.paramBudget)}
                            </span>
                        )}
                        {config?.optimizer && (
                            <span className="hpo-dashboard__config-item">
                                <Zap size={14} />
                                {config.optimizer}
                            </span>
                        )}
                        {config?.curriculumName && (
                            <span className="hpo-dashboard__config-item">
                                <Target size={14} />
                                {config.curriculumName}
                            </span>
                        )}
                    </div>
                    {configExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>
                {configExpanded && (
                    <div className="hpo-dashboard__config-details">
                        <div className="hpo-dashboard__config-grid">
                            {config?.curriculumName && (
                                <div className="hpo-dashboard__config-detail">
                                    <span className="hpo-dashboard__config-label">Curriculum</span>
                                    <span className="hpo-dashboard__config-value">{config.curriculumName}</span>
                                </div>
                            )}
                            {config?.paramBudget && (
                                <div className="hpo-dashboard__config-detail">
                                    <span className="hpo-dashboard__config-label">Parameter Budget</span>
                                    <span className="hpo-dashboard__config-value">{formatParams(config.paramBudget)}</span>
                                </div>
                            )}
                            {config?.optimizer && (
                                <div className="hpo-dashboard__config-detail">
                                    <span className="hpo-dashboard__config-label">Optimizer</span>
                                    <span className="hpo-dashboard__config-value">{config.optimizer}</span>
                                </div>
                            )}
                            {config?.optimizationMode && (
                                <div className="hpo-dashboard__config-detail">
                                    <span className="hpo-dashboard__config-label">Mode</span>
                                    <span className="hpo-dashboard__config-value">{config.optimizationMode}</span>
                                </div>
                            )}
                            <div className="hpo-dashboard__config-detail">
                                <span className="hpo-dashboard__config-label">Search Strategy</span>
                                <span className="hpo-dashboard__config-value">Quantum Adaptive HPO</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Metrics Section */}
            <div className="hpo-dashboard__metrics">
                <div className="hpo-dashboard__metric-card hpo-dashboard__metric-card--primary">
                    <div className="hpo-dashboard__metric-header">
                        <TrendingDown size={18} />
                        <span>Best Loss</span>
                    </div>
                    <div className="hpo-dashboard__metric-value">
                        {currentMetrics.bestLoss?.toFixed(4) ?? '—'}
                    </div>
                    {currentMetrics.bestTrial && (
                        <div className="hpo-dashboard__metric-sub">
                            Trial #{typeof currentMetrics.bestTrial.trial_id === 'string'
                                ? currentMetrics.bestTrial.trial_id.replace(/\D/g, '')
                                : currentMetrics.bestTrial.trial_id}
                        </div>
                    )}
                </div>

                <div className="hpo-dashboard__metric-card">
                    <div className="hpo-dashboard__metric-header">
                        <Activity size={18} />
                        <span>Best Perplexity</span>
                    </div>
                    <div className="hpo-dashboard__metric-value">
                        {currentMetrics.bestPerplexity?.toFixed(2) ?? '—'}
                    </div>
                </div>

                <div className="hpo-dashboard__metric-card">
                    <div className="hpo-dashboard__metric-header">
                        <BarChart3 size={18} />
                        <span>Best Confidence</span>
                    </div>
                    <div className="hpo-dashboard__metric-value">
                        {currentMetrics.bestConfidence
                            ? `${(currentMetrics.bestConfidence * 100).toFixed(1)}%`
                            : '—'}
                    </div>
                </div>

                <div className="hpo-dashboard__metric-card hpo-dashboard__metric-card--highlight">
                    <div className="hpo-dashboard__metric-header">
                        <Target size={18} />
                        <span>Composite Score</span>
                    </div>
                    <div className="hpo-dashboard__metric-value">
                        {currentMetrics.bestComposite?.toFixed(4) ?? '—'}
                    </div>
                </div>

                {/* Best Architecture Card - shows winning configuration */}
                {sweepStatus?.best_hyperparams && (
                    <div className="hpo-dashboard__metric-card hpo-dashboard__metric-card--wide">
                        <div className="hpo-dashboard__metric-header">
                            <Cpu size={18} />
                            <span>Best Architecture</span>
                        </div>
                        <div className="hpo-dashboard__architecture-grid">
                            {sweepStatus.best_hyperparams.hidden_dim && (
                                <div className="hpo-dashboard__arch-item">
                                    <span className="hpo-dashboard__arch-label">Dim</span>
                                    <span className="hpo-dashboard__arch-value">
                                        {sweepStatus.best_hyperparams.hidden_dim}
                                    </span>
                                </div>
                            )}
                            {sweepStatus.best_hyperparams.num_reasoning_blocks && (
                                <div className="hpo-dashboard__arch-item">
                                    <span className="hpo-dashboard__arch-label">Blocks</span>
                                    <span className="hpo-dashboard__arch-value">
                                        {sweepStatus.best_hyperparams.num_reasoning_blocks}
                                    </span>
                                </div>
                            )}
                            {sweepStatus.best_hyperparams.num_moe_experts && (
                                <div className="hpo-dashboard__arch-item">
                                    <span className="hpo-dashboard__arch-label">Experts</span>
                                    <span className="hpo-dashboard__arch-value">
                                        {sweepStatus.best_hyperparams.num_moe_experts}
                                    </span>
                                </div>
                            )}
                            {sweepStatus.best_hyperparams.learning_rate && (
                                <div className="hpo-dashboard__arch-item">
                                    <span className="hpo-dashboard__arch-label">LR</span>
                                    <span className="hpo-dashboard__arch-value">
                                        {typeof sweepStatus.best_hyperparams.learning_rate === 'number'
                                            ? sweepStatus.best_hyperparams.learning_rate.toExponential(2)
                                            : sweepStatus.best_hyperparams.learning_rate}
                                    </span>
                                </div>
                            )}
                            {sweepStatus.best_hyperparams.optimizer && (
                                <div className="hpo-dashboard__arch-item">
                                    <span className="hpo-dashboard__arch-label">Optimizer</span>
                                    <span className="hpo-dashboard__arch-value">
                                        {sweepStatus.best_hyperparams.optimizer}
                                    </span>
                                </div>
                            )}
                            {sweepStatus.best_memory_mb && (
                                <div className="hpo-dashboard__arch-item">
                                    <span className="hpo-dashboard__arch-label">Memory</span>
                                    <span className="hpo-dashboard__arch-value">
                                        {sweepStatus.best_memory_mb.toFixed(0)}MB
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {/* Smart Tuner Status Section */}
            {(smartTunerStatus || tunerMemoryStats) && (
                <div className={`hpo-dashboard__smart-tuner ${smartTunerExpanded ? 'hpo-dashboard__smart-tuner--expanded' : ''}`}>
                    <button
                        className="hpo-dashboard__smart-tuner-toggle"
                        onClick={() => setSmartTunerExpanded(!smartTunerExpanded)}
                    >
                        <div className="hpo-dashboard__smart-tuner-summary">
                            <Brain size={16} />
                            <span className="hpo-dashboard__smart-tuner-title">Smart Tuner</span>
                            {smartTunerStatus && (
                                <>
                                    <span
                                        className="hpo-dashboard__smart-tuner-phase"
                                        style={{ color: getPhaseInfo(smartTunerStatus.current_phase).color }}
                                    >
                                        {getPhaseInfo(smartTunerStatus.current_phase).icon}
                                        {getPhaseInfo(smartTunerStatus.current_phase).label}
                                    </span>
                                    {smartTunerStatus.emergency_mode && (
                                        <span className="hpo-dashboard__smart-tuner-emergency">
                                            <AlertTriangle size={14} />
                                            Emergency Mode
                                        </span>
                                    )}
                                </>
                            )}
                            {tunerMemoryStats && tunerMemoryStats.trial_count > 0 && (
                                <span className="hpo-dashboard__smart-tuner-memory">
                                    <Database size={14} />
                                    {tunerMemoryStats.trial_count} learned trials
                                </span>
                            )}
                        </div>
                        {smartTunerExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </button>
                    {smartTunerExpanded && (
                        <div className="hpo-dashboard__smart-tuner-details">
                            <div className="hpo-dashboard__smart-tuner-grid">
                                {smartTunerStatus && (
                                    <>
                                        <div className="hpo-dashboard__smart-tuner-item">
                                            <span className="hpo-dashboard__smart-tuner-label">Phase</span>
                                            <span
                                                className="hpo-dashboard__smart-tuner-value"
                                                style={{ color: getPhaseInfo(smartTunerStatus.current_phase).color }}
                                            >
                                                {getPhaseInfo(smartTunerStatus.current_phase).icon}
                                                {getPhaseInfo(smartTunerStatus.current_phase).label}
                                            </span>
                                        </div>
                                        <div className="hpo-dashboard__smart-tuner-item">
                                            <span className="hpo-dashboard__smart-tuner-label">Exploration Factor</span>
                                            <span className="hpo-dashboard__smart-tuner-value">
                                                {(smartTunerStatus.exploration_factor * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                        <div className="hpo-dashboard__smart-tuner-item">
                                            <span className="hpo-dashboard__smart-tuner-label">Coordination Mode</span>
                                            <span className="hpo-dashboard__smart-tuner-value">
                                                {smartTunerStatus.coordination_mode.charAt(0).toUpperCase() +
                                                    smartTunerStatus.coordination_mode.slice(1)}
                                            </span>
                                        </div>
                                        <div className="hpo-dashboard__smart-tuner-item">
                                            <span className="hpo-dashboard__smart-tuner-label">Global Step</span>
                                            <span className="hpo-dashboard__smart-tuner-value">
                                                {smartTunerStatus.global_step.toLocaleString()}
                                            </span>
                                        </div>
                                    </>
                                )}
                                {tunerMemoryStats && !tunerMemoryStats.error && (
                                    <>
                                        <div className="hpo-dashboard__smart-tuner-item">
                                            <span className="hpo-dashboard__smart-tuner-label">Memory Trials</span>
                                            <span className="hpo-dashboard__smart-tuner-value">
                                                {tunerMemoryStats.trial_count}
                                            </span>
                                        </div>
                                        {tunerMemoryStats.best_loss !== null && (
                                            <div className="hpo-dashboard__smart-tuner-item">
                                                <span className="hpo-dashboard__smart-tuner-label">Best Historical Loss</span>
                                                <span className="hpo-dashboard__smart-tuner-value">
                                                    {tunerMemoryStats.best_loss.toFixed(4)}
                                                </span>
                                            </div>
                                        )}
                                        <div className="hpo-dashboard__smart-tuner-item">
                                            <span className="hpo-dashboard__smart-tuner-label">Unique Architectures</span>
                                            <span className="hpo-dashboard__smart-tuner-value">
                                                {tunerMemoryStats.unique_architectures}
                                            </span>
                                        </div>
                                        {tunerMemoryStats.converged_count !== undefined && (
                                            <div className="hpo-dashboard__smart-tuner-item">
                                                <span className="hpo-dashboard__smart-tuner-label">Converged Trials</span>
                                                <span className="hpo-dashboard__smart-tuner-value">
                                                    {tunerMemoryStats.converged_count}
                                                </span>
                                            </div>
                                        )}
                                    </>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Charts Section */}
            <div className="hpo-dashboard__charts">
                <div className="hpo-dashboard__chart-main">
                    <LossChart
                        trials={chartData}
                        currentTrialId={currentMetrics.currentTrial?.trial_id}
                        bestTrialId={sweepStatus?.best_trial_id}
                        height={300}
                        showLegend
                    />
                </div>
                <div className="hpo-dashboard__chart-gauges">
                    <MetricGauge
                        value={currentMetrics.bestLoss ?? null}
                        label="Loss"
                        min={0}
                        max={1}
                        thresholds={{ good: 20, warning: 50, danger: 80 }}
                        inverse
                        size="lg"
                        formatValue={(v) => v.toFixed(4)}
                    />
                    <ConfidenceGauge value={currentMetrics.bestConfidence ?? null} />
                </div>
            </div>

            {/* Trial Results */}
            <div className="hpo-dashboard__trials">
                <TrialResultsTable
                    trials={trials}
                    bestTrialId={sweepStatus?.best_trial_id}
                    currentTrialId={currentMetrics.currentTrial?.trial_id}
                    maxHeight={350}
                />
            </div>

            {/* Hyperparameter Importance Analysis */}
            {sweepStatus?.sweep_id && (
                <HPOImportanceChart
                    sweepId={sweepStatus.sweep_id}
                    isRunning={isRunning}
                    completedTrials={sweepStatus.completed_trials}
                />
            )}

            {/* Training Console */}
            <div className={`hpo-dashboard__console ${consoleExpanded ? 'hpo-dashboard__console--expanded' : ''}`}>
                <button
                    className="hpo-dashboard__console-toggle"
                    onClick={() => setConsoleExpanded(!consoleExpanded)}
                >
                    <span>Training Console</span>
                    {consoleExpanded ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
                </button>
                {consoleExpanded && (
                    <TrainingConsole
                        sweepId={sweepStatus?.sweep_id || null}
                        isRunning={isRunning}
                        devMode={devMode}
                    />
                )}
            </div>
        </div>
    );
}

export default HPODashboard;
