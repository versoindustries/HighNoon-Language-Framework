// CockpitHUD.tsx - Main F1/Jet-style Cockpit Heads-Up Display
// Unified dashboard for HPO optimization with tabbed Trial Results

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
    Activity,
    BarChart3,
    List,
    TrendingDown,
    Zap,
    Cpu,
    Clock,
    Gauge,
    Target,
    ChevronDown,
    ChevronUp,
} from 'lucide-react';
import { HUDSpeedometer, HUDTachometer, HUDStatusRing, HUDAlertPanel, createBarrenPlateauAlert, createMemoryAlert, createGradientAlert } from './hud';
import type { HUDAlert } from './hud';
import { LossChart } from './charts/LossChart';
import { TrialResultsTable } from './TrialResultsTable';
import HPOImportanceChart from './HPOImportanceChart';
import type { HPOSweepInfo, HPOTrialInfo } from '../api/types';
import './CockpitHUD.css';

interface QULSTelemetry {
    type: string;
    timestamp: string;
    job_id: string;
    loss: {
        total: number;
        fidelity: number;
        entropy: number;
        coherence: number;
        symplectic: number;
        born_rule: number;
        holographic: number;
        barren_plateau_detected: boolean;
    };
    hpo: {
        trial_id: number;
        tunneling_probability: number;
        annealing_temperature: number;
        exploration_mode: string;
    };
    memory: {
        rss_mb: number;
        peak_mb: number;
        gpu_allocated_mb: number;
    };
    gradients: {
        global_norm: number;
        vqc_variance: number;
        max_layer_norm: number;
    };
    training: {
        epoch: number;
        step: number;
        learning_rate: number;
        throughput_samples_sec: number;
    };
    // Quality metrics for CockpitHUD gauges
    quality?: {
        perplexity: number | null;
        mean_confidence: number | null;
    };
}

interface CockpitHUDProps {
    sweepStatus: HPOSweepInfo | null;
    trials: HPOTrialInfo[];
    isRunning: boolean;
    devMode?: boolean;
    onPause?: () => void;
    onStop?: () => void;
    config?: {
        curriculumName?: string;
        paramBudget?: number;
        optimizer?: string;
    };
}

type TabId = 'trials' | 'importance' | 'details';

// Helper formatters
function formatDuration(seconds: number): string {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
}

function formatParams(num: number): string {
    if (num >= 1_000_000_000) return `${(num / 1_000_000_000).toFixed(1)}B`;
    if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(0)}M`;
    if (num >= 1_000) return `${(num / 1_000).toFixed(0)}K`;
    return num.toString();
}

export function CockpitHUD({
    sweepStatus,
    trials,
    isRunning,
    devMode = false,
    onPause,
    onStop,
    config,
}: CockpitHUDProps) {
    const [activeTab, setActiveTab] = useState<TabId>('trials');
    const [telemetry, setTelemetry] = useState<QULSTelemetry | null>(null);
    const [wsConnected, setWsConnected] = useState(false);
    const [alerts, setAlerts] = useState<HUDAlert[]>([]);
    const wsRef = useRef<WebSocket | null>(null);

    // Connect to telemetry WebSocket
    useEffect(() => {
        if (!sweepStatus?.sweep_id || !isRunning) {
            setWsConnected(false);
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/telemetry/${sweepStatus.sweep_id}`;

        try {
            wsRef.current = new WebSocket(wsUrl);

            wsRef.current.onopen = () => {
                setWsConnected(true);
            };

            wsRef.current.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'full_telemetry') {
                        setTelemetry(data);
                        // Check for alerts
                        updateAlerts(data);
                    } else if (data.type === 'finished') {
                        setWsConnected(false);
                    } else if (data.type === 'waiting') {
                        // Sweep is starting up, no trial data yet - keep connected
                        console.debug('[CockpitHUD] Waiting for training data:', data.message);
                    } else if (data.error) {
                        // Handle error messages gracefully - don't disconnect immediately
                        console.warn('[CockpitHUD] Telemetry warning:', data.error);
                    }
                } catch (err) {
                    console.error('[CockpitHUD] Failed to parse telemetry:', err);
                }
            };

            wsRef.current.onerror = () => setWsConnected(false);
            wsRef.current.onclose = () => setWsConnected(false);
        } catch (err) {
            console.error('[CockpitHUD] WebSocket connection failed:', err);
        }

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, [sweepStatus?.sweep_id, isRunning]);

    // Update alerts based on telemetry
    const updateAlerts = useCallback((data: QULSTelemetry) => {
        const newAlerts: HUDAlert[] = [];

        // Barren plateau detection
        if (data.loss.barren_plateau_detected) {
            newAlerts.push(createBarrenPlateauAlert(data.gradients.vqc_variance));
        }

        // Memory pressure (>85%)
        const memoryPct = (data.memory.rss_mb / 64000) * 100; // Assuming 64GB system
        if (memoryPct > 85) {
            newAlerts.push(createMemoryAlert(memoryPct));
        }

        // Gradient issues
        if (data.gradients.global_norm > 100) {
            newAlerts.push(createGradientAlert('explosion', data.gradients.global_norm));
        } else if (data.gradients.global_norm < 1e-6) {
            newAlerts.push(createGradientAlert('vanishing', data.gradients.global_norm));
        }

        setAlerts(newAlerts);
    }, []);

    // Compute metrics from trials and telemetry
    const metrics = useMemo(() => {
        const completedTrials = trials.filter(t => t.status === 'completed');
        const bestTrial = completedTrials.reduce((best, t) => {
            if (!best) return t;
            return (t.composite_score ?? 0) > (best.composite_score ?? 0) ? t : best;
        }, null as HPOTrialInfo | null);

        return {
            completedCount: completedTrials.length,
            totalTrials: sweepStatus?.max_trials ?? 0,
            bestLoss: sweepStatus?.best_loss ?? bestTrial?.loss ?? null,
            // Use real-time telemetry quality metrics, fallback to sweep/trial values
            bestPerplexity: telemetry?.quality?.perplexity ?? sweepStatus?.best_perplexity ?? bestTrial?.perplexity ?? null,
            bestConfidence: telemetry?.quality?.mean_confidence ?? sweepStatus?.best_confidence ?? bestTrial?.mean_confidence ?? null,
            bestComposite: sweepStatus?.best_composite_score ?? bestTrial?.composite_score ?? null,
            currentTrial: trials.find(t => t.status === 'running'),
        };
    }, [trials, sweepStatus, telemetry]);

    // Elapsed time
    const elapsedTime = useMemo(() => {
        if (!sweepStatus?.started_at) return null;
        const start = new Date(sweepStatus.started_at).getTime();
        return Math.floor((Date.now() - start) / 1000);
    }, [sweepStatus?.started_at]);

    // Chart data transformation
    const chartData = useMemo(() => {
        return trials.map(trial => ({
            trialId: trial.trial_id,
            status: trial.status,
            isBest: trial.trial_id === sweepStatus?.best_trial_id,
            points: trial.loss !== null && trial.loss !== undefined
                ? [{ step: trial.step ?? 0, loss: trial.loss, trialId: trial.trial_id }]
                : [],
        }));
    }, [trials, sweepStatus]);

    // Determine optimization phase
    const phase = useMemo(() => {
        if (!isRunning) return 'idle';
        if (telemetry?.loss.barren_plateau_detected) return 'emergency';
        const mode = telemetry?.hpo.exploration_mode ?? 'exploit';
        if (mode === 'explore') return 'exploration';
        if (mode === 'exploit') return 'exploitation';
        return 'warmup';
    }, [isRunning, telemetry]);

    // Dismiss alert handler
    const handleDismissAlert = useCallback((id: string) => {
        setAlerts(prev => prev.filter(a => a.id !== id));
    }, []);

    const tabs = [
        { id: 'trials' as TabId, label: 'Trials', icon: <List size={14} /> },
        { id: 'importance' as TabId, label: 'Importance', icon: <BarChart3 size={14} /> },
        { id: 'details' as TabId, label: 'Config', icon: <Cpu size={14} /> },
    ];

    return (
        <div className="cockpit-hud">
            {/* Scanline overlay for retro effect */}
            <div className="cockpit-hud__scanlines" />

            {/* Header bar */}
            <div className="cockpit-hud__header">
                <div className="cockpit-hud__header-left">
                    <div className={`cockpit-hud__status ${isRunning ? 'cockpit-hud__status--running' : ''}`}>
                        <span className="cockpit-hud__status-dot" />
                        <span className="cockpit-hud__status-text">
                            {isRunning ? 'OPTIMIZING' : 'STANDBY'}
                        </span>
                    </div>
                    <span className="cockpit-hud__badge">QAHPO</span>
                    {wsConnected && (
                        <span className="cockpit-hud__ws-status">
                            <Activity size={12} />
                            LIVE
                        </span>
                    )}
                </div>
                <div className="cockpit-hud__header-center">
                    <span className="cockpit-hud__title">QUANTUM ADAPTIVE HPO</span>
                </div>
                <div className="cockpit-hud__header-right">
                    {elapsedTime !== null && (
                        <span className="cockpit-hud__elapsed">
                            <Clock size={14} />
                            {formatDuration(elapsedTime)}
                        </span>
                    )}
                    <span className="cockpit-hud__trial-counter">
                        TRIAL {metrics.completedCount + (metrics.currentTrial ? 1 : 0)}
                        {metrics.totalTrials > 0 && metrics.totalTrials < 999 && ` / ${metrics.totalTrials}`}
                    </span>
                </div>
            </div>

            {/* Main instrument cluster */}
            <div className="cockpit-hud__instruments">
                {/* Primary gauges row */}
                <div className="cockpit-hud__primary-gauges">
                    <HUDSpeedometer
                        value={metrics.bestLoss}
                        label="LOSS"
                        sublabel="Best"
                        min={0}
                        max={1}
                        inverse
                        formatValue={(v) => v.toFixed(4)}
                        size="lg"
                    />
                    <HUDSpeedometer
                        value={metrics.bestPerplexity}
                        label="PERPLEXITY"
                        sublabel="Best"
                        min={1}
                        max={100}
                        inverse
                        formatValue={(v) => v.toFixed(1)}
                        size="md"
                    />

                    {/* Status ring in center */}
                    <HUDStatusRing
                        phase={phase}
                        trialNumber={telemetry?.hpo.trial_id ?? metrics.currentTrial?.trial_id as number}
                        elapsedTime={elapsedTime ? formatDuration(elapsedTime) : undefined}
                        isRunning={isRunning}
                        emergencyMode={telemetry?.loss.barren_plateau_detected}
                        tunneling={telemetry?.hpo.tunneling_probability ?? 0}
                        temperature={telemetry?.hpo.annealing_temperature ?? 1.0}
                    />

                    <HUDSpeedometer
                        value={metrics.bestConfidence !== null ? metrics.bestConfidence * 100 : null}
                        label="CONFIDENCE"
                        sublabel="Best"
                        unit="%"
                        min={0}
                        max={100}
                        formatValue={(v) => v.toFixed(1)}
                        size="md"
                    />
                    <HUDSpeedometer
                        value={metrics.bestComposite}
                        label="COMPOSITE"
                        sublabel="Score"
                        min={0}
                        max={1}
                        formatValue={(v) => v.toFixed(4)}
                        size="lg"
                    />
                </div>

                {/* Secondary telemetry gauges */}
                <div className="cockpit-hud__telemetry-row">
                    <HUDTachometer
                        value={telemetry?.training.learning_rate ?? null}
                        label="LR"
                        max={0.001}
                        formatValue={(v) => v.toExponential(1)}
                        size="sm"
                    />
                    <HUDTachometer
                        value={telemetry?.gradients.global_norm ?? null}
                        label="GRAD"
                        max={10}
                        redline={80}
                        formatValue={(v) => v.toFixed(2)}
                        size="sm"
                    />
                    <HUDTachometer
                        value={telemetry?.memory.rss_mb ? telemetry.memory.rss_mb / 1024 : null}
                        label="RAM"
                        max={64}
                        redline={85}
                        formatValue={(v) => `${v.toFixed(0)}G`}
                        size="sm"
                    />
                    <HUDTachometer
                        value={telemetry?.memory.gpu_allocated_mb ? telemetry.memory.gpu_allocated_mb / 1024 : null}
                        label="GPU"
                        max={48}
                        redline={90}
                        formatValue={(v) => `${v.toFixed(0)}G`}
                        size="sm"
                    />
                    <HUDTachometer
                        value={telemetry?.training.throughput_samples_sec ?? null}
                        label="TOK/S"
                        max={5000}
                        formatValue={(v) => v >= 1000 ? `${(v / 1000).toFixed(1)}K` : v.toFixed(0)}
                        size="sm"
                    />
                </div>
            </div>

            {/* Alert panel */}
            <div className="cockpit-hud__alerts">
                <HUDAlertPanel
                    alerts={alerts}
                    onDismiss={handleDismissAlert}
                    compact
                />
            </div>

            {/* Loss convergence chart */}
            <div className="cockpit-hud__chart">
                <LossChart
                    trials={chartData}
                    currentTrialId={metrics.currentTrial?.trial_id}
                    bestTrialId={sweepStatus?.best_trial_id}
                    height={220}
                    showLegend={false}
                />
            </div>

            {/* Tabbed content area */}
            <div className="cockpit-hud__tabs">
                <div className="cockpit-hud__tab-list">
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            className={`cockpit-hud__tab ${activeTab === tab.id ? 'cockpit-hud__tab--active' : ''}`}
                            onClick={() => setActiveTab(tab.id)}
                        >
                            {tab.icon}
                            <span>{tab.label}</span>
                        </button>
                    ))}
                </div>
                <div className="cockpit-hud__tab-content">
                    {activeTab === 'trials' && (
                        <TrialResultsTable
                            trials={trials}
                            bestTrialId={sweepStatus?.best_trial_id}
                            currentTrialId={metrics.currentTrial?.trial_id}
                            maxHeight={300}
                        />
                    )}
                    {activeTab === 'importance' && sweepStatus?.sweep_id && (
                        <HPOImportanceChart
                            sweepId={sweepStatus.sweep_id}
                            isRunning={isRunning}
                            completedTrials={sweepStatus.completed_trials}
                        />
                    )}
                    {activeTab === 'details' && (
                        <div className="cockpit-hud__details">
                            <div className="cockpit-hud__details-grid">
                                {config?.paramBudget && (
                                    <div className="cockpit-hud__detail-item">
                                        <span className="cockpit-hud__detail-label">Parameter Budget</span>
                                        <span className="cockpit-hud__detail-value">{formatParams(config.paramBudget)}</span>
                                    </div>
                                )}
                                {config?.optimizer && (
                                    <div className="cockpit-hud__detail-item">
                                        <span className="cockpit-hud__detail-label">Optimizer</span>
                                        <span className="cockpit-hud__detail-value">{config.optimizer}</span>
                                    </div>
                                )}
                                {config?.curriculumName && (
                                    <div className="cockpit-hud__detail-item">
                                        <span className="cockpit-hud__detail-label">Curriculum</span>
                                        <span className="cockpit-hud__detail-value">{config.curriculumName}</span>
                                    </div>
                                )}
                                {sweepStatus?.best_hyperparams && (
                                    <>
                                        {sweepStatus.best_hyperparams.hidden_dim && (
                                            <div className="cockpit-hud__detail-item">
                                                <span className="cockpit-hud__detail-label">Hidden Dim</span>
                                                <span className="cockpit-hud__detail-value">{sweepStatus.best_hyperparams.hidden_dim as number}</span>
                                            </div>
                                        )}
                                        {sweepStatus.best_hyperparams.num_reasoning_blocks && (
                                            <div className="cockpit-hud__detail-item">
                                                <span className="cockpit-hud__detail-label">Blocks</span>
                                                <span className="cockpit-hud__detail-value">{sweepStatus.best_hyperparams.num_reasoning_blocks as number}</span>
                                            </div>
                                        )}
                                        {sweepStatus.best_hyperparams.num_moe_experts && (
                                            <div className="cockpit-hud__detail-item">
                                                <span className="cockpit-hud__detail-label">Experts</span>
                                                <span className="cockpit-hud__detail-value">{sweepStatus.best_hyperparams.num_moe_experts as number}</span>
                                            </div>
                                        )}
                                        {sweepStatus.best_hyperparams.learning_rate && (
                                            <div className="cockpit-hud__detail-item">
                                                <span className="cockpit-hud__detail-label">Learning Rate</span>
                                                <span className="cockpit-hud__detail-value">
                                                    {typeof sweepStatus.best_hyperparams.learning_rate === 'number'
                                                        ? (sweepStatus.best_hyperparams.learning_rate as number).toExponential(2)
                                                        : String(sweepStatus.best_hyperparams.learning_rate)}
                                                </span>
                                            </div>
                                        )}
                                    </>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default CockpitHUD;
