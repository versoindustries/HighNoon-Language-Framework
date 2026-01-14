// HighNoon Dashboard - Training Page
// HPO Orchestrator is the source of truth for model size and training configuration
import { useState, useEffect } from 'react';
import { Play, Pause, Square, AlertCircle, Clock, Cpu, Settings, Layers, Target } from 'lucide-react';
import { Card, CardHeader, CardContent, Button, ProgressBar } from '../components/ui';
import { TrainingDiagnostics } from '../components/TrainingDiagnostics';
import { TensorSurfaceViz, useActivationStream } from '../components/visualization';
import { trainingApi, hpoApi, activationApi } from '../api/client';
import type { ActivationSurfaceData } from '../api/client';
import type { TrainingJobInfo, TrainingMetrics, HPOSweepInfo } from '../api/types';
import './Training.css';

// Format parameter count for display
function formatParams(num: number): string {
    if (num >= 1_000_000_000) return `${(num / 1_000_000_000).toFixed(1)}B`;
    if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(0)}M`;
    if (num >= 1_000) return `${(num / 1_000).toFixed(0)}K`;
    return num.toString();
}

// Estimate RAM requirements based on param count (rough heuristic)
function estimateRAM(params: number): string {
    // ~4 bytes per param for FP32, plus overhead
    const gbNeeded = Math.ceil((params * 4) / (1024 * 1024 * 1024) * 2);
    return `~${gbNeeded} GB`;
}

export function Training() {
    // HPO sweeps - source of truth for model configuration
    const [selectedSweep, setSelectedSweep] = useState<HPOSweepInfo | null>(null);
    const [sweeps, setSweeps] = useState<HPOSweepInfo[]>([]);
    const [loadingSweeps, setLoadingSweeps] = useState(true);

    // Training job state
    const [job, setJob] = useState<TrainingJobInfo | null>(null);
    const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
    const [lossHistory, setLossHistory] = useState<{ step: number; loss: number }[]>([]);

    // Activation visualization - use live WebSocket streaming when training is running
    const jobIsActive = job?.state === 'running' || job?.state === 'paused';
    const activationStream = useActivationStream(
        jobIsActive && job ? job.job_id : null,
        { enabled: job?.state === 'running' }
    );

    // Fallback state for manual layer selection
    const [manualActivationData, setManualActivationData] = useState<ActivationSurfaceData | null>(null);
    const [selectedLayer, setSelectedLayer] = useState<string>('embedding');
    const [activationLoading, setActivationLoading] = useState(false);

    // Use stream data when available, fallback to manual fetch
    const activationData = activationStream.data || manualActivationData;
    const isLiveStreaming = activationStream.isConnected && activationStream.data !== null;

    // Load HPO sweeps - both completed and running for visibility
    useEffect(() => {
        async function loadSweeps() {
            try {
                const allSweeps = await hpoApi.listSweeps();
                // Sort by completion status (completed first) then by date
                const sortedSweeps = allSweeps.sort((a, b) => {
                    if (a.state === 'completed' && b.state !== 'completed') return -1;
                    if (b.state === 'completed' && a.state !== 'completed') return 1;
                    return (b.started_at || '').localeCompare(a.started_at || '');
                });
                setSweeps(sortedSweeps);
            } catch (err) {
                console.error('Failed to load HPO sweeps:', err);
            } finally {
                setLoadingSweeps(false);
            }
        }
        loadSweeps();
    }, []);

    // Poll for metrics when job is running
    useEffect(() => {
        if (!job || (job.state !== 'running' && job.state !== 'paused')) {
            return;
        }

        const pollMetrics = async () => {
            try {
                const metricsData = await trainingApi.getMetrics(job.job_id);
                setMetrics(metricsData);

                // Update loss history for chart
                if (metricsData.loss !== undefined && metricsData.global_step !== undefined) {
                    setLossHistory(prev => {
                        // Avoid duplicates
                        if (prev.length > 0 && prev[prev.length - 1].step === metricsData.global_step) {
                            return prev;
                        }
                        return [...prev, { step: metricsData.global_step, loss: metricsData.loss }].slice(-100);
                    });
                }

                // Also refresh job status
                const jobStatus = await trainingApi.getStatus(job.job_id);
                setJob(jobStatus);
            } catch (err) {
                console.error('Failed to fetch metrics:', err);
            }
        };

        // Initial fetch
        pollMetrics();

        // Poll every 2 seconds when running
        const interval = setInterval(pollMetrics, 2000);
        return () => clearInterval(interval);
    }, [job?.job_id, job?.state]);

    const isRunning = job?.state === 'running';
    const isPaused = job?.state === 'paused';
    const completedSweeps = sweeps.filter(s => s.state === 'completed');

    const handleStartTraining = async () => {
        if (!selectedSweep) {
            console.error('No HPO sweep selected');
            return;
        }

        if (selectedSweep.state !== 'completed') {
            console.error('Cannot train with incomplete sweep');
            return;
        }

        try {
            const response = await trainingApi.start({
                mode: 'auto_tune',
                sweep_id: selectedSweep.sweep_id,
            });
            setJob(response);
            console.log('Started training with HPO config:', response);
        } catch (err) {
            console.error('Failed to start training:', err);
        }
    };

    const handlePause = async () => {
        if (!job) return;
        try {
            await trainingApi.pause(job.job_id);
            setJob({ ...job, state: 'paused' });
        } catch (err) {
            console.error('Failed to pause:', err);
        }
    };

    const handleResume = async () => {
        if (!job) return;
        try {
            await trainingApi.resume(job.job_id);
            setJob({ ...job, state: 'running' });
        } catch (err) {
            console.error('Failed to resume:', err);
        }
    };

    const handleStop = async () => {
        if (!job) return;
        try {
            await trainingApi.cancel(job.job_id);
            setJob({ ...job, state: 'cancelled' });
        } catch (err) {
            console.error('Failed to stop:', err);
        }
    };

    const canStartTraining = !!selectedSweep && selectedSweep.state === 'completed' && !isRunning;

    return (
        <div className="page">
            <div className="page-header">
                <div className="page-header-content">
                    <h1 className="page-title">Training</h1>
                    <p className="page-subtitle">
                        Train using optimized hyperparameters from HPO Orchestrator
                    </p>
                </div>
                <div className="page-header-actions">
                    {isRunning ? (
                        <>
                            <Button
                                variant="secondary"
                                leftIcon={isPaused ? <Play size={16} /> : <Pause size={16} />}
                                onClick={isPaused ? handleResume : handlePause}
                            >
                                {isPaused ? 'Resume' : 'Pause'}
                            </Button>
                            <Button variant="danger" leftIcon={<Square size={16} />} onClick={handleStop}>
                                Stop
                            </Button>
                        </>
                    ) : (
                        <Button
                            variant="primary"
                            leftIcon={<Play size={16} />}
                            onClick={handleStartTraining}
                            disabled={!canStartTraining}
                        >
                            Start Training
                        </Button>
                    )}
                </div>
            </div>

            <div className="training-layout">
                {/* Status Panel */}
                <Card variant="glass" padding="lg" className="training-status-card">
                    <div className="training-status">
                        <div className="status-header">
                            <div className="status-indicator">
                                <span className={`status-dot status-${job?.state || 'idle'}`}></span>
                                <span className="status-text">{job?.state || 'Ready'}</span>
                            </div>
                            {job && (
                                <span className="job-id">Job: {job.job_id}</span>
                            )}
                        </div>

                        <div className="training-metrics-grid">
                            <div className="training-metric">
                                <span className="metric-label">Step</span>
                                <span className="metric-value">{metrics?.global_step ?? 0}</span>
                            </div>
                            <div className="training-metric">
                                <span className="metric-label">Epoch</span>
                                <span className="metric-value">{metrics?.current_epoch ?? 0}</span>
                            </div>
                            <div className="training-metric">
                                <span className="metric-label">Loss</span>
                                <span className="metric-value">{metrics?.loss?.toFixed(4) ?? '—'}</span>
                            </div>
                            <div className="training-metric">
                                <span className="metric-label">Learning Rate</span>
                                <span className="metric-value">{metrics?.learning_rate?.toExponential(2) ?? '—'}</span>
                            </div>
                            <div className="training-metric">
                                <span className="metric-label">Throughput</span>
                                <span className="metric-value">{metrics?.throughput ? `${metrics.throughput.toFixed(0)} tok/s` : '—'}</span>
                            </div>
                        </div>

                        <ProgressBar
                            value={metrics?.progress_percent ?? 0}
                            max={100}
                            variant="gradient"
                            size="lg"
                            showLabel
                            label="Training Progress"
                            animated={isRunning}
                        />
                    </div>
                </Card>

                {/* HPO Configuration Selection - Source of Truth */}
                <Card padding="lg">
                    <CardHeader
                        title="Select HPO Configuration"
                        subtitle="HPO Orchestrator is the source of truth for model size and training parameters"
                    />
                    <CardContent>
                        {loadingSweeps ? (
                            <div className="loading-placeholder">Loading HPO configurations...</div>
                        ) : selectedSweep ? (
                            <div className="selected-sweep">
                                {/* Model Size - Primary Display */}
                                <div className="sweep-model-size">
                                    <Cpu size={24} className="model-icon" />
                                    <div className="model-info">
                                        <span className="model-params">
                                            {selectedSweep.model_config?.param_budget
                                                ? formatParams(selectedSweep.model_config.param_budget)
                                                : 'Unknown'} Parameters
                                        </span>
                                        <span className="model-ram">
                                            {selectedSweep.model_config?.param_budget
                                                ? estimateRAM(selectedSweep.model_config.param_budget)
                                                : '—'} RAM
                                        </span>
                                    </div>
                                </div>

                                <div className="sweep-header">
                                    <div className="sweep-meta">
                                        <span className="sweep-id">{selectedSweep.sweep_id}</span>
                                        <span className={`sweep-state sweep-state-${selectedSweep.state}`}>
                                            {selectedSweep.state}
                                        </span>
                                    </div>
                                    <div className="sweep-stats">
                                        <span className="sweep-trials">
                                            <Target size={14} />
                                            {selectedSweep.completed_trials} trials
                                        </span>
                                        <span className="sweep-loss">
                                            Best: {selectedSweep.best_loss?.toFixed(4) ?? '—'}
                                        </span>
                                    </div>
                                </div>

                                {/* Architecture Details */}
                                {selectedSweep.model_config && (
                                    <div className="sweep-architecture">
                                        <div className="arch-item">
                                            <Layers size={14} />
                                            <span>{selectedSweep.model_config.num_reasoning_blocks} Reasoning Blocks</span>
                                        </div>
                                        <div className="arch-item">
                                            <Settings size={14} />
                                            <span>{selectedSweep.model_config.num_moe_experts} MoE Experts</span>
                                        </div>
                                        <div className="arch-item">
                                            <span>Embed: {selectedSweep.model_config.embedding_dim}</span>
                                        </div>
                                        <div className="arch-item">
                                            <span>Context: {formatParams(selectedSweep.model_config.context_window)}</span>
                                        </div>
                                    </div>
                                )}

                                {/* Best Hyperparameters */}
                                {selectedSweep.best_hyperparams && (
                                    <div className="sweep-params">
                                        <h4>Optimized Hyperparameters</h4>
                                        <div className="params-grid">
                                            {Object.entries(selectedSweep.best_hyperparams).slice(0, 6).map(([key, value]) => (
                                                <div key={key} className="param-item">
                                                    <span className="param-key">{key}:</span>
                                                    <span className="param-value">{String(value)}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => setSelectedSweep(null)}
                                >
                                    Select Different Configuration
                                </Button>
                            </div>
                        ) : completedSweeps.length === 0 ? (
                            <div className="no-sweep">
                                <AlertCircle size={32} />
                                <h3>No HPO Configurations Available</h3>
                                <p>
                                    Run an HPO sweep first to optimize hyperparameters for your model.
                                    The HPO Orchestrator will find the best configuration for your target model size.
                                </p>
                                <Button variant="primary" onClick={() => window.location.href = '/hpo'}>
                                    Go to HPO Orchestrator
                                </Button>
                            </div>
                        ) : (
                            <div className="sweep-grid">
                                {completedSweeps.map(sweep => (
                                    <button
                                        key={sweep.sweep_id}
                                        className="sweep-card"
                                        onClick={() => setSelectedSweep(sweep)}
                                    >
                                        {/* Model Size as Primary Info */}
                                        <div className="sweep-card-size">
                                            <Cpu size={20} />
                                            <span className="size-value">
                                                {sweep.model_config?.param_budget
                                                    ? formatParams(sweep.model_config.param_budget)
                                                    : '—'}
                                            </span>
                                        </div>
                                        <div className="sweep-card-details">
                                            <span className="sweep-id">{sweep.sweep_id}</span>
                                            <span className="sweep-trials">{sweep.completed_trials} trials</span>
                                        </div>
                                        <div className="sweep-card-result">
                                            <span className="sweep-loss">
                                                Loss: {sweep.best_loss?.toFixed(4) ?? '—'}
                                            </span>
                                            {sweep.model_config && (
                                                <span className="sweep-arch">
                                                    {sweep.model_config.num_reasoning_blocks}L / {sweep.model_config.num_moe_experts}E
                                                </span>
                                            )}
                                        </div>
                                    </button>
                                ))}
                            </div>
                        )}
                    </CardContent>
                </Card>

                {/* Loss Chart */}
                <Card padding="lg" className="loss-chart-card">
                    <CardHeader title="Loss Curve" />
                    <CardContent>
                        <div className="chart-container">
                            {lossHistory.length > 0 ? (
                                <div className="chart-placeholder">
                                    <p>Loss chart with {lossHistory.length} points</p>
                                </div>
                            ) : (
                                <div className="empty-state-sm">
                                    <Clock size={24} />
                                    <p>Loss curve will appear during training</p>
                                </div>
                            )}
                        </div>
                    </CardContent>
                </Card>

                {/* Tensor Activations Visualization */}
                <Card padding="lg" className="activation-viz-card">
                    <CardHeader title="Tensor Activations">
                        <select
                            value={selectedLayer}
                            onChange={async (e) => {
                                setSelectedLayer(e.target.value);
                                setActivationLoading(true);
                                try {
                                    const data = await activationApi.getSurface(
                                        e.target.value,
                                        job?.job_id || 'demo'
                                    );
                                    setManualActivationData(data);
                                } finally {
                                    setActivationLoading(false);
                                }
                            }}
                            className="layer-select-sm"
                        >
                            {['embedding', 'block_0', 'block_1', 'attention_0', 'moe_gate', 'output_proj'].map(layer => (
                                <option key={layer} value={layer}>{layer}</option>
                            ))}
                        </select>
                    </CardHeader>
                    <CardContent>
                        <TensorSurfaceViz
                            data={activationData}
                            loading={activationLoading}
                            autoRotate={job?.state === 'running'}
                            height={250}
                        />
                    </CardContent>
                </Card>

                {/* Training Diagnostics */}
                <TrainingDiagnostics jobId={job?.job_id} />
            </div>
        </div>
    );
}
