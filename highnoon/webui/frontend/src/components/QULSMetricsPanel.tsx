// QULSMetricsPanel.tsx - QULS Full Telemetry Metrics Panel
// Phase 200+: HIGHNOON_UPGRADE_ROADMAP.md Phase 4.2 - WebUI Metrics Dashboard
// Displays real-time QULS metrics via WebSocket telemetry

import { useState, useEffect, useRef, useCallback } from 'react';
import {
    Activity,
    Zap,
    AlertTriangle,
    TrendingDown,
    Cpu,
    Target,
    Thermometer,
    Gauge,
    ChevronDown,
    ChevronUp,
    Waves,
    GitBranch
} from 'lucide-react';
import './QULSMetricsPanel.css';

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
}

interface QULSMetricsPanelProps {
    jobId: string | null;
    isRunning: boolean;
}

function formatNumber(num: number, decimals = 4): string {
    if (num === 0) return '0';
    if (Math.abs(num) < 0.0001) return num.toExponential(2);
    return num.toFixed(decimals);
}

function formatMemory(mb: number): string {
    if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
    return `${Math.round(mb)} MB`;
}

export function QULSMetricsPanel({ jobId, isRunning }: QULSMetricsPanelProps) {
    const [expanded, setExpanded] = useState(true);
    const [telemetry, setTelemetry] = useState<QULSTelemetry | null>(null);
    const [connected, setConnected] = useState(false);
    const wsRef = useRef<WebSocket | null>(null);

    // Connect to telemetry WebSocket
    useEffect(() => {
        if (!jobId || !isRunning) {
            setConnected(false);
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/telemetry/${jobId}`;

        try {
            wsRef.current = new WebSocket(wsUrl);

            wsRef.current.onopen = () => {
                setConnected(true);
            };

            wsRef.current.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'full_telemetry') {
                        setTelemetry(data);
                    } else if (data.type === 'finished') {
                        setConnected(false);
                    }
                } catch (err) {
                    console.error('[QULS] Failed to parse telemetry:', err);
                }
            };

            wsRef.current.onerror = () => {
                setConnected(false);
            };

            wsRef.current.onclose = () => {
                setConnected(false);
            };
        } catch (err) {
            console.error('[QULS] WebSocket connection failed:', err);
        }

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, [jobId, isRunning]);

    // Don't render if no job or telemetry
    if (!jobId) {
        return null;
    }

    return (
        <div className={`quls-panel ${expanded ? 'quls-panel--expanded' : ''}`}>
            <button
                className="quls-panel__header"
                onClick={() => setExpanded(!expanded)}
            >
                <div className="quls-panel__title">
                    <Waves size={16} />
                    <span>QULS Metrics</span>
                    <span className={`quls-panel__status ${connected ? 'quls-panel__status--connected' : ''}`}>
                        {connected ? '● Live' : '○ Offline'}
                    </span>
                    {telemetry?.loss.barren_plateau_detected && (
                        <span className="quls-panel__alert">
                            <AlertTriangle size={14} />
                            Barren Plateau
                        </span>
                    )}
                </div>
                {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>

            {expanded && (
                <div className="quls-panel__content">
                    {/* Loss Components */}
                    <div className="quls-panel__section">
                        <h4 className="quls-panel__section-title">
                            <TrendingDown size={14} />
                            Loss Components
                        </h4>
                        <div className="quls-panel__grid">
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Total</span>
                                <span className="quls-panel__metric-value quls-panel__metric-value--primary">
                                    {telemetry ? formatNumber(telemetry.loss.total) : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Fidelity</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? formatNumber(telemetry.loss.fidelity) : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Entropy</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? formatNumber(telemetry.loss.entropy) : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Coherence</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? formatNumber(telemetry.loss.coherence) : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Symplectic</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? formatNumber(telemetry.loss.symplectic) : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Born Rule</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? formatNumber(telemetry.loss.born_rule) : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Holographic</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? formatNumber(telemetry.loss.holographic) : '—'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* HPO State */}
                    <div className="quls-panel__section">
                        <h4 className="quls-panel__section-title">
                            <Target size={14} />
                            HPO State
                        </h4>
                        <div className="quls-panel__grid quls-panel__grid--compact">
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Trial</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? `#${telemetry.hpo.trial_id}` : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Tunneling</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? `${(telemetry.hpo.tunneling_probability * 100).toFixed(1)}%` : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Temperature</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? formatNumber(telemetry.hpo.annealing_temperature, 3) : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Mode</span>
                                <span className={`quls-panel__metric-value quls-panel__mode--${telemetry?.hpo.exploration_mode || 'unknown'}`}>
                                    {telemetry?.hpo.exploration_mode || '—'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Gradients */}
                    <div className="quls-panel__section">
                        <h4 className="quls-panel__section-title">
                            <GitBranch size={14} />
                            Gradients
                        </h4>
                        <div className="quls-panel__grid quls-panel__grid--compact">
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Global Norm</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? formatNumber(telemetry.gradients.global_norm) : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">VQC Variance</span>
                                <span className={`quls-panel__metric-value ${telemetry?.loss.barren_plateau_detected ? 'quls-panel__metric-value--warning' : ''}`}>
                                    {telemetry ? formatNumber(telemetry.gradients.vqc_variance) : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Max Layer</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? formatNumber(telemetry.gradients.max_layer_norm) : '—'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Memory */}
                    <div className="quls-panel__section">
                        <h4 className="quls-panel__section-title">
                            <Cpu size={14} />
                            Memory
                        </h4>
                        <div className="quls-panel__grid quls-panel__grid--compact">
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">RSS</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? formatMemory(telemetry.memory.rss_mb) : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Peak</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? formatMemory(telemetry.memory.peak_mb) : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">GPU</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? formatMemory(telemetry.memory.gpu_allocated_mb) : '—'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Training Stats */}
                    <div className="quls-panel__section">
                        <h4 className="quls-panel__section-title">
                            <Activity size={14} />
                            Training
                        </h4>
                        <div className="quls-panel__grid quls-panel__grid--compact">
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Epoch</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry?.training.epoch ?? '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Step</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? telemetry.training.step.toLocaleString() : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">LR</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? telemetry.training.learning_rate.toExponential(2) : '—'}
                                </span>
                            </div>
                            <div className="quls-panel__metric">
                                <span className="quls-panel__metric-label">Throughput</span>
                                <span className="quls-panel__metric-value">
                                    {telemetry ? `${telemetry.training.throughput_samples_sec.toFixed(1)} s/s` : '—'}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default QULSMetricsPanel;
