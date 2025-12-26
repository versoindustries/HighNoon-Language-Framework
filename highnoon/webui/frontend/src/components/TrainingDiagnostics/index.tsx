// TrainingDiagnostics - Real-time training monitoring
// System health, live logs, and gradient visualization

import { useState, useEffect, useRef } from 'react';
import {
    Activity, Cpu, HardDrive, Thermometer, Terminal,
    RefreshCw, Pause, Play, Download, ChevronDown, ChevronUp
} from 'lucide-react';
import { Button, Card, CardHeader, CardContent, ProgressBar } from '../ui';
import { trainingApi } from '../../api/client';
import './TrainingDiagnostics.css';

// =============================================================================
// TYPES
// =============================================================================

interface SystemHealth {
    cpuPercent: number;
    memoryUsedGb: number;
    memoryTotalGb: number;
    diskUsedGb: number;
    diskTotalGb: number;
    gpuTemp?: number;
    gpuUtilization?: number;
}

interface LogEntry {
    timestamp: string;
    level: 'info' | 'warning' | 'error' | 'debug';
    message: string;
}

interface TrainingMetrics {
    step: number;
    epoch: number;
    loss: number;
    learningRate: number;
    throughput: number;
    gradientNorm?: number;
}

// =============================================================================
// SYSTEM HEALTH PANEL
// =============================================================================

interface SystemHealthPanelProps {
    health: SystemHealth | null;
}

function SystemHealthPanel({ health }: SystemHealthPanelProps) {
    if (!health) {
        return (
            <div className="health-panel health-loading">
                <RefreshCw size={20} className="spin" />
                <span>Loading system health...</span>
            </div>
        );
    }

    const memoryPercent = (health.memoryUsedGb / health.memoryTotalGb) * 100;
    const diskPercent = (health.diskUsedGb / health.diskTotalGb) * 100;

    return (
        <div className="health-panel">
            <div className="health-metric">
                <div className="health-metric-header">
                    <Cpu size={16} />
                    <span>CPU</span>
                    <span className="health-value">{health.cpuPercent.toFixed(0)}%</span>
                </div>
                <ProgressBar
                    value={health.cpuPercent}
                    max={100}
                    size="sm"
                    variant={health.cpuPercent > 90 ? 'danger' : health.cpuPercent > 70 ? 'warning' : 'success'}
                />
            </div>

            <div className="health-metric">
                <div className="health-metric-header">
                    <HardDrive size={16} />
                    <span>Memory</span>
                    <span className="health-value">
                        {health.memoryUsedGb.toFixed(1)} / {health.memoryTotalGb.toFixed(0)} GB
                    </span>
                </div>
                <ProgressBar
                    value={memoryPercent}
                    max={100}
                    size="sm"
                    variant={memoryPercent > 90 ? 'danger' : memoryPercent > 70 ? 'warning' : 'success'}
                />
            </div>

            <div className="health-metric">
                <div className="health-metric-header">
                    <Activity size={16} />
                    <span>Disk</span>
                    <span className="health-value">
                        {health.diskUsedGb.toFixed(0)} / {health.diskTotalGb.toFixed(0)} GB
                    </span>
                </div>
                <ProgressBar
                    value={diskPercent}
                    max={100}
                    size="sm"
                    variant={diskPercent > 90 ? 'danger' : diskPercent > 70 ? 'warning' : 'success'}
                />
            </div>

            {health.gpuTemp !== undefined && (
                <div className="health-metric">
                    <div className="health-metric-header">
                        <Thermometer size={16} />
                        <span>GPU Temp</span>
                        <span className="health-value">{health.gpuTemp}°C</span>
                    </div>
                    <ProgressBar
                        value={health.gpuTemp}
                        max={100}
                        size="sm"
                        variant={health.gpuTemp > 80 ? 'danger' : health.gpuTemp > 70 ? 'warning' : 'success'}
                    />
                </div>
            )}
        </div>
    );
}

// =============================================================================
// LOG VIEWER
// =============================================================================

interface LogViewerProps {
    logs: LogEntry[];
    maxHeight?: number;
}

function LogViewer({ logs, maxHeight = 300 }: LogViewerProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [autoScroll, setAutoScroll] = useState(true);

    useEffect(() => {
        if (autoScroll && containerRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
    }, [logs, autoScroll]);

    const levelColors = {
        info: 'var(--color-info)',
        warning: 'var(--color-warning)',
        error: 'var(--color-danger)',
        debug: 'var(--text-muted)',
    };

    return (
        <div className="log-viewer">
            <div className="log-viewer-header">
                <Terminal size={14} />
                <span>Training Logs</span>
                <div className="log-controls">
                    <button
                        className={`log-autoscroll ${autoScroll ? 'active' : ''}`}
                        onClick={() => setAutoScroll(!autoScroll)}
                        title={autoScroll ? 'Auto-scroll enabled' : 'Auto-scroll disabled'}
                    >
                        {autoScroll ? <ChevronDown size={14} /> : <Pause size={14} />}
                    </button>
                </div>
            </div>
            <div
                className="log-content"
                ref={containerRef}
                style={{ maxHeight }}
            >
                {logs.length === 0 ? (
                    <div className="log-empty">Waiting for logs...</div>
                ) : (
                    logs.map((log, i) => (
                        <div key={i} className="log-entry">
                            <span className="log-timestamp">{log.timestamp}</span>
                            <span
                                className="log-level"
                                style={{ color: levelColors[log.level] }}
                            >
                                [{log.level.toUpperCase()}]
                            </span>
                            <span className="log-message">{log.message}</span>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}

// =============================================================================
// METRICS DISPLAY
// =============================================================================

interface MetricsDisplayProps {
    metrics: TrainingMetrics | null;
}

function MetricsDisplay({ metrics }: MetricsDisplayProps) {
    if (!metrics) {
        return (
            <div className="metrics-empty">
                <span>No training in progress</span>
            </div>
        );
    }

    return (
        <div className="metrics-grid">
            <div className="metric-item">
                <span className="metric-label">Step</span>
                <span className="metric-value">{metrics.step.toLocaleString()}</span>
            </div>
            <div className="metric-item">
                <span className="metric-label">Epoch</span>
                <span className="metric-value">{metrics.epoch}</span>
            </div>
            <div className="metric-item">
                <span className="metric-label">Loss</span>
                <span className="metric-value loss">{metrics.loss.toFixed(4)}</span>
            </div>
            <div className="metric-item">
                <span className="metric-label">Learning Rate</span>
                <span className="metric-value">{metrics.learningRate.toExponential(2)}</span>
            </div>
            <div className="metric-item">
                <span className="metric-label">Throughput</span>
                <span className="metric-value">{metrics.throughput.toFixed(1)} tok/s</span>
            </div>
            {metrics.gradientNorm !== undefined && (
                <div className="metric-item">
                    <span className="metric-label">Gradient Norm</span>
                    <span className="metric-value">{metrics.gradientNorm.toFixed(4)}</span>
                </div>
            )}
        </div>
    );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

interface TrainingDiagnosticsProps {
    jobId?: string;
    isCollapsed?: boolean;
    onToggleCollapse?: () => void;
}

export function TrainingDiagnostics({
    jobId,
    isCollapsed = false,
    onToggleCollapse
}: TrainingDiagnosticsProps) {
    const [health, setHealth] = useState<SystemHealth | null>(null);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
    const [logIndex, setLogIndex] = useState(0);

    // Fetch system health from backend
    useEffect(() => {
        if (!jobId) return;

        const fetchHealth = async () => {
            try {
                const data = await trainingApi.getHealth(jobId);
                setHealth({
                    cpuPercent: data.cpu_percent,
                    memoryUsedGb: data.memory_used_gb,
                    memoryTotalGb: data.memory_total_gb,
                    diskUsedGb: data.disk_used_gb,
                    diskTotalGb: data.disk_total_gb,
                });
            } catch (err) {
                console.error('Failed to fetch health:', err);
            }
        };

        fetchHealth();
        const interval = setInterval(fetchHealth, 3000);
        return () => clearInterval(interval);
    }, [jobId]);

    // Fetch logs and metrics from backend
    useEffect(() => {
        if (!jobId) return;

        const fetchLogs = async () => {
            try {
                const data = await trainingApi.getLogs(jobId, logIndex, 50);
                if (data.logs && data.logs.length > 0) {
                    const newLogs: LogEntry[] = data.logs.map(log => ({
                        timestamp: new Date(log.timestamp).toLocaleTimeString(),
                        level: (log.level?.toLowerCase() || 'info') as 'info' | 'warning' | 'error' | 'debug',
                        message: log.message || '',
                    }));
                    setLogs(prev => [...prev, ...newLogs]);
                    setLogIndex(data.next_index);

                    // Update metrics from latest log with training data
                    const latestWithMetrics = [...data.logs].reverse().find(l => l.step !== undefined);
                    if (latestWithMetrics) {
                        setMetrics({
                            step: latestWithMetrics.step || 0,
                            epoch: 0, // Would need to be passed from job
                            loss: latestWithMetrics.loss || 0,
                            learningRate: latestWithMetrics.learning_rate || 0,
                            throughput: latestWithMetrics.throughput || 0,
                            gradientNorm: latestWithMetrics.gradient_norm,
                        });
                    }
                }
            } catch (err) {
                console.error('Failed to fetch logs:', err);
            }
        };

        fetchLogs();
        const interval = setInterval(fetchLogs, 2000);
        return () => clearInterval(interval);
    }, [jobId, logIndex]);

    // Reset when job changes
    useEffect(() => {
        setLogs([]);
        setLogIndex(0);
        setMetrics(null);
    }, [jobId]);

    if (isCollapsed) {
        return (
            <div className="diagnostics-collapsed">
                <button className="diagnostics-expand" onClick={onToggleCollapse}>
                    <Activity size={14} />
                    <span>Diagnostics</span>
                    <ChevronUp size={14} />
                </button>
            </div>
        );
    }

    return (
        <div className="training-diagnostics">
            <div className="diagnostics-header">
                <div className="diagnostics-title">
                    <Activity size={16} />
                    <span>Training Diagnostics</span>
                </div>
                {onToggleCollapse && (
                    <button className="diagnostics-collapse" onClick={onToggleCollapse}>
                        <ChevronDown size={14} />
                    </button>
                )}
            </div>

            <div className="diagnostics-content">
                <div className="diagnostics-column">
                    <div className="diagnostics-section">
                        <h4>System Health</h4>
                        <SystemHealthPanel health={health} />
                    </div>
                </div>

                <div className="diagnostics-column diagnostics-column-wide">
                    <div className="diagnostics-section">
                        <h4>Live Metrics</h4>
                        <MetricsDisplay metrics={metrics} />
                    </div>

                    <div className="diagnostics-section">
                        <LogViewer logs={logs} />
                    </div>
                </div>
            </div>
        </div>
    );
}

export default TrainingDiagnostics;
