// TrainingDiagnostics - Real-time training monitoring
// System health, live logs, and gradient visualization

import { useState, useEffect, useRef } from 'react';
import {
    Activity, Cpu, HardDrive, Thermometer, Terminal,
    RefreshCw, Pause, Play, Download, ChevronDown, ChevronUp
} from 'lucide-react';
import { Button, Card, CardHeader, CardContent, ProgressBar } from '../ui';
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

    // Simulate system health updates
    useEffect(() => {
        // Simulated health data (would come from backend in production)
        const interval = setInterval(() => {
            setHealth({
                cpuPercent: 20 + Math.random() * 40,
                memoryUsedGb: 8 + Math.random() * 16,
                memoryTotalGb: 64,
                diskUsedGb: 120 + Math.random() * 10,
                diskTotalGb: 500,
            });
        }, 2000);

        return () => clearInterval(interval);
    }, []);

    // Simulate logs (would come from WebSocket in production)
    useEffect(() => {
        if (!jobId) return;

        const messages = [
            'Loading training configuration...',
            'Initializing model weights...',
            'Starting training loop...',
            'Step 100: loss=2.4532, lr=1e-4',
            'Step 200: loss=2.1245, lr=1e-4',
            'Checkpoint saved at step 200',
        ];

        let index = 0;
        const interval = setInterval(() => {
            if (index < messages.length) {
                setLogs(prev => [...prev, {
                    timestamp: new Date().toLocaleTimeString(),
                    level: 'info',
                    message: messages[index],
                }]);
                index++;
            }
        }, 1500);

        return () => clearInterval(interval);
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
