// TrainingConsole.tsx - Real-time training log display for HPO
// Displays live loss metrics, gradient norms, and training progress

import { useState, useEffect, useRef, useCallback } from 'react';
import { Terminal, Maximize2, Minimize2, Filter, Trash2, Bug } from 'lucide-react';
import { Button, Card, CardHeader, CardContent } from './ui';
import './TrainingConsole.css';

interface LogEntry {
    timestamp: string;
    level: 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG';
    message: string;
    step?: number;
    loss?: number;
    gradient_norm?: number;
    learning_rate?: number;
    epoch?: number;
    trial_id?: string;
    // Memory metrics
    memory_mb?: number;
    peak_memory_mb?: number;
    // Quality metrics (on trial completion)
    perplexity?: number;
    mean_confidence?: number;
    composite_score?: number;
    efficiency_score?: number;
    param_count?: number;
    // Architecture info (on trial start)
    hidden_dim?: number;
    num_blocks?: number;
    num_experts?: number;
    batch_size?: number;
    optimizer?: string;
    vocab_size?: number;
    sequence_length?: number;
    // Training state indicators
    phase?: string;
    barren_plateau_detected?: boolean;
    crystallization_active?: boolean;
}

interface TrainingConsoleProps {
    sweepId: string | null;
    isRunning: boolean;
    devMode?: boolean;
}

const LOG_LEVEL_COLORS: Record<string, string> = {
    INFO: 'var(--color-text-secondary)',
    WARNING: '#f59e0b',
    ERROR: '#ef4444',
    DEBUG: '#8b5cf6',
};

// Safe accessor for log level color
function getLogLevelColor(level: string | undefined): string {
    if (!level) return 'var(--color-text-secondary)';
    return LOG_LEVEL_COLORS[level] || 'var(--color-text-secondary)';
}

export function TrainingConsole({ sweepId, isRunning, devMode = false }: TrainingConsoleProps) {
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [isExpanded, setIsExpanded] = useState(false);
    const [filter, setFilter] = useState<'all' | 'metrics' | 'errors'>('all');
    const [showDebug, setShowDebug] = useState(devMode);
    const [currentLoss, setCurrentLoss] = useState<number | null>(null);
    const [currentStep, setCurrentStep] = useState<number | null>(null);
    const [currentPerplexity, setCurrentPerplexity] = useState<number | null>(null);
    const [currentConfidence, setCurrentConfidence] = useState<number | null>(null);
    const [currentMemoryMb, setCurrentMemoryMb] = useState<number | null>(null);
    const [currentGradNorm, setCurrentGradNorm] = useState<number | null>(null);
    const [currentLearningRate, setCurrentLearningRate] = useState<number | null>(null);
    const [currentEpoch, setCurrentEpoch] = useState<number | null>(null);
    const [logIndex, setLogIndex] = useState(0);
    const consoleRef = useRef<HTMLDivElement>(null);
    const pollIntervalRef = useRef<number | null>(null);

    // Auto-scroll to bottom
    const scrollToBottom = useCallback(() => {
        if (consoleRef.current) {
            consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
        }
    }, []);

    // Poll for new logs
    useEffect(() => {
        if (!sweepId || !isRunning) {
            if (pollIntervalRef.current) {
                window.clearInterval(pollIntervalRef.current);
                pollIntervalRef.current = null;
            }
            return;
        }

        const fetchLogs = async () => {
            try {
                const response = await fetch(
                    `/api/hpo/sweep/${sweepId}/logs?since_index=${logIndex}&limit=50`
                );
                if (response.ok) {
                    const data = await response.json();
                    if (data.logs && data.logs.length > 0) {
                        setLogs((prev) => [...prev, ...data.logs]);
                        setLogIndex(data.next_index);

                        // Update current metrics from latest logs
                        const latestWithLoss = [...data.logs].reverse().find((l: LogEntry) => l.loss !== null && l.loss !== undefined);
                        if (latestWithLoss) {
                            setCurrentLoss(latestWithLoss.loss);
                            setCurrentStep(latestWithLoss.step);
                        }
                        // Update memory from latest log with memory_mb
                        const latestWithMemory = [...data.logs].reverse().find((l: LogEntry) => l.memory_mb !== null && l.memory_mb !== undefined);
                        if (latestWithMemory) {
                            setCurrentMemoryMb(latestWithMemory.memory_mb);
                        }
                        // Update perplexity and confidence from trial completion logs
                        const latestWithPerplexity = [...data.logs].reverse().find((l: LogEntry) => l.perplexity !== null && l.perplexity !== undefined);
                        if (latestWithPerplexity) {
                            setCurrentPerplexity(latestWithPerplexity.perplexity);
                        }
                        const latestWithConfidence = [...data.logs].reverse().find((l: LogEntry) => l.mean_confidence !== null && l.mean_confidence !== undefined);
                        if (latestWithConfidence) {
                            setCurrentConfidence(latestWithConfidence.mean_confidence);
                        }
                        // Update gradient norm from latest log
                        const latestWithGradNorm = [...data.logs].reverse().find((l: LogEntry) => l.gradient_norm !== null && l.gradient_norm !== undefined);
                        if (latestWithGradNorm) {
                            setCurrentGradNorm(latestWithGradNorm.gradient_norm);
                        }
                        // Update learning rate from latest log
                        const latestWithLR = [...data.logs].reverse().find((l: LogEntry) => l.learning_rate !== null && l.learning_rate !== undefined);
                        if (latestWithLR) {
                            setCurrentLearningRate(latestWithLR.learning_rate);
                        }
                        // Update epoch from latest log
                        const latestWithEpoch = [...data.logs].reverse().find((l: LogEntry) => l.epoch !== null && l.epoch !== undefined);
                        if (latestWithEpoch) {
                            setCurrentEpoch(latestWithEpoch.epoch);
                        }

                        scrollToBottom();
                    }
                }
            } catch (err) {
                console.error('Failed to fetch logs:', err);
            }
        };

        // Initial fetch
        fetchLogs();

        // Poll every second
        pollIntervalRef.current = window.setInterval(fetchLogs, 1000);

        return () => {
            if (pollIntervalRef.current) {
                window.clearInterval(pollIntervalRef.current);
            }
        };
    }, [sweepId, isRunning, logIndex, scrollToBottom]);

    // Reset when sweep changes
    useEffect(() => {
        setLogs([]);
        setLogIndex(0);
        setCurrentLoss(null);
        setCurrentStep(null);
        setCurrentPerplexity(null);
        setCurrentConfidence(null);
        setCurrentMemoryMb(null);
        setCurrentGradNorm(null);
        setCurrentLearningRate(null);
        setCurrentEpoch(null);
    }, [sweepId]);

    // Filter logs - with null safety
    const filteredLogs = logs.filter((log) => {
        if (!log) return false;
        const level = log.level || 'INFO';
        if (!showDebug && level === 'DEBUG') return false;
        if (filter === 'metrics') return log.loss !== undefined || log.step !== undefined;
        if (filter === 'errors') return level === 'ERROR' || level === 'WARNING';
        return true;
    });

    const handleClear = async () => {
        if (sweepId) {
            try {
                await fetch(`/api/hpo/sweep/${sweepId}/logs`, { method: 'DELETE' });
                setLogs([]);
                setLogIndex(0);
            } catch (err) {
                console.error('Failed to clear logs:', err);
            }
        }
    };

    const formatTimestamp = (ts: string) => {
        try {
            return new Date(ts).toLocaleTimeString('en-US', {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
            });
        } catch {
            return ts;
        }
    };

    const formatLoss = (loss: number | undefined) => {
        if (loss === undefined) return '';
        return loss.toFixed(6);
    };

    return (
        <Card
            variant="glass"
            padding="none"
            className={`training-console ${isExpanded ? 'training-console--expanded' : ''}`}
        >
            <CardHeader
                title="Training Console"
                subtitle={isRunning ? 'Live' : sweepId ? 'Stopped' : 'No active sweep'}
            />

            {/* Live Metrics Header - shows real-time training state */}
            <div className="training-console__header">
                <Terminal size={16} />
                {/* Primary metrics row */}
                <div className="training-console__metrics-row">
                    {currentEpoch !== null && (
                        <span className="training-console__metric training-console__metric--epoch">
                            Epoch: <strong>{currentEpoch}</strong>
                        </span>
                    )}
                    {currentStep !== null && (
                        <span className="training-console__metric">
                            Step: <strong>{currentStep}</strong>
                        </span>
                    )}
                    {currentLoss !== null && (
                        <span className="training-console__metric training-console__metric--primary">
                            Loss: <strong>{formatLoss(currentLoss)}</strong>
                        </span>
                    )}
                    {currentPerplexity !== null && (
                        <span className="training-console__metric">
                            PPL: <strong>{currentPerplexity.toFixed(2)}</strong>
                        </span>
                    )}
                </div>
                {/* Secondary metrics row */}
                <div className="training-console__metrics-row training-console__metrics-row--secondary">
                    {currentGradNorm !== null && (
                        <span className="training-console__metric training-console__metric--grad">
                            Grad: <strong>{currentGradNorm.toFixed(4)}</strong>
                        </span>
                    )}
                    {currentLearningRate !== null && (
                        <span className="training-console__metric training-console__metric--lr">
                            LR: <strong>{currentLearningRate.toExponential(2)}</strong>
                        </span>
                    )}
                    {currentConfidence !== null && (
                        <span className="training-console__metric">
                            Conf: <strong>{(currentConfidence * 100).toFixed(1)}%</strong>
                        </span>
                    )}
                    {currentMemoryMb !== null && (
                        <span className="training-console__metric training-console__metric--mem">
                            Mem: <strong>{currentMemoryMb.toFixed(0)}MB</strong>
                        </span>
                    )}
                </div>
                {devMode && (
                    <span className="training-console__dev-badge">DEV</span>
                )}
            </div>

            <div className="training-console__toolbar">
                <div className="training-console__filters">
                    <button
                        className={`training-console__filter-btn ${filter === 'all' ? 'active' : ''}`}
                        onClick={() => setFilter('all')}
                    >
                        All
                    </button>
                    <button
                        className={`training-console__filter-btn ${filter === 'metrics' ? 'active' : ''}`}
                        onClick={() => setFilter('metrics')}
                    >
                        Metrics
                    </button>
                    <button
                        className={`training-console__filter-btn ${filter === 'errors' ? 'active' : ''}`}
                        onClick={() => setFilter('errors')}
                    >
                        Errors
                    </button>
                </div>

                <div className="training-console__actions">
                    {devMode && (
                        <button
                            className={`training-console__action-btn ${showDebug ? 'active' : ''}`}
                            onClick={() => setShowDebug(!showDebug)}
                            title="Toggle debug logs"
                        >
                            <Bug size={14} />
                        </button>
                    )}
                    <button
                        className="training-console__action-btn"
                        onClick={handleClear}
                        title="Clear logs"
                    >
                        <Trash2 size={14} />
                    </button>
                    <button
                        className="training-console__action-btn"
                        onClick={() => setIsExpanded(!isExpanded)}
                        title={isExpanded ? 'Minimize' : 'Maximize'}
                    >
                        {isExpanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
                    </button>
                </div>
            </div>

            <CardContent>
                <div className="training-console__logs" ref={consoleRef}>
                    {filteredLogs.length === 0 ? (
                        <div className="training-console__empty">
                            {isRunning ? 'Waiting for logs...' : 'No logs to display'}
                        </div>
                    ) : (
                        filteredLogs.map((log, idx) => {
                            const level = log.level || 'INFO';
                            const trialIdDisplay = log.trial_id && typeof log.trial_id === 'string'
                                ? log.trial_id.slice(0, 8)
                                : null;
                            return (
                                <div
                                    key={idx}
                                    className={`training-console__entry training-console__entry--${level.toLowerCase()}`}
                                >
                                    <span className="training-console__timestamp">
                                        {formatTimestamp(log.timestamp || '')}
                                    </span>
                                    <span
                                        className="training-console__level"
                                        style={{ color: getLogLevelColor(level) }}
                                    >
                                        [{level}]
                                    </span>
                                    {trialIdDisplay && (
                                        <span className="training-console__trial">
                                            [Trial {trialIdDisplay}]
                                        </span>
                                    )}
                                    {log.epoch !== undefined && log.epoch !== null && (
                                        <span className="training-console__epoch">
                                            [E{log.epoch}]
                                        </span>
                                    )}
                                    {log.step !== undefined && log.step !== null && (
                                        <span className="training-console__step">
                                            step={log.step}
                                        </span>
                                    )}
                                    {log.loss !== undefined && log.loss !== null && (
                                        <span className="training-console__loss">
                                            loss={formatLoss(log.loss)}
                                        </span>
                                    )}
                                    {log.perplexity !== undefined && log.perplexity !== null && (
                                        <span className="training-console__perplexity">
                                            ppl={log.perplexity.toFixed(2)}
                                        </span>
                                    )}
                                    {log.gradient_norm !== undefined && log.gradient_norm !== null && (
                                        <span className="training-console__grad-norm">
                                            grad={log.gradient_norm.toFixed(4)}
                                        </span>
                                    )}
                                    {log.learning_rate !== undefined && log.learning_rate !== null && (
                                        <span className="training-console__lr">
                                            lr={log.learning_rate.toExponential(2)}
                                        </span>
                                    )}
                                    {log.mean_confidence !== undefined && log.mean_confidence !== null && (
                                        <span className="training-console__confidence">
                                            conf={(log.mean_confidence * 100).toFixed(1)}%
                                        </span>
                                    )}
                                    {log.memory_mb !== undefined && log.memory_mb !== null && (
                                        <span className="training-console__memory">
                                            mem={log.memory_mb.toFixed(0)}MB
                                        </span>
                                    )}
                                    <span className="training-console__message">
                                        {log.message || ''}
                                    </span>
                                </div>
                            );
                        })
                    )}
                </div>
            </CardContent>

            {/* Status bar */}
            <div className="training-console__status-bar">
                <div className="training-console__status-indicator">
                    <span className={`training-console__status-dot ${isRunning ? 'training-console__status-dot--live' : ''}`} />
                    <span>{isRunning ? 'STREAMING' : sweepId ? 'IDLE' : 'NO SWEEP'}</span>
                </div>
                <span className="training-console__log-count">
                    {filteredLogs.length} entries
                </span>
            </div>
        </Card>
    );
}

export default TrainingConsole;
