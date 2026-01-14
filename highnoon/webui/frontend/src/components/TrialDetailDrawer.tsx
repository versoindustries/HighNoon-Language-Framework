// TrialDetailDrawer.tsx - Slide-in drawer for detailed trial inspection
// Shows full hyperparameters, metrics timeline, and export functionality

import { useState, useCallback, useEffect } from 'react';
import {
    X,
    Download,
    Settings,
    TrendingDown,
    MemoryStick,
    Clock,
    Copy,
    Check,
    ChevronRight
} from 'lucide-react';
import type { HPOTrialInfo } from '../api/types';
import './TrialDetailDrawer.css';

interface TrialDetailDrawerProps {
    trial: HPOTrialInfo | null;
    isOpen: boolean;
    onClose: () => void;
    bestTrialId?: string | number | null;
}

type TabId = 'config' | 'metrics' | 'memory';

// Format hyperparameter value for display
function formatHyperparamValue(value: unknown): string {
    if (typeof value === 'number') {
        if (value < 0.001 && value > 0) return value.toExponential(3);
        if (value > 1000000) return `${(value / 1000000).toFixed(2)}M`;
        if (value > 1000) return `${(value / 1000).toFixed(1)}K`;
        return value.toFixed(Number.isInteger(value) ? 0 : 4);
    }
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    return String(value);
}

// Human-readable parameter names
const PARAM_DISPLAY_NAMES: Record<string, string> = {
    learning_rate: 'Learning Rate',
    embedding_dim: 'Embedding Dimension',
    num_reasoning_blocks: 'Reasoning Blocks',
    num_moe_experts: 'MoE Experts',
    batch_size: 'Batch Size',
    mamba_state_dim: 'Mamba State Dim',
    superposition_dim: 'Superposition Dim',
    wlam_heads: 'WLAM Heads',
    weight_decay: 'Weight Decay',
    grad_clip: 'Gradient Clip',
    hd_sample_length: 'HD Sample Length',
    vocab_size: 'Vocabulary Size',
    optimizer: 'Optimizer',
    scheduler: 'LR Scheduler',
    max_seq_len: 'Max Sequence Length',
    context_window: 'Context Window',
};

export function TrialDetailDrawer({
    trial,
    isOpen,
    onClose,
    bestTrialId,
}: TrialDetailDrawerProps) {
    const [activeTab, setActiveTab] = useState<TabId>('config');
    const [copied, setCopied] = useState(false);

    // Reset tab when opening new trial
    useEffect(() => {
        if (isOpen) {
            setActiveTab('config');
            setCopied(false);
        }
    }, [isOpen, trial?.trial_id]);

    const handleCopyConfig = useCallback(async () => {
        if (!trial?.hyperparams) return;

        const config = {
            trial_id: trial.trial_id,
            hyperparameters: trial.hyperparams,
            metrics: {
                loss: trial.loss,
                perplexity: trial.perplexity,
                confidence: trial.mean_confidence,
                composite_score: trial.composite_score,
            },
        };

        try {
            await navigator.clipboard.writeText(JSON.stringify(config, null, 2));
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    }, [trial]);

    const handleDownloadConfig = useCallback(() => {
        if (!trial?.hyperparams) return;

        const config = {
            trial_id: trial.trial_id,
            status: trial.status,
            hyperparameters: trial.hyperparams,
            metrics: {
                loss: trial.loss,
                perplexity: trial.perplexity,
                mean_confidence: trial.mean_confidence,
                composite_score: trial.composite_score,
                throughput_tokens_per_sec: trial.throughput_tokens_per_sec,
            },
            memory: {
                peak_mb: trial.peak_memory_mb,
                current_mb: trial.memory_mb,
            },
            duration_seconds: trial.duration_seconds,
        };

        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `trial_${trial.trial_id}_config.json`;
        a.click();
        URL.revokeObjectURL(url);
    }, [trial]);

    if (!trial) return null;

    const tabs = [
        { id: 'config' as TabId, label: 'Config', icon: <Settings size={14} /> },
        { id: 'metrics' as TabId, label: 'Metrics', icon: <TrendingDown size={14} /> },
        { id: 'memory' as TabId, label: 'Memory', icon: <MemoryStick size={14} /> },
    ];

    const isBest = trial.trial_id === bestTrialId;

    return (
        <>
            {/* Backdrop */}
            <div
                className={`drawer-backdrop ${isOpen ? 'drawer-backdrop--visible' : ''}`}
                onClick={onClose}
            />

            {/* Drawer */}
            <div className={`trial-drawer ${isOpen ? 'trial-drawer--open' : ''}`}>
                <div className="trial-drawer__header">
                    <div className="trial-drawer__title-section">
                        <h3 className="trial-drawer__title">
                            Trial {trial.trial_id}
                            {isBest && <span className="trial-drawer__best-badge">★ Best</span>}
                        </h3>
                        <span className={`trial-drawer__status trial-drawer__status--${trial.status}`}>
                            {trial.status}
                        </span>
                    </div>
                    <div className="trial-drawer__actions">
                        <button
                            className="trial-drawer__action-btn"
                            onClick={handleCopyConfig}
                            title="Copy config"
                        >
                            {copied ? <Check size={16} /> : <Copy size={16} />}
                        </button>
                        <button
                            className="trial-drawer__action-btn"
                            onClick={handleDownloadConfig}
                            title="Download config"
                        >
                            <Download size={16} />
                        </button>
                        <button
                            className="trial-drawer__close-btn"
                            onClick={onClose}
                        >
                            <X size={18} />
                        </button>
                    </div>
                </div>

                <div className="trial-drawer__tabs">
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            className={`trial-drawer__tab ${activeTab === tab.id ? 'trial-drawer__tab--active' : ''}`}
                            onClick={() => setActiveTab(tab.id)}
                        >
                            {tab.icon}
                            {tab.label}
                        </button>
                    ))}
                </div>

                <div className="trial-drawer__content">
                    {activeTab === 'config' && (
                        <div className="trial-drawer__config">
                            <table className="trial-drawer__param-table">
                                <tbody>
                                    {trial.hyperparams && Object.entries(trial.hyperparams)
                                        .filter(([key]) => !key.startsWith('_'))
                                        .sort(([a], [b]) => a.localeCompare(b))
                                        .map(([key, value]) => (
                                            <tr key={key}>
                                                <td className="trial-drawer__param-name">
                                                    {PARAM_DISPLAY_NAMES[key] || key.replace(/_/g, ' ')}
                                                </td>
                                                <td className="trial-drawer__param-value">
                                                    {formatHyperparamValue(value)}
                                                </td>
                                            </tr>
                                        ))
                                    }
                                    {!trial.hyperparams && (
                                        <tr>
                                            <td colSpan={2} className="trial-drawer__empty">
                                                No hyperparameters available
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    )}

                    {activeTab === 'metrics' && (
                        <div className="trial-drawer__metrics">
                            <div className="trial-drawer__metric-grid">
                                <div className="trial-drawer__metric-card">
                                    <span className="trial-drawer__metric-label">Loss</span>
                                    <span className="trial-drawer__metric-value">
                                        {trial.loss?.toFixed(6) ?? '—'}
                                    </span>
                                </div>
                                <div className="trial-drawer__metric-card">
                                    <span className="trial-drawer__metric-label">Perplexity</span>
                                    <span className="trial-drawer__metric-value">
                                        {trial.perplexity?.toFixed(2) ?? '—'}
                                    </span>
                                </div>
                                <div className="trial-drawer__metric-card">
                                    <span className="trial-drawer__metric-label">Confidence</span>
                                    <span className="trial-drawer__metric-value">
                                        {trial.mean_confidence ? `${(trial.mean_confidence * 100).toFixed(1)}%` : '—'}
                                    </span>
                                </div>
                                <div className="trial-drawer__metric-card">
                                    <span className="trial-drawer__metric-label">Composite</span>
                                    <span className="trial-drawer__metric-value">
                                        {trial.composite_score?.toFixed(4) ?? '—'}
                                    </span>
                                </div>
                                <div className="trial-drawer__metric-card">
                                    <span className="trial-drawer__metric-label">Throughput</span>
                                    <span className="trial-drawer__metric-value">
                                        {trial.throughput_tokens_per_sec
                                            ? `${(trial.throughput_tokens_per_sec / 1000).toFixed(1)}K tok/s`
                                            : '—'}
                                    </span>
                                </div>
                                <div className="trial-drawer__metric-card">
                                    <span className="trial-drawer__metric-label">Duration</span>
                                    <span className="trial-drawer__metric-value">
                                        {trial.duration_seconds
                                            ? `${Math.floor(trial.duration_seconds / 60)}m ${Math.round(trial.duration_seconds % 60)}s`
                                            : '—'}
                                    </span>
                                </div>
                            </div>

                            {trial.step && (
                                <div className="trial-drawer__progress">
                                    <span className="trial-drawer__progress-label">
                                        Training Progress
                                    </span>
                                    <div className="trial-drawer__progress-bar">
                                        <div
                                            className="trial-drawer__progress-fill"
                                            style={{ width: `${Math.min((trial.step / 1000) * 100, 100)}%` }}
                                        />
                                    </div>
                                    <span className="trial-drawer__progress-value">
                                        Step {trial.step}
                                    </span>
                                </div>
                            )}
                        </div>
                    )}

                    {activeTab === 'memory' && (
                        <div className="trial-drawer__memory">
                            <div className="trial-drawer__memory-bars">
                                <div className="trial-drawer__memory-item">
                                    <span className="trial-drawer__memory-label">Current RSS</span>
                                    <div className="trial-drawer__memory-bar-container">
                                        <div
                                            className="trial-drawer__memory-bar"
                                            style={{
                                                width: `${Math.min((trial.memory_mb || 0) / 64000 * 100, 100)}%`,
                                                background: 'linear-gradient(90deg, #6366f1, #8b5cf6)',
                                            }}
                                        />
                                    </div>
                                    <span className="trial-drawer__memory-value">
                                        {trial.memory_mb
                                            ? `${(trial.memory_mb / 1024).toFixed(1)} GB`
                                            : '—'}
                                    </span>
                                </div>

                                <div className="trial-drawer__memory-item">
                                    <span className="trial-drawer__memory-label">Peak Memory</span>
                                    <div className="trial-drawer__memory-bar-container">
                                        <div
                                            className="trial-drawer__memory-bar"
                                            style={{
                                                width: `${Math.min((trial.peak_memory_mb || 0) / 64000 * 100, 100)}%`,
                                                background: 'linear-gradient(90deg, #f59e0b, #ef4444)',
                                            }}
                                        />
                                    </div>
                                    <span className="trial-drawer__memory-value">
                                        {trial.peak_memory_mb
                                            ? `${(trial.peak_memory_mb / 1024).toFixed(1)} GB`
                                            : '—'}
                                    </span>
                                </div>
                            </div>

                            {trial.pruned_at_step && (
                                <div className="trial-drawer__prune-info">
                                    <span>Pruned at step {trial.pruned_at_step}</span>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </>
    );
}

export default TrialDetailDrawer;
