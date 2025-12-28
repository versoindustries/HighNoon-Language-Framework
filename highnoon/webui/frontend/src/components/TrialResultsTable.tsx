// TrialResultsTable.tsx - Enterprise-grade trial results table
// Sortable, expandable rows with status badges and best trial highlighting

import { useState, useMemo, useCallback } from 'react';
import {
    ChevronDown,
    ChevronRight,
    Star,
    Download,
    ArrowUpDown,
    ArrowUp,
    ArrowDown,
    CheckCircle2,
    XCircle,
    AlertTriangle,
    Loader2,
    SkipForward
} from 'lucide-react';
import { Button } from './ui';
import type { HPOTrialInfo } from '../api/types';
import './TrialResultsTable.css';

interface TrialResultsTableProps {
    trials: HPOTrialInfo[];
    bestTrialId?: string | number | null;
    currentTrialId?: string | number | null;
    onExport?: () => void;
    maxHeight?: number | string;
}

type SortKey = 'trial_id' | 'status' | 'loss' | 'perplexity' | 'mean_confidence' | 'composite_score' | 'memory_mb' | 'throughput';
type SortDirection = 'asc' | 'desc';

interface SortConfig {
    key: SortKey;
    direction: SortDirection;
}

const STATUS_CONFIG: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
    running: {
        icon: <Loader2 size={14} className="animate-spin" />,
        color: 'var(--color-info)',
        label: 'Running'
    },
    completed: {
        icon: <CheckCircle2 size={14} />,
        color: 'var(--color-success)',
        label: 'Completed'
    },
    failed: {
        icon: <XCircle size={14} />,
        color: 'var(--color-danger)',
        label: 'Failed'
    },
    pruned: {
        icon: <AlertTriangle size={14} />,
        color: 'var(--color-warning)',
        label: 'Pruned'
    },
    skipped: {
        icon: <SkipForward size={14} />,
        color: 'var(--text-muted)',
        label: 'Skipped'
    },
};

export function TrialResultsTable({
    trials,
    bestTrialId,
    currentTrialId,
    onExport,
    maxHeight = 400,
}: TrialResultsTableProps) {
    const [expandedRows, setExpandedRows] = useState<Set<string | number>>(new Set());
    const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'trial_id', direction: 'desc' });

    // Toggle row expansion
    const toggleRow = useCallback((trialId: string | number) => {
        setExpandedRows(prev => {
            const next = new Set(prev);
            if (next.has(trialId)) {
                next.delete(trialId);
            } else {
                next.add(trialId);
            }
            return next;
        });
    }, []);

    // Handle sorting
    const handleSort = useCallback((key: SortKey) => {
        setSortConfig(prev => ({
            key,
            direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc',
        }));
    }, []);

    // Sort trials
    const sortedTrials = useMemo(() => {
        const sorted = [...trials];
        sorted.sort((a, b) => {
            const { key, direction } = sortConfig;
            let aVal: number | string = 0;
            let bVal: number | string = 0;

            switch (key) {
                case 'trial_id':
                    aVal = typeof a.trial_id === 'string' ? parseInt(a.trial_id.replace(/\D/g, '')) || 0 : a.trial_id;
                    bVal = typeof b.trial_id === 'string' ? parseInt(b.trial_id.replace(/\D/g, '')) || 0 : b.trial_id;
                    break;
                case 'status':
                    aVal = a.status;
                    bVal = b.status;
                    break;
                case 'loss':
                    aVal = a.loss ?? Infinity;
                    bVal = b.loss ?? Infinity;
                    break;
                case 'perplexity':
                    aVal = a.perplexity ?? Infinity;
                    bVal = b.perplexity ?? Infinity;
                    break;
                case 'mean_confidence':
                    aVal = a.mean_confidence ?? -Infinity;
                    bVal = b.mean_confidence ?? -Infinity;
                    break;
                case 'composite_score':
                    aVal = a.composite_score ?? -Infinity;
                    bVal = b.composite_score ?? -Infinity;
                    break;
                case 'memory_mb':
                    aVal = a.memory_mb ?? 0;
                    bVal = b.memory_mb ?? 0;
                    break;
                case 'throughput':
                    aVal = a.throughput_tokens_per_sec ?? 0;
                    bVal = b.throughput_tokens_per_sec ?? 0;
                    break;
            }

            if (aVal < bVal) return direction === 'asc' ? -1 : 1;
            if (aVal > bVal) return direction === 'asc' ? 1 : -1;
            return 0;
        });
        return sorted;
    }, [trials, sortConfig]);

    // Get sort icon
    const getSortIcon = (key: SortKey) => {
        if (sortConfig.key !== key) return <ArrowUpDown size={14} className="sort-icon sort-icon--inactive" />;
        return sortConfig.direction === 'asc'
            ? <ArrowUp size={14} className="sort-icon" />
            : <ArrowDown size={14} className="sort-icon" />;
    };

    // Format trial ID for display
    const formatTrialId = (id: string | number) => {
        if (typeof id === 'string') {
            const match = id.match(/\d+/);
            return match ? `#${match[0]}` : id;
        }
        return `#${id}`;
    };

    // Export to CSV
    const handleExport = useCallback(() => {
        if (onExport) {
            onExport();
            return;
        }

        const headers = ['Trial', 'Status', 'Learning Rate', 'Loss', 'Perplexity', 'Confidence', 'Composite', 'Memory (MB)', 'Throughput (tok/s)'];
        const rows = trials.map(t => [
            formatTrialId(t.trial_id),
            t.status,
            t.learning_rate?.toExponential(2) ?? '',
            t.loss?.toFixed(6) ?? '',
            t.perplexity?.toFixed(2) ?? '',
            t.mean_confidence?.toFixed(4) ?? '',
            t.composite_score?.toFixed(4) ?? '',
            t.memory_mb?.toFixed(0) ?? '',
            t.throughput_tokens_per_sec?.toFixed(1) ?? '',
        ]);

        const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `hpo_trials_${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    }, [trials, onExport]);

    if (trials.length === 0) {
        return (
            <div className="trial-table trial-table--empty">
                <div className="trial-table__empty-state">
                    <span className="trial-table__empty-icon">📊</span>
                    <p>No trials yet</p>
                    <span className="trial-table__empty-hint">Trials will appear here once optimization begins</span>
                </div>
            </div>
        );
    }

    return (
        <div className="trial-table">
            <div className="trial-table__header">
                <h4 className="trial-table__title">Trial Results</h4>
                <div className="trial-table__actions">
                    <span className="trial-table__count">{trials.length} trials</span>
                    <Button
                        variant="ghost"
                        size="sm"
                        leftIcon={<Download size={14} />}
                        onClick={handleExport}
                    >
                        Export
                    </Button>
                </div>
            </div>
            <div className="trial-table__wrapper" style={{ maxHeight }}>
                <table className="trial-table__table">
                    <thead>
                        <tr>
                            <th className="trial-table__th trial-table__th--expand"></th>
                            <th
                                className="trial-table__th trial-table__th--sortable"
                                onClick={() => handleSort('trial_id')}
                            >
                                Trial {getSortIcon('trial_id')}
                            </th>
                            <th
                                className="trial-table__th trial-table__th--sortable"
                                onClick={() => handleSort('status')}
                            >
                                Status {getSortIcon('status')}
                            </th>
                            <th className="trial-table__th">LR</th>
                            <th
                                className="trial-table__th trial-table__th--sortable"
                                onClick={() => handleSort('loss')}
                            >
                                Loss {getSortIcon('loss')}
                            </th>
                            <th
                                className="trial-table__th trial-table__th--sortable"
                                onClick={() => handleSort('perplexity')}
                            >
                                PPL {getSortIcon('perplexity')}
                            </th>
                            <th
                                className="trial-table__th trial-table__th--sortable"
                                onClick={() => handleSort('mean_confidence')}
                            >
                                Conf {getSortIcon('mean_confidence')}
                            </th>
                            <th
                                className="trial-table__th trial-table__th--sortable"
                                onClick={() => handleSort('composite_score')}
                            >
                                Composite {getSortIcon('composite_score')}
                            </th>
                            <th
                                className="trial-table__th trial-table__th--sortable"
                                onClick={() => handleSort('memory_mb')}
                            >
                                Memory {getSortIcon('memory_mb')}
                            </th>
                            <th
                                className="trial-table__th trial-table__th--sortable"
                                onClick={() => handleSort('throughput')}
                            >
                                tok/s {getSortIcon('throughput')}
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {sortedTrials.map((trial) => {
                            const isBest = trial.trial_id === bestTrialId;
                            const isCurrent = trial.trial_id === currentTrialId;
                            const isExpanded = expandedRows.has(trial.trial_id);
                            const statusConfig = STATUS_CONFIG[trial.status] || STATUS_CONFIG.completed;

                            return (
                                <>
                                    <tr
                                        key={trial.trial_id}
                                        className={`trial-table__row ${isBest ? 'trial-table__row--best' : ''} ${isCurrent ? 'trial-table__row--current' : ''}`}
                                    >
                                        <td className="trial-table__td trial-table__td--expand">
                                            <button
                                                className="trial-table__expand-btn"
                                                onClick={() => toggleRow(trial.trial_id)}
                                                aria-label={isExpanded ? 'Collapse row' : 'Expand row'}
                                            >
                                                {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                                            </button>
                                        </td>
                                        <td className="trial-table__td trial-table__td--id">
                                            <span className="trial-table__id">
                                                {formatTrialId(trial.trial_id)}
                                                {isBest && (
                                                    <Star size={14} className="trial-table__star" fill="currentColor" />
                                                )}
                                            </span>
                                        </td>
                                        <td className="trial-table__td">
                                            <span
                                                className="trial-table__status"
                                                style={{ color: statusConfig.color }}
                                            >
                                                {statusConfig.icon}
                                                {statusConfig.label}
                                            </span>
                                        </td>
                                        <td className="trial-table__td trial-table__td--mono">
                                            {trial.learning_rate?.toExponential(2) ?? '—'}
                                        </td>
                                        <td className="trial-table__td trial-table__td--mono">
                                            {trial.loss?.toFixed(4) ?? '—'}
                                        </td>
                                        <td className="trial-table__td trial-table__td--mono">
                                            {trial.perplexity?.toFixed(2) ?? '—'}
                                        </td>
                                        <td className="trial-table__td trial-table__td--mono">
                                            {trial.mean_confidence?.toFixed(3) ?? '—'}
                                        </td>
                                        <td className="trial-table__td trial-table__td--mono trial-table__td--highlight">
                                            {trial.composite_score?.toFixed(4) ?? '—'}
                                        </td>
                                        <td className="trial-table__td trial-table__td--mono">
                                            {trial.memory_mb ? `${trial.memory_mb.toFixed(0)} MB` : '—'}
                                        </td>
                                        <td className="trial-table__td trial-table__td--mono">
                                            {trial.throughput_tokens_per_sec ? `${trial.throughput_tokens_per_sec.toFixed(1)}` : '—'}
                                        </td>
                                    </tr>
                                    {isExpanded && trial.hyperparams && (
                                        <tr className="trial-table__row trial-table__row--expanded">
                                            <td colSpan={10} className="trial-table__td trial-table__td--expanded">
                                                <div className="trial-table__details">
                                                    <h5 className="trial-table__details-title">Hyperparameters</h5>
                                                    <div className="trial-table__details-grid">
                                                        {Object.entries(trial.hyperparams).map(([key, value]) => (
                                                            <div key={key} className="trial-table__detail-item">
                                                                <span className="trial-table__detail-key">{key}</span>
                                                                <span className="trial-table__detail-value">
                                                                    {typeof value === 'number'
                                                                        ? (Number.isInteger(value) ? value : value.toFixed(6))
                                                                        : String(value)}
                                                                </span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                    {trial.duration_seconds && (
                                                        <div className="trial-table__details-footer">
                                                            <span>Duration: {Math.round(trial.duration_seconds)}s</span>
                                                            {trial.pruned_at_step && (
                                                                <span>Pruned at step {trial.pruned_at_step}</span>
                                                            )}
                                                        </div>
                                                    )}
                                                </div>
                                            </td>
                                        </tr>
                                    )}
                                </>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

export default TrialResultsTable;
