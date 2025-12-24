// DatasetDetailModal - Enhanced Dataset View with Preview
// Shows dataset information, data samples, and actions

import { useState, useEffect } from 'react';
import {
    Database, ExternalLink, Download, Plus, X, RefreshCw,
    AlertCircle, FileText, Layers, Hash
} from 'lucide-react';
import { Button, Modal, ConfirmModal } from '../ui';
import { datasetApi } from '../../api/client';
import type { DatasetInfo } from '../../api/types';
import './DatasetDetail.css';

// =============================================================================
// TYPES
// =============================================================================

interface DatasetPreview {
    features: Array<{ name: string; type: string }>;
    rows: Array<Record<string, unknown>>;
    totalRows: number;
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

interface DatasetDetailModalProps {
    dataset: DatasetInfo | null;
    open: boolean;
    onClose: () => void;
    onAddToCurriculum?: (datasetId: string) => void;
    onDelete?: (datasetId: string) => void;
}

export function DatasetDetailModal({
    dataset,
    open,
    onClose,
    onAddToCurriculum,
    onDelete
}: DatasetDetailModalProps) {
    const [preview, setPreview] = useState<DatasetPreview | null>(null);
    const [loadingPreview, setLoadingPreview] = useState(false);
    const [previewError, setPreviewError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<'overview' | 'preview'>('overview');
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);

    // Load preview when opening
    useEffect(() => {
        if (open && dataset && activeTab === 'preview') {
            loadPreview();
        }
    }, [open, dataset, activeTab]);

    const loadPreview = async () => {
        if (!dataset) return;

        setLoadingPreview(true);
        setPreviewError(null);

        try {
            // For HuggingFace datasets, use the preview API
            if (dataset.source === 'huggingface') {
                const response = await fetch(
                    `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(dataset.id)}&config=default&split=train&offset=0&length=10`
                );

                if (!response.ok) {
                    throw new Error('Preview not available for this dataset');
                }

                const data = await response.json();

                setPreview({
                    features: data.features?.map((f: { name: string; type: { dtype?: string } }) => ({
                        name: f.name,
                        type: f.type?.dtype || 'unknown'
                    })) || [],
                    rows: data.rows?.map((r: { row: Record<string, unknown> }) => r.row) || [],
                    totalRows: data.num_rows_total || 0,
                });
            } else {
                setPreviewError('Preview only available for HuggingFace datasets');
            }
        } catch (err) {
            setPreviewError(err instanceof Error ? err.message : 'Failed to load preview');
        } finally {
            setLoadingPreview(false);
        }
    };

    const handleDelete = async () => {
        if (!dataset) return;

        setIsDeleting(true);
        try {
            await datasetApi.remove(dataset.id);
            if (onDelete) onDelete(dataset.id);
            setShowDeleteConfirm(false);
            onClose();
        } catch (err) {
            console.error('Failed to delete:', err);
        } finally {
            setIsDeleting(false);
        }
    };

    if (!dataset) return null;

    return (
        <>
            <Modal open={open} onClose={onClose} title="" size="lg">
                <div className="dataset-detail">
                    {/* Header */}
                    <div className="dataset-detail-header">
                        <div className="dataset-detail-icon">
                            <Database size={28} />
                        </div>
                        <div className="dataset-detail-title">
                            <h2>{dataset.name}</h2>
                            <div className="dataset-detail-badges">
                                <span className={`source-badge source-${dataset.source}`}>
                                    {dataset.source}
                                </span>
                                {dataset.download_status && (
                                    <span className={`status-badge status-${dataset.download_status}`}>
                                        {dataset.download_status}
                                    </span>
                                )}
                            </div>
                        </div>
                        <button className="detail-close" onClick={onClose}>
                            <X size={20} />
                        </button>
                    </div>

                    {/* Tabs */}
                    <div className="detail-tabs">
                        <button
                            className={`detail-tab ${activeTab === 'overview' ? 'detail-tab-active' : ''}`}
                            onClick={() => setActiveTab('overview')}
                        >
                            Overview
                        </button>
                        <button
                            className={`detail-tab ${activeTab === 'preview' ? 'detail-tab-active' : ''}`}
                            onClick={() => setActiveTab('preview')}
                        >
                            Data Preview
                        </button>
                    </div>

                    {/* Content */}
                    <div className="detail-content">
                        {activeTab === 'overview' && (
                            <div className="detail-overview">
                                <p className="dataset-description">{dataset.description || 'No description available.'}</p>

                                <div className="detail-stats">
                                    <div className="detail-stat">
                                        <Hash size={16} />
                                        <span className="stat-label">Examples</span>
                                        <span className="stat-value">{dataset.num_examples.toLocaleString()}</span>
                                    </div>
                                    {dataset.size_bytes && (
                                        <div className="detail-stat">
                                            <Layers size={16} />
                                            <span className="stat-label">Size</span>
                                            <span className="stat-value">
                                                {(dataset.size_bytes / 1024 / 1024 / 1024).toFixed(2)} GB
                                            </span>
                                        </div>
                                    )}
                                    {dataset.splits && (
                                        <div className="detail-stat">
                                            <FileText size={16} />
                                            <span className="stat-label">Splits</span>
                                            <span className="stat-value">{dataset.splits.join(', ')}</span>
                                        </div>
                                    )}
                                </div>

                                {dataset.features && Object.keys(dataset.features).length > 0 && (
                                    <div className="detail-features">
                                        <h4>Features</h4>
                                        <div className="features-grid">
                                            {Object.entries(dataset.features).map(([name, type]) => (
                                                <div key={name} className="feature-item">
                                                    <span className="feature-name">{name}</span>
                                                    <span className="feature-type">{type}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {dataset.source === 'huggingface' && (
                                    <a
                                        href={`https://huggingface.co/datasets/${dataset.id}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="hf-link"
                                    >
                                        <ExternalLink size={14} /> View on HuggingFace
                                    </a>
                                )}
                            </div>
                        )}

                        {activeTab === 'preview' && (
                            <div className="detail-preview">
                                {loadingPreview ? (
                                    <div className="preview-loading">
                                        <RefreshCw size={24} className="spin" />
                                        <span>Loading preview...</span>
                                    </div>
                                ) : previewError ? (
                                    <div className="preview-error">
                                        <AlertCircle size={24} />
                                        <span>{previewError}</span>
                                        <Button size="sm" onClick={loadPreview}>Retry</Button>
                                    </div>
                                ) : preview ? (
                                    <>
                                        <div className="preview-info">
                                            Showing {preview.rows.length} of {preview.totalRows.toLocaleString()} rows
                                        </div>
                                        <div className="preview-table-wrapper">
                                            <table className="preview-table">
                                                <thead>
                                                    <tr>
                                                        {preview.features.slice(0, 5).map((f) => (
                                                            <th key={f.name}>
                                                                {f.name}
                                                                <span className="col-type">{f.type}</span>
                                                            </th>
                                                        ))}
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {preview.rows.map((row, i) => (
                                                        <tr key={i}>
                                                            {preview.features.slice(0, 5).map((f) => (
                                                                <td key={f.name}>
                                                                    {truncateValue(row[f.name])}
                                                                </td>
                                                            ))}
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </>
                                ) : (
                                    <div className="preview-empty">
                                        <p>Click to load data preview</p>
                                        <Button onClick={loadPreview}>Load Preview</Button>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    <div className="detail-actions">
                        <Button
                            variant="danger"
                            onClick={() => setShowDeleteConfirm(true)}
                        >
                            Delete
                        </Button>
                        <div className="action-spacer" />
                        {onAddToCurriculum && (
                            <Button
                                variant="primary"
                                leftIcon={<Plus size={16} />}
                                onClick={() => onAddToCurriculum(dataset.id)}
                            >
                                Add to Curriculum
                            </Button>
                        )}
                    </div>
                </div>
            </Modal>

            {/* Delete Confirmation Modal */}
            <ConfirmModal
                open={showDeleteConfirm}
                onClose={() => setShowDeleteConfirm(false)}
                onConfirm={handleDelete}
                title="Delete Dataset"
                description={`Are you sure you want to delete "${dataset.name}"? This action cannot be undone.`}
                confirmText="Delete"
                cancelText="Cancel"
                variant="danger"
                loading={isDeleting}
            />
        </>
    );
}

// Helper to truncate long values
function truncateValue(value: unknown, maxLength: number = 100): string {
    if (value === null || value === undefined) return '—';
    const str = typeof value === 'string' ? value : JSON.stringify(value);
    return str.length > maxLength ? str.slice(0, maxLength) + '...' : str;
}

export default DatasetDetailModal;
