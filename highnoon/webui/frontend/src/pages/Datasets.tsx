// HighNoon Dashboard - Datasets Page
import { useState, useEffect, useCallback } from 'react';
import { Search, Plus, RefreshCw, ExternalLink, Download, Database, AlertCircle, Trash2, Lock, Settings, AlertTriangle } from 'lucide-react';
import { Card, Button, Modal, ConfirmModal } from '../components/ui';
import { DatasetDetailModal } from '../components/DatasetDetail';
import { datasetApi, settingsApi } from '../api/client';
import type { DatasetInfo, HuggingFaceDataset } from '../api/types';
import './Datasets.css';

const API_BASE = 'http://localhost:8000';

export function Datasets() {
    const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [filter, setFilter] = useState<'all' | 'huggingface' | 'local'>('all');
    const [isSearchModalOpen, setIsSearchModalOpen] = useState(false);
    const [hfSearchQuery, setHfSearchQuery] = useState('');
    const [hfResults, setHfResults] = useState<HuggingFaceDataset[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [hfSearchError, setHfSearchError] = useState<string | null>(null);
    const [selectedDataset, setSelectedDataset] = useState<DatasetInfo | null>(null);
    const [addingDataset, setAddingDataset] = useState<string | null>(null);
    const [deletingDataset, setDeletingDataset] = useState<string | null>(null);
    const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

    // HuggingFace token state
    const [hfTokenConfigured, setHfTokenConfigured] = useState(false);
    const [gatedWarningDataset, setGatedWarningDataset] = useState<HuggingFaceDataset | null>(null);

    // Fetch HF token status
    const fetchHfTokenStatus = useCallback(async () => {
        try {
            const status = await settingsApi.getHfTokenStatus();
            setHfTokenConfigured(status.configured);
        } catch {
            // Silent fail - assume no token
        }
    }, []);

    // Load datasets on mount
    const loadDatasets = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const data = await datasetApi.list();
            setDatasets(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load datasets');
            console.error('Failed to load datasets:', err);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        loadDatasets();
        fetchHfTokenStatus();
    }, [loadDatasets, fetchHfTokenStatus]);

    const filteredDatasets = datasets.filter((ds) => {
        if (filter !== 'all' && ds.source !== filter) return false;
        if (searchQuery && !ds.name.toLowerCase().includes(searchQuery.toLowerCase())) return false;
        return true;
    });

    const handleHfSearch = async () => {
        if (!hfSearchQuery.trim()) return;
        setIsSearching(true);
        setHfSearchError(null);
        try {
            const results = await datasetApi.searchHuggingFace(hfSearchQuery.trim(), 20);
            setHfResults(results);
        } catch (err) {
            setHfSearchError(err instanceof Error ? err.message : 'Search failed');
            console.error('HuggingFace search failed:', err);
        } finally {
            setIsSearching(false);
        }
    };

    const handleAddDataset = async (dataset: HuggingFaceDataset, forceAdd = false) => {
        // Check if gated and no token configured
        if (dataset.gated && !hfTokenConfigured && !forceAdd) {
            setGatedWarningDataset(dataset);
            return;
        }

        setAddingDataset(dataset.id);
        setGatedWarningDataset(null);
        try {
            const added = await datasetApi.addHuggingFace(dataset.id);
            setDatasets([...datasets, added]);
            setIsSearchModalOpen(false);
            setHfSearchQuery('');
            setHfResults([]);
        } catch (err) {
            console.error('Failed to add dataset:', err);
            // Show error but don't close modal
            setHfSearchError(err instanceof Error ? err.message : 'Failed to add dataset');
        } finally {
            setAddingDataset(null);
        }
    };

    const handleDeleteClick = (datasetId: string, e: React.MouseEvent) => {
        e.stopPropagation(); // Prevent card click
        setConfirmDeleteId(datasetId);
    };

    const handleConfirmDelete = async () => {
        if (!confirmDeleteId) return;

        setDeletingDataset(confirmDeleteId);
        try {
            await datasetApi.remove(confirmDeleteId);
            setDatasets(datasets.filter(ds => ds.id !== confirmDeleteId));
            setConfirmDeleteId(null);
        } catch (err) {
            console.error('Failed to delete dataset:', err);
            setError(err instanceof Error ? err.message : 'Failed to delete dataset');
        } finally {
            setDeletingDataset(null);
        }
    };

    return (
        <div className="page">
            <div className="page-header">
                <div className="page-header-content">
                    <h1 className="page-title">Datasets</h1>
                    <p className="page-subtitle">
                        Browse and manage datasets from HuggingFace Hub and local sources
                    </p>
                </div>
                <div className="page-header-actions">
                    <Button
                        variant="secondary"
                        leftIcon={<RefreshCw size={16} />}
                        loading={isLoading}
                        onClick={loadDatasets}
                    >
                        Refresh
                    </Button>
                    <Button
                        variant="primary"
                        leftIcon={<Plus size={16} />}
                        onClick={() => setIsSearchModalOpen(true)}
                    >
                        Add Dataset
                    </Button>
                </div>
            </div>

            {/* Filters */}
            <Card variant="glass" padding="md" className="filters-card">
                <div className="filters-row">
                    <div className="search-wrapper">
                        <Search size={18} className="search-icon" />
                        <input
                            type="text"
                            className="search-input"
                            placeholder="Search datasets..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                    </div>
                    <div className="filter-chips">
                        <button
                            className={`chip ${filter === 'all' ? 'chip-active' : ''}`}
                            onClick={() => setFilter('all')}
                        >
                            All
                        </button>
                        <button
                            className={`chip ${filter === 'huggingface' ? 'chip-active' : ''}`}
                            onClick={() => setFilter('huggingface')}
                        >
                            HuggingFace
                        </button>
                        <button
                            className={`chip ${filter === 'local' ? 'chip-active' : ''}`}
                            onClick={() => setFilter('local')}
                        >
                            Local
                        </button>
                    </div>
                </div>
            </Card>

            {/* Datasets Grid */}
            <div className="datasets-grid">
                {isLoading ? (
                    <div className="datasets-loading-state">
                        <RefreshCw size={32} className="spin" />
                        <p>Loading datasets...</p>
                    </div>
                ) : error ? (
                    <div className="datasets-error-state">
                        <AlertCircle size={48} />
                        <h3>Failed to load datasets</h3>
                        <p>{error}</p>
                        <Button variant="primary" onClick={loadDatasets}>
                            Try Again
                        </Button>
                    </div>
                ) : filteredDatasets.length > 0 ? (
                    filteredDatasets.map((dataset) => (
                        <Card
                            key={dataset.id}
                            hover
                            padding="lg"
                            className="dataset-card"
                            onClick={() => setSelectedDataset(dataset)}
                        >
                            <div className="dataset-header">
                                <div className="dataset-icon">
                                    <Database size={24} />
                                </div>
                                <span className={`dataset-source source-${dataset.source}`}>
                                    {dataset.source}
                                </span>
                                <button
                                    className="dataset-delete"
                                    onClick={(e) => handleDeleteClick(dataset.id, e)}
                                    disabled={deletingDataset === dataset.id}
                                    aria-label={`Delete ${dataset.name}`}
                                    title="Delete dataset"
                                >
                                    {deletingDataset === dataset.id ? (
                                        <RefreshCw size={16} className="spin" />
                                    ) : (
                                        <Trash2 size={16} />
                                    )}
                                </button>
                            </div>
                            <h3 className="dataset-name">{dataset.name}</h3>
                            <p className="dataset-description">{dataset.description}</p>
                            <div className="dataset-meta">
                                <span>{dataset.num_examples.toLocaleString()} examples</span>
                                {dataset.download_status && (
                                    <span className={`status-badge status-${dataset.download_status}`}>
                                        {dataset.download_status}
                                    </span>
                                )}
                            </div>
                        </Card>
                    ))
                ) : (
                    <div className="datasets-empty-state">
                        <Database size={48} />
                        <h3>No datasets yet</h3>
                        <p>Add datasets from HuggingFace Hub to get started</p>
                        <Button
                            variant="primary"
                            leftIcon={<Plus size={16} />}
                            onClick={() => setIsSearchModalOpen(true)}
                        >
                            Add Dataset
                        </Button>
                    </div>
                )}
            </div>

            {/* HuggingFace Search Modal */}
            <Modal
                open={isSearchModalOpen}
                onClose={() => setIsSearchModalOpen(false)}
                title="Add Dataset from HuggingFace"
                size="lg"
            >
                <div className="hf-search">
                    <div className="hf-search-bar">
                        <Search size={18} className="search-icon" />
                        <input
                            type="text"
                            className="search-input"
                            placeholder="Search HuggingFace datasets (e.g., 'code', 'openwebtext')..."
                            value={hfSearchQuery}
                            onChange={(e) => setHfSearchQuery(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleHfSearch()}
                        />
                        <Button loading={isSearching} onClick={handleHfSearch}>
                            Search
                        </Button>
                    </div>

                    {hfSearchError && (
                        <div className="hf-search-error">
                            <AlertCircle size={16} />
                            <span>{hfSearchError}</span>
                        </div>
                    )}

                    <div className="hf-results">
                        {hfResults.length > 0 ? (
                            hfResults.map((dataset) => (
                                <div key={dataset.id} className="hf-result-card">
                                    <div className="hf-result-info">
                                        <div className="hf-result-header">
                                            <span className="hf-result-author">{dataset.author}</span>
                                            <span className="hf-result-name">{dataset.name}</span>
                                            {dataset.gated && (
                                                <span className="hf-gated-badge" title="This dataset requires authentication">
                                                    <Lock size={12} />
                                                    Gated
                                                </span>
                                            )}
                                        </div>
                                        <p className="hf-result-description">{dataset.description}</p>
                                        <div className="hf-result-meta">
                                            <span>↓ {dataset.downloads.toLocaleString()}</span>
                                            <span>♥ {dataset.likes.toLocaleString()}</span>
                                            {dataset.tags.slice(0, 3).map((tag) => (
                                                <span key={tag} className="hf-tag">{tag}</span>
                                            ))}
                                        </div>
                                    </div>
                                    <div className="hf-result-actions">
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            leftIcon={<ExternalLink size={14} />}
                                            onClick={() =>
                                                window.open(`https://huggingface.co/datasets/${dataset.id}`, '_blank')
                                            }
                                        >
                                            View
                                        </Button>
                                        <Button
                                            variant="primary"
                                            size="sm"
                                            leftIcon={<Download size={14} />}
                                            loading={addingDataset === dataset.id}
                                            onClick={() => handleAddDataset(dataset)}
                                        >
                                            Add
                                        </Button>
                                    </div>
                                </div>
                            ))
                        ) : (
                            <div className="hf-results-empty">
                                <Search size={32} />
                                <p>Search for datasets on HuggingFace Hub</p>
                            </div>
                        )}
                    </div>
                </div>
            </Modal>

            {/* Dataset Detail Modal */}
            <DatasetDetailModal
                dataset={selectedDataset}
                open={!!selectedDataset}
                onClose={() => setSelectedDataset(null)}
                onDelete={(id) => setDatasets(datasets.filter(d => d.id !== id))}
            />

            {/* Delete Confirmation Modal */}
            <ConfirmModal
                open={!!confirmDeleteId}
                onClose={() => setConfirmDeleteId(null)}
                onConfirm={handleConfirmDelete}
                title="Delete Dataset"
                description={`Are you sure you want to delete "${confirmDeleteId}"? This action cannot be undone.`}
                confirmText="Delete"
                cancelText="Cancel"
                variant="danger"
                loading={deletingDataset === confirmDeleteId}
            />

            {/* Gated Dataset Warning Modal */}
            <Modal
                open={!!gatedWarningDataset}
                onClose={() => setGatedWarningDataset(null)}
                title="Gated Dataset"
                size="md"
            >
                <div className="gated-warning-content">
                    <div className="gated-warning-icon">
                        <AlertTriangle size={48} />
                    </div>
                    <h3>Authentication Required</h3>
                    <p>
                        <strong>{gatedWarningDataset?.id}</strong> is a gated dataset that requires
                        authentication to access. You'll need to configure your HuggingFace token
                        in Settings to download and use this dataset for training.
                    </p>
                    <div className="gated-warning-actions">
                        <Button
                            variant="primary"
                            leftIcon={<Settings size={16} />}
                            onClick={() => {
                                setGatedWarningDataset(null);
                                setIsSearchModalOpen(false);
                                // Navigate to settings - using window.location for simplicity
                                window.location.href = '/settings';
                            }}
                        >
                            Configure Token
                        </Button>
                        <Button
                            variant="ghost"
                            onClick={() => {
                                if (gatedWarningDataset) {
                                    handleAddDataset(gatedWarningDataset, true);
                                }
                            }}
                            loading={addingDataset === gatedWarningDataset?.id}
                        >
                            Add Anyway
                        </Button>
                    </div>
                    <p className="gated-warning-note">
                        <Lock size={14} />
                        Without a token, you may not be able to access this dataset's content.
                    </p>
                </div>
            </Modal>
        </div>
    );
}
