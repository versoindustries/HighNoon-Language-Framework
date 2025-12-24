// HighNoon Dashboard - Curriculum Builder Page
import { useState, useEffect, useCallback } from 'react';
import { Plus, Trash2, GripVertical, Save, Upload, Search, Database, AlertCircle, RefreshCw, BookOpen } from 'lucide-react';
import { Card, CardHeader, CardContent, Button, Input, Select, Modal } from '../components/ui';
import { datasetApi, curriculumApi } from '../api/client';
import { TemplateGallery, TEMPLATES, templateToCurriculumStages } from '../components/TemplateGallery';
import type { TrainingTemplate } from '../components/TemplateGallery';
import type { CurriculumStage, DatasetInfo, CurriculumDataset, Curriculum } from '../api/types';
import './Curriculum.css';

const DEFAULT_STAGES: CurriculumStage[] = [
    {
        name: 'foundation',
        display_name: 'Foundation',
        module: 'language_modeling',
        datasets: [],
        epochs: 1,
        learning_rate: '1e-4',
        batch_size: 8,
        weight: 1.0,
    },
];

export function Curriculum() {
    const [stages, setStages] = useState<CurriculumStage[]>(DEFAULT_STAGES);
    const [selectedStage, setSelectedStage] = useState<string | null>('foundation');
    const [curriculumName, setCurriculumName] = useState('My Curriculum');
    const [curriculumId, setCurriculumId] = useState<string | null>(null);  // Null = new curriculum
    const [isLoadModalOpen, setIsLoadModalOpen] = useState(false);
    const [savedCurricula, setSavedCurricula] = useState<Curriculum[]>([]);
    const [loadingCurricula, setLoadingCurricula] = useState(false);
    const [isAddModalOpen, setIsAddModalOpen] = useState(false);
    const [isDatasetModalOpen, setIsDatasetModalOpen] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [saveError, setSaveError] = useState<string | null>(null);
    const [saveSuccess, setSaveSuccess] = useState(false);

    // Baseline template gallery state
    const [isTemplateGalleryOpen, setIsTemplateGalleryOpen] = useState(false);
    const [loadedTemplateName, setLoadedTemplateName] = useState<string | null>(null);



    // New stage form
    const [newStageId, setNewStageId] = useState('');
    const [newStageName, setNewStageName] = useState('');
    const [newStageModule, setNewStageModule] = useState('language_modeling');

    // Dataset picker state
    const [availableDatasets, setAvailableDatasets] = useState<DatasetInfo[]>([]);
    const [loadingDatasets, setLoadingDatasets] = useState(false);
    const [datasetError, setDatasetError] = useState<string | null>(null);
    const [datasetSearchQuery, setDatasetSearchQuery] = useState('');
    const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
    const [datasetWeight, setDatasetWeight] = useState(1.0);

    const currentStage = stages.find((s) => s.name === selectedStage);

    // Load available datasets when modal opens
    const loadAvailableDatasets = useCallback(async () => {
        setLoadingDatasets(true);
        setDatasetError(null);
        try {
            const datasets = await datasetApi.list();
            setAvailableDatasets(datasets);
        } catch (err) {
            setDatasetError(err instanceof Error ? err.message : 'Failed to load datasets');
            console.error('Failed to load datasets:', err);
        } finally {
            setLoadingDatasets(false);
        }
    }, []);

    useEffect(() => {
        if (isDatasetModalOpen) {
            loadAvailableDatasets();
        }
    }, [isDatasetModalOpen, loadAvailableDatasets]);

    // Check for template selection from Dashboard on page load
    useEffect(() => {
        const storedTemplate = sessionStorage.getItem('highnoon-selected-template');
        if (storedTemplate) {
            try {
                const { id } = JSON.parse(storedTemplate);
                const template = TEMPLATES.find(t => t.id === id);
                if (template) {
                    const convertedStages = templateToCurriculumStages(template);
                    setStages(convertedStages);
                    setCurriculumName(`${template.name} Curriculum`);
                    setSelectedStage(convertedStages[0]?.name ?? null);
                    setLoadedTemplateName(template.name);
                }
                // Clear after loading
                sessionStorage.removeItem('highnoon-selected-template');
            } catch (err) {
                console.error('Failed to parse stored template:', err);
            }
        }
    }, []);

    // Filter datasets based on search query
    const filteredDatasets = availableDatasets.filter((ds) => {
        if (!datasetSearchQuery) return true;
        return (
            ds.name.toLowerCase().includes(datasetSearchQuery.toLowerCase()) ||
            ds.id.toLowerCase().includes(datasetSearchQuery.toLowerCase()) ||
            ds.description.toLowerCase().includes(datasetSearchQuery.toLowerCase())
        );
    });

    // Filter out already added datasets
    const availableForSelection = filteredDatasets.filter((ds) => {
        if (!currentStage) return true;
        return !currentStage.datasets.some((d) => d.dataset_id === ds.id);
    });

    const handleAddStage = () => {
        if (!newStageId || !newStageName) return;

        const newStage: CurriculumStage = {
            name: newStageId.toLowerCase().replace(/\s+/g, '_'),
            display_name: newStageName,
            module: newStageModule,
            datasets: [],
            epochs: 1,
            learning_rate: '1e-4',
            batch_size: 8,
            weight: 1.0,
        };

        setStages([...stages, newStage]);
        setSelectedStage(newStage.name);
        setIsAddModalOpen(false);
        setNewStageId('');
        setNewStageName('');
    };

    const handleDeleteStage = (stageName: string) => {
        setStages(stages.filter((s) => s.name !== stageName));
        if (selectedStage === stageName) {
            setSelectedStage(stages[0]?.name ?? null);
        }
    };

    const handleUpdateStage = (updates: Partial<CurriculumStage>) => {
        if (!selectedStage) return;
        setStages(
            stages.map((s) => (s.name === selectedStage ? { ...s, ...updates } : s))
        );
    };

    const handleAddDataset = () => {
        if (!selectedDatasetId || !currentStage) return;

        const newDataset: CurriculumDataset = {
            dataset_id: selectedDatasetId,
            weight: datasetWeight,
        };

        handleUpdateStage({
            datasets: [...currentStage.datasets, newDataset],
        });

        // Reset and close modal
        setSelectedDatasetId(null);
        setDatasetWeight(1.0);
        setDatasetSearchQuery('');
        setIsDatasetModalOpen(false);
    };

    const handleSave = async () => {
        setIsSaving(true);
        setSaveError(null);
        setSaveSuccess(false);

        try {
            // Generate ID if this is a new curriculum
            const id = curriculumId || crypto.randomUUID();
            const now = new Date().toISOString();

            const curriculumToSave: Curriculum = {
                id,
                name: curriculumName,
                stages,
                created_at: curriculumId ? '' : now,  // Backend handles this
                updated_at: now,
            };

            await curriculumApi.save(curriculumToSave);

            // Update local ID state after successful save
            setCurriculumId(id);
            setSaveSuccess(true);
            // Clear success message after 3 seconds
            setTimeout(() => setSaveSuccess(false), 3000);
        } catch (err) {
            setSaveError(err instanceof Error ? err.message : 'Failed to save curriculum');
            console.error('Failed to save curriculum:', err);
        } finally {
            setIsSaving(false);
        }
    };

    const handleLoad = async () => {
        setIsLoadModalOpen(true);
        setLoadingCurricula(true);
        try {
            const curricula = await curriculumApi.list();
            setSavedCurricula(curricula);
        } catch (err) {
            console.error('Failed to load curricula:', err);
            setSavedCurricula([]);
        } finally {
            setLoadingCurricula(false);
        }
    };

    const handleSelectCurriculum = (curriculum: Curriculum) => {
        setCurriculumId(curriculum.id);
        setCurriculumName(curriculum.name);
        setStages(curriculum.stages);
        setSelectedStage(curriculum.stages[0]?.name ?? null);
        setLoadedTemplateName(null); // Clear template name since this is custom
        setIsLoadModalOpen(false);
    };

    // Handle loading a baseline template
    const handleLoadBaseline = (template: TrainingTemplate) => {
        const convertedStages = templateToCurriculumStages(template);
        setStages(convertedStages);
        setCurriculumName(`${template.name} Curriculum`);
        setSelectedStage(convertedStages[0]?.name ?? null);
        setLoadedTemplateName(template.name);
        setIsTemplateGalleryOpen(false);
    };

    const handleStageKeyDown = (e: React.KeyboardEvent, stageName: string) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            setSelectedStage(stageName);
        }
    };

    return (
        <div className="page">
            <div className="page-header">
                <div className="page-header-content">
                    <h1 className="page-title">Curriculum Builder</h1>
                    <p className="page-subtitle">
                        Design multi-stage training curricula with dataset mixing
                    </p>
                </div>
                <div className="page-header-actions">
                    {loadedTemplateName && (
                        <span className="template-badge-loaded">
                            <BookOpen size={14} />
                            {loadedTemplateName}
                        </span>
                    )}
                    {saveSuccess && (
                        <span className="save-success-message">Saved successfully!</span>
                    )}
                    {saveError && (
                        <span className="save-error-message">{saveError}</span>
                    )}
                    <Button
                        variant="outline"
                        leftIcon={<BookOpen size={16} />}
                        onClick={() => setIsTemplateGalleryOpen(true)}
                    >
                        Load Baseline
                    </Button>
                    <Button variant="secondary" leftIcon={<Upload size={16} />} onClick={handleLoad}>
                        Load
                    </Button>
                    <Button variant="primary" leftIcon={<Save size={16} />} onClick={handleSave} loading={isSaving}>
                        Save
                    </Button>
                </div>
            </div>

            {/* Curriculum Name */}
            <Card variant="glass" padding="md" className="curriculum-name-card">
                <Input
                    label="Curriculum Name"
                    value={curriculumName}
                    onChange={(e) => setCurriculumName(e.target.value)}
                    placeholder="Enter a name for this curriculum"
                    fullWidth
                />
            </Card>

            <div className="curriculum-layout">
                {/* Stages List */}
                <Card padding="none" className="stages-card">
                    <CardHeader
                        title="Training Stages"
                        action={
                            <Button
                                variant="ghost"
                                size="sm"
                                leftIcon={<Plus size={14} />}
                                onClick={() => setIsAddModalOpen(true)}
                            >
                                Add Stage
                            </Button>
                        }
                    />
                    <div className="stages-list">
                        {stages.map((stage, index) => (
                            <div
                                key={stage.name}
                                className={`stage-item ${selectedStage === stage.name ? 'stage-item-active' : ''}`}
                                onClick={() => setSelectedStage(stage.name)}
                                onKeyDown={(e) => handleStageKeyDown(e, stage.name)}
                                role="button"
                                tabIndex={0}
                                aria-pressed={selectedStage === stage.name}
                            >
                                <span className="stage-grip">
                                    <GripVertical size={14} />
                                </span>
                                <span className="stage-number">{index + 1}</span>
                                <div className="stage-info">
                                    <span className="stage-name">{stage.display_name}</span>
                                    <span className="stage-meta">
                                        {stage.datasets.length} dataset{stage.datasets.length !== 1 ? 's' : ''} · {stage.epochs} epoch{stage.epochs !== 1 ? 's' : ''}
                                    </span>
                                </div>
                                <button
                                    type="button"
                                    className="stage-delete"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleDeleteStage(stage.name);
                                    }}
                                    aria-label={`Delete ${stage.display_name} stage`}
                                >
                                    <Trash2 size={14} />
                                </button>
                            </div>
                        ))}
                        {stages.length === 0 && (
                            <div className="stages-empty">
                                <p>No stages yet</p>
                                <span>Add a stage to get started</span>
                            </div>
                        )}
                    </div>
                </Card>

                {/* Stage Editor */}
                <Card padding="lg" className="editor-card">
                    {currentStage ? (
                        <>
                            <CardHeader
                                title={currentStage.display_name}
                                subtitle={`Stage: ${currentStage.name}`}
                            />
                            <CardContent>
                                <div className="editor-form">
                                    <div className="form-row">
                                        <Input
                                            label="Display Name"
                                            value={currentStage.display_name}
                                            onChange={(e) => handleUpdateStage({ display_name: e.target.value })}
                                            fullWidth
                                        />
                                        <Select
                                            label="Module Focus"
                                            options={[
                                                { value: 'language_modeling', label: 'Language Modeling' },
                                                { value: 'code_generation', label: 'Code Generation' },
                                                { value: 'instruction_tuning', label: 'Instruction Tuning' },
                                                { value: 'reasoning', label: 'Reasoning' },
                                                { value: 'chat', label: 'Chat/Dialog' },
                                                { value: 'alignment', label: 'Alignment' },
                                            ]}
                                            value={currentStage.module}
                                            onChange={(e) => handleUpdateStage({ module: e.target.value })}
                                            fullWidth
                                        />
                                    </div>

                                    <div className="form-section">
                                        <div className="datasets-header">
                                            <h4>Datasets</h4>
                                            <span className="datasets-count">
                                                {currentStage.datasets.length} added
                                            </span>
                                        </div>
                                        <div className="datasets-list">
                                            {currentStage.datasets.length > 0 ? (
                                                currentStage.datasets.map((ds, i) => (
                                                    <div key={i} className="dataset-item">
                                                        <Database size={16} className="dataset-icon-small" />
                                                        <span className="dataset-id">{ds.dataset_id}</span>
                                                        <span className="dataset-weight">×{ds.weight}</span>
                                                        <button
                                                            type="button"
                                                            className="dataset-remove"
                                                            onClick={() => {
                                                                handleUpdateStage({
                                                                    datasets: currentStage.datasets.filter((_, idx) => idx !== i),
                                                                });
                                                            }}
                                                            aria-label={`Remove ${ds.dataset_id}`}
                                                        >
                                                            <Trash2 size={14} />
                                                        </button>
                                                    </div>
                                                ))
                                            ) : (
                                                <div className="datasets-empty">
                                                    <Database size={24} className="datasets-empty-icon" />
                                                    <p>No datasets added yet</p>
                                                    <span>Add datasets from your catalog to this stage</span>
                                                </div>
                                            )}
                                        </div>
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            leftIcon={<Plus size={14} />}
                                            onClick={() => setIsDatasetModalOpen(true)}
                                        >
                                            Add Dataset
                                        </Button>
                                    </div>

                                    <div className="form-row form-row-3">
                                        <Input
                                            label="Epochs"
                                            type="number"
                                            value={currentStage.epochs}
                                            onChange={(e) => handleUpdateStage({ epochs: parseInt(e.target.value) || 1 })}
                                        />
                                        <Input
                                            label="Learning Rate"
                                            value={currentStage.learning_rate}
                                            onChange={(e) => handleUpdateStage({ learning_rate: e.target.value })}
                                        />
                                        <Input
                                            label="Batch Size"
                                            type="number"
                                            value={currentStage.batch_size}
                                            onChange={(e) => handleUpdateStage({ batch_size: parseInt(e.target.value) || 8 })}
                                        />
                                    </div>
                                </div>
                            </CardContent>
                        </>
                    ) : (
                        <div className="editor-empty">
                            <p>Select a stage to edit</p>
                        </div>
                    )}
                </Card>

                {/* JSON Preview */}
                <Card padding="lg" className="preview-card">
                    <CardHeader
                        title="JSON Preview"
                        action={
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => navigator.clipboard.writeText(JSON.stringify({ name: curriculumName, stages }, null, 2))}
                            >
                                Copy
                            </Button>
                        }
                    />
                    <CardContent>
                        <pre className="json-preview">
                            {JSON.stringify({ name: curriculumName, stages }, null, 2)}
                        </pre>
                    </CardContent>
                </Card>
            </div>

            {/* Add Stage Modal */}
            <Modal
                open={isAddModalOpen}
                onClose={() => setIsAddModalOpen(false)}
                title="Add New Stage"
                size="sm"
                footer={
                    <>
                        <Button variant="ghost" onClick={() => setIsAddModalOpen(false)}>
                            Cancel
                        </Button>
                        <Button variant="primary" onClick={handleAddStage}>
                            Create Stage
                        </Button>
                    </>
                }
            >
                <div className="modal-form">
                    <Input
                        label="Stage ID"
                        value={newStageId}
                        onChange={(e) => setNewStageId(e.target.value)}
                        hint="Lowercase, no spaces (use underscores)"
                        fullWidth
                    />
                    <Input
                        label="Display Name"
                        value={newStageName}
                        onChange={(e) => setNewStageName(e.target.value)}
                        fullWidth
                    />
                    <Select
                        label="Module Focus"
                        options={[
                            { value: 'language_modeling', label: 'Language Modeling' },
                            { value: 'code_generation', label: 'Code Generation' },
                            { value: 'instruction_tuning', label: 'Instruction Tuning' },
                            { value: 'reasoning', label: 'Reasoning' },
                            { value: 'chat', label: 'Chat/Dialog' },
                            { value: 'alignment', label: 'Alignment' },
                        ]}
                        value={newStageModule}
                        onChange={(e) => setNewStageModule(e.target.value)}
                        fullWidth
                    />
                </div>
            </Modal>

            {/* Dataset Picker Modal */}
            <Modal
                open={isDatasetModalOpen}
                onClose={() => {
                    setIsDatasetModalOpen(false);
                    setSelectedDatasetId(null);
                    setDatasetWeight(1.0);
                    setDatasetSearchQuery('');
                }}
                title="Add Dataset to Stage"
                description="Select a dataset from your catalog to add to this training stage."
                size="lg"
                footer={
                    <>
                        <Button variant="ghost" onClick={() => setIsDatasetModalOpen(false)}>
                            Cancel
                        </Button>
                        <Button
                            variant="primary"
                            onClick={handleAddDataset}
                            disabled={!selectedDatasetId}
                        >
                            Add to Stage
                        </Button>
                    </>
                }
            >
                <div className="dataset-picker">
                    {/* Search Bar */}
                    <div className="dataset-picker-search">
                        <Search size={18} className="search-icon" />
                        <input
                            type="text"
                            className="search-input"
                            placeholder="Search datasets..."
                            value={datasetSearchQuery}
                            onChange={(e) => setDatasetSearchQuery(e.target.value)}
                        />
                        <Button
                            variant="ghost"
                            size="sm"
                            leftIcon={<RefreshCw size={14} />}
                            loading={loadingDatasets}
                            onClick={loadAvailableDatasets}
                        >
                            Refresh
                        </Button>
                    </div>

                    {/* Weight Configuration */}
                    {selectedDatasetId && (
                        <div className="dataset-picker-weight">
                            <Input
                                label="Dataset Weight"
                                type="number"
                                value={datasetWeight}
                                onChange={(e) => setDatasetWeight(parseFloat(e.target.value) || 1.0)}
                                hint="Weight determines relative sampling frequency (1.0 = normal)"
                                step="0.1"
                                min="0.1"
                                max="10"
                            />
                        </div>
                    )}

                    {/* Error State */}
                    {datasetError && (
                        <div className="dataset-picker-error">
                            <AlertCircle size={16} />
                            <span>{datasetError}</span>
                        </div>
                    )}

                    {/* Dataset List */}
                    <div className="dataset-picker-list">
                        {loadingDatasets ? (
                            <div className="dataset-picker-loading">
                                <RefreshCw size={24} className="spin" />
                                <p>Loading datasets...</p>
                            </div>
                        ) : availableForSelection.length > 0 ? (
                            availableForSelection.map((dataset) => (
                                <div
                                    key={dataset.id}
                                    className={`dataset-picker-item ${selectedDatasetId === dataset.id ? 'dataset-picker-item-selected' : ''}`}
                                    onClick={() => setSelectedDatasetId(dataset.id)}
                                    role="button"
                                    tabIndex={0}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter' || e.key === ' ') {
                                            e.preventDefault();
                                            setSelectedDatasetId(dataset.id);
                                        }
                                    }}
                                >
                                    <div className="dataset-picker-item-icon">
                                        <Database size={20} />
                                    </div>
                                    <div className="dataset-picker-item-info">
                                        <span className="dataset-picker-item-name">{dataset.name}</span>
                                        <span className="dataset-picker-item-description">
                                            {dataset.description || 'No description available'}
                                        </span>
                                        <div className="dataset-picker-item-meta">
                                            <span className={`source-badge source-${dataset.source}`}>
                                                {dataset.source}
                                            </span>
                                            <span>{dataset.num_examples.toLocaleString()} examples</span>
                                        </div>
                                    </div>
                                    <div className="dataset-picker-item-check">
                                        {selectedDatasetId === dataset.id && (
                                            <div className="check-circle">✓</div>
                                        )}
                                    </div>
                                </div>
                            ))
                        ) : availableDatasets.length > 0 ? (
                            <div className="dataset-picker-empty">
                                <Database size={32} />
                                <p>All available datasets are already added to this stage</p>
                            </div>
                        ) : (
                            <div className="dataset-picker-empty">
                                <Database size={32} />
                                <h4>No datasets available</h4>
                                <p>Add datasets from the Datasets page first</p>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => window.location.href = '/datasets'}
                                >
                                    Go to Datasets
                                </Button>
                            </div>
                        )}
                    </div>
                </div>
            </Modal>

            {/* Baseline Template Gallery Modal */}
            <TemplateGallery
                open={isTemplateGalleryOpen}
                onClose={() => setIsTemplateGalleryOpen(false)}
                onSelectTemplate={handleLoadBaseline}
            />

            {/* Load Curriculum Modal */}
            <Modal
                open={isLoadModalOpen}
                onClose={() => setIsLoadModalOpen(false)}
                title="Load Curriculum"
                description="Select a saved curriculum to load."
                size="md"
            >
                <div className="curriculum-load-modal">
                    {loadingCurricula ? (
                        <div className="curriculum-loading">
                            <RefreshCw size={24} className="spin" />
                            <p>Loading saved curricula...</p>
                        </div>
                    ) : savedCurricula.length === 0 ? (
                        <div className="curriculum-empty">
                            <Database size={32} />
                            <h4>No saved curricula</h4>
                            <p>Save a curriculum first to load it here.</p>
                        </div>
                    ) : (
                        <div className="curriculum-list">
                            {savedCurricula.map((curriculum) => (
                                <div
                                    key={curriculum.id}
                                    className="curriculum-load-item"
                                    onClick={() => handleSelectCurriculum(curriculum)}
                                    role="button"
                                    tabIndex={0}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter' || e.key === ' ') {
                                            e.preventDefault();
                                            handleSelectCurriculum(curriculum);
                                        }
                                    }}
                                >
                                    <div className="curriculum-load-item-info">
                                        <span className="curriculum-load-item-name">{curriculum.name}</span>
                                        <span className="curriculum-load-item-meta">
                                            {curriculum.stages.length} stage{curriculum.stages.length !== 1 ? 's' : ''}
                                            {' · '}
                                            Updated {new Date(curriculum.updated_at).toLocaleDateString()}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </Modal>
        </div>
    );
}
