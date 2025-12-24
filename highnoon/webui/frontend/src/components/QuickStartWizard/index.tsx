// QuickStartWizard - Streamlined "2-minute to training" flow
// Combines dataset selection, auto-curriculum, and time budget selection

import { useState, useEffect, useCallback } from 'react';
import {
    Play, Zap, Clock, Sparkles, Database, Check, ChevronRight,
    RefreshCw, AlertCircle, X, Rocket
} from 'lucide-react';
import { Card, Button, Modal, ProgressBar } from '../ui';
import { datasetApi } from '../../api/client';
import type { DatasetInfo } from '../../api/types';
import './QuickStartWizard.css';

// =============================================================================
// TIME BUDGET OPTIONS
// =============================================================================

type TimeBudget = 'quick' | 'standard' | 'thorough';

interface TimeBudgetOption {
    id: TimeBudget;
    label: string;
    duration: string;
    description: string;
    trials: number;
    icon: React.ReactNode;
}

const TIME_BUDGETS: TimeBudgetOption[] = [
    {
        id: 'quick',
        label: 'Quick',
        duration: '~15 min',
        description: 'Fast exploration with 5 trials',
        trials: 5,
        icon: <Zap size={20} />,
    },
    {
        id: 'standard',
        label: 'Standard',
        duration: '~1 hour',
        description: 'Balanced search with 15 trials',
        trials: 15,
        icon: <Clock size={20} />,
    },
    {
        id: 'thorough',
        label: 'Thorough',
        duration: '~4 hours',
        description: 'Comprehensive with 40 trials',
        trials: 40,
        icon: <Sparkles size={20} />,
    },
];

// =============================================================================
// STEP INDICATOR
// =============================================================================

interface StepIndicatorProps {
    currentStep: number;
    steps: string[];
}

function StepIndicator({ currentStep, steps }: StepIndicatorProps) {
    return (
        <div className="qs-step-indicator">
            {steps.map((step, index) => (
                <div
                    key={step}
                    className={`qs-step-item ${index < currentStep ? 'qs-step-complete' : ''} ${index === currentStep ? 'qs-step-active' : ''}`}
                >
                    <div className="qs-step-number">
                        {index < currentStep ? <Check size={14} /> : index + 1}
                    </div>
                    <span className="qs-step-label">{step}</span>
                    {index < steps.length - 1 && <ChevronRight size={16} className="qs-step-arrow" />}
                </div>
            ))}
        </div>
    );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

interface QuickStartWizardProps {
    open: boolean;
    onClose: () => void;
    onComplete?: (jobId: string) => void;
}

export function QuickStartWizard({ open, onClose, onComplete }: QuickStartWizardProps) {
    // Wizard state
    const [currentStep, setCurrentStep] = useState(0);
    const steps = ['Dataset', 'Configure', 'Launch'];

    // Dataset step
    const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
    const [loadingDatasets, setLoadingDatasets] = useState(false);
    const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
    const [datasetError, setDatasetError] = useState<string | null>(null);

    // Config step
    const [timeBudget, setTimeBudget] = useState<TimeBudget>('standard');
    const [modelName, setModelName] = useState('');

    // Launch step
    const [isLaunching, setIsLaunching] = useState(false);
    const [launchError, setLaunchError] = useState<string | null>(null);

    // Load datasets when wizard opens
    useEffect(() => {
        if (open) {
            loadDatasets();
        }
    }, [open]);

    const loadDatasets = async () => {
        setLoadingDatasets(true);
        setDatasetError(null);
        try {
            const data = await datasetApi.list();
            setDatasets(data);
            // Auto-select first if only one
            if (data.length === 1) {
                setSelectedDataset(data[0].id);
            }
        } catch (err) {
            setDatasetError(err instanceof Error ? err.message : 'Failed to load datasets');
        } finally {
            setLoadingDatasets(false);
        }
    };

    const handleNext = () => {
        if (currentStep < steps.length - 1 && canProceed()) {
            setCurrentStep(currentStep + 1);
        }
    };

    const handleBack = () => {
        if (currentStep > 0) {
            setCurrentStep(currentStep - 1);
            setLaunchError(null);
        }
    };

    const canProceed = () => {
        if (currentStep === 0) return !!selectedDataset;
        return true;
    };

    const handleLaunch = async () => {
        setIsLaunching(true);
        setLaunchError(null);

        try {
            const selectedTimeBudget = TIME_BUDGETS.find(t => t.id === timeBudget)!;

            // For QuickStart, we'll log the config and redirect to HPO page
            // In a full implementation, this would create an auto-curriculum and start HPO
            console.log('QuickStart launching with:', {
                dataset: selectedDataset,
                timeBudget: selectedTimeBudget,
                modelName: modelName || 'highnoon-model',
            });

            // Success - notify parent and redirect to HPO
            if (onComplete) {
                onComplete('quickstart-' + Date.now());
            }
            handleClose();

            // Redirect to HPO page for now
            window.location.href = '/hpo';
        } catch (err) {
            setLaunchError(err instanceof Error ? err.message : 'Failed to start training');
        } finally {
            setIsLaunching(false);
        }
    };

    const handleClose = () => {
        // Reset state
        setCurrentStep(0);
        setSelectedDataset(null);
        setTimeBudget('standard');
        setModelName('');
        setLaunchError(null);
        onClose();
    };

    const selectedTimeBudget = TIME_BUDGETS.find(t => t.id === timeBudget)!;
    const selectedDs = datasets.find(d => d.id === selectedDataset);

    return (
        <Modal
            open={open}
            onClose={handleClose}
            title=""
            size="lg"
        >
            <div className="qs-wizard">
                <div className="qs-wizard-header">
                    <div className="qs-wizard-icon">
                        <Rocket size={24} />
                    </div>
                    <div className="qs-wizard-title">
                        <h2>Quick Start</h2>
                        <p>Train a model in 3 simple steps</p>
                    </div>
                    <button className="qs-close" onClick={handleClose}>
                        <X size={20} />
                    </button>
                </div>

                <StepIndicator currentStep={currentStep} steps={steps} />

                <div className="qs-content">
                    {/* Step 1: Dataset Selection */}
                    {currentStep === 0 && (
                        <div className="qs-step">
                            <h3 className="qs-step-title">Select your dataset</h3>
                            <p className="qs-step-desc">
                                Choose a dataset to train on. We'll auto-generate an optimized curriculum.
                            </p>

                            {loadingDatasets ? (
                                <div className="qs-loading">
                                    <RefreshCw size={24} className="spin" />
                                    <span>Loading datasets...</span>
                                </div>
                            ) : datasetError ? (
                                <div className="qs-error">
                                    <AlertCircle size={20} />
                                    <span>{datasetError}</span>
                                    <Button size="sm" onClick={loadDatasets}>Retry</Button>
                                </div>
                            ) : datasets.length === 0 ? (
                                <div className="qs-empty">
                                    <Database size={32} />
                                    <p>No datasets available</p>
                                    <span>Add datasets from the Datasets page first</span>
                                    <Button variant="primary" onClick={() => window.location.href = '/datasets'}>
                                        Go to Datasets
                                    </Button>
                                </div>
                            ) : (
                                <div className="qs-dataset-grid">
                                    {datasets.map((ds) => (
                                        <button
                                            key={ds.id}
                                            className={`qs-dataset-card ${selectedDataset === ds.id ? 'qs-dataset-card-selected' : ''}`}
                                            onClick={() => setSelectedDataset(ds.id)}
                                        >
                                            <div className="qs-dataset-icon">
                                                <Database size={20} />
                                            </div>
                                            <div className="qs-dataset-info">
                                                <span className="qs-dataset-name">{ds.name}</span>
                                                <span className="qs-dataset-meta">
                                                    {ds.num_examples.toLocaleString()} examples
                                                </span>
                                            </div>
                                            {selectedDataset === ds.id && (
                                                <div className="qs-dataset-check">
                                                    <Check size={16} />
                                                </div>
                                            )}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Step 2: Configuration */}
                    {currentStep === 1 && (
                        <div className="qs-step">
                            <h3 className="qs-step-title">Configure training</h3>
                            <p className="qs-step-desc">
                                Choose how long to optimize and optionally name your model.
                            </p>

                            <div className="qs-config-section">
                                <label className="qs-config-label">Training Time</label>
                                <div className="qs-time-budget-grid">
                                    {TIME_BUDGETS.map((budget) => (
                                        <button
                                            key={budget.id}
                                            className={`qs-time-budget ${timeBudget === budget.id ? 'qs-time-budget-selected' : ''}`}
                                            onClick={() => setTimeBudget(budget.id)}
                                        >
                                            <div className="qs-time-budget-icon">{budget.icon}</div>
                                            <div className="qs-time-budget-info">
                                                <span className="qs-time-budget-label">{budget.label}</span>
                                                <span className="qs-time-budget-duration">{budget.duration}</span>
                                            </div>
                                            <span className="qs-time-budget-desc">{budget.description}</span>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="qs-config-section">
                                <label className="qs-config-label">Model Name (optional)</label>
                                <input
                                    type="text"
                                    className="qs-input"
                                    placeholder="my-custom-model"
                                    value={modelName}
                                    onChange={(e) => setModelName(e.target.value)}
                                />
                                <span className="qs-input-hint">Leave blank for auto-generated name</span>
                            </div>
                        </div>
                    )}

                    {/* Step 3: Review & Launch */}
                    {currentStep === 2 && (
                        <div className="qs-step">
                            <h3 className="qs-step-title">Ready to launch</h3>
                            <p className="qs-step-desc">
                                Review your configuration and start training.
                            </p>

                            <div className="qs-review">
                                <div className="qs-review-item">
                                    <span className="qs-review-label">Dataset</span>
                                    <span className="qs-review-value">{selectedDs?.name}</span>
                                </div>
                                <div className="qs-review-item">
                                    <span className="qs-review-label">Examples</span>
                                    <span className="qs-review-value">{selectedDs?.num_examples.toLocaleString()}</span>
                                </div>
                                <div className="qs-review-item">
                                    <span className="qs-review-label">Duration</span>
                                    <span className="qs-review-value">{selectedTimeBudget.duration}</span>
                                </div>
                                <div className="qs-review-item">
                                    <span className="qs-review-label">HPO Trials</span>
                                    <span className="qs-review-value">{selectedTimeBudget.trials}</span>
                                </div>
                                <div className="qs-review-item">
                                    <span className="qs-review-label">Model Name</span>
                                    <span className="qs-review-value">{modelName || 'highnoon-model (auto)'}</span>
                                </div>
                            </div>

                            <div className="qs-auto-note">
                                <Sparkles size={16} />
                                <span>Learning rate, batch size, optimizer, and curriculum will be auto-optimized</span>
                            </div>

                            {launchError && (
                                <div className="qs-error">
                                    <AlertCircle size={16} />
                                    <span>{launchError}</span>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Navigation */}
                <div className="qs-nav">
                    {currentStep > 0 && (
                        <Button variant="ghost" onClick={handleBack} disabled={isLaunching}>
                            Back
                        </Button>
                    )}
                    <div className="qs-nav-spacer" />
                    {currentStep < steps.length - 1 ? (
                        <Button variant="primary" onClick={handleNext} disabled={!canProceed()}>
                            Continue
                        </Button>
                    ) : (
                        <Button
                            variant="primary"
                            leftIcon={<Rocket size={16} />}
                            onClick={handleLaunch}
                            loading={isLaunching}
                        >
                            Launch Training
                        </Button>
                    )}
                </div>
            </div>
        </Modal>
    );
}

export default QuickStartWizard;
