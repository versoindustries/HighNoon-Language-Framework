// HighNoon Dashboard - HPO Orchestrator Page (Enterprise UX)
// Tiered licensing with progressive disclosure
// Phase-based UI: Setup Mode -> Dashboard Mode when optimization starts
import { useState, useCallback, useEffect, createContext, useContext, useRef } from 'react';
import {
    Play, Pause, Square, Download, RefreshCw, Check, Clock, Zap, Sparkles,
    ChevronRight, ChevronDown, Info, Lock, Crown, Settings, Cpu, Target,
    Layers, AlertTriangle, BarChart3, ArrowLeft
} from 'lucide-react';
import {
    Card,
    CardHeader,
    CardContent,
    Button,
    Select,
    ProgressBar,
    Modal,
} from '../components/ui';
import { curriculumApi, hpoApi } from '../api/client';
import { createHPOWebSocket } from '../api/websocket';
import type {
    HPOSweepInfo,
    HPOTrialInfo,
    Curriculum,
    WebSocketMessage,
} from '../api/types';
import { TrainingConsole } from '../components/TrainingConsole';
import { HPODashboard } from '../components/HPODashboard';
import './HPO.css';

// =============================================================================
// LICENSE TIERS - Lite edition supports full 5M context per GEMINI.md
// =============================================================================

type LicenseTier = 'lite' | 'pro' | 'enterprise';

interface TierLimits {
    maxVocabSize: number;
    maxContextWindow: number;
    maxTrials: number;
    allowedStrategies: string[];
    allowedOptimizers: string[];
    advancedOptions: boolean;
}

const TIER_LIMITS: Record<LicenseTier, TierLimits> = {
    lite: {
        maxVocabSize: 256000,      // 256K vocab limit for Lite
        maxContextWindow: 5000000, // 5M context - FULL Lite capability
        maxTrials: 20,
        allowedStrategies: ['bayesian', 'random', 'hyperband', 'successive_halving', 'pbt'],
        allowedOptimizers: ['sophiag', 'qiao', 'grover', 'sympflow', 'sympflowqng'],  // Quantum optimizers: SophiaG, QIAO, Grover-Q, SympFlow, SympFlowQNG
        advancedOptions: true,  // Enabled for HPO enhancements
    },
    pro: {
        maxVocabSize: 128000,
        maxContextWindow: 5000000,
        maxTrials: 100,
        allowedStrategies: ['bayesian', 'random', 'hyperband', 'successive_halving', 'pbt'],
        allowedOptimizers: ['sophiag', 'qiao', 'grover', 'sympflow', 'sympflowqng', 'adamw', 'lion'],  // Quantum + standard optimizers
        advancedOptions: true,
    },
    enterprise: {
        maxVocabSize: 500000,
        maxContextWindow: 5000000,
        maxTrials: 999,
        allowedStrategies: ['bayesian', 'random', 'hyperband', 'successive_halving', 'pbt', 'custom'],
        allowedOptimizers: ['sophiag', 'qiao', 'grover', 'sympflow', 'sympflowqng', 'adamw', 'adam', 'lion', 'custom'],  // Full optimizer suite
        advancedOptions: true,
    },
};

// License context (would come from app-level auth in production)
const LicenseContext = createContext<LicenseTier>('lite');

// =============================================================================
// CONSTANTS
// =============================================================================

// HPO always runs until convergence - no manual mode selection needed


// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function formatNumber(num: number): string {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(0)}K`;
    return num.toString();
}

function getTierBadge(tier: LicenseTier) {
    if (tier === 'lite') return null;
    return (
        <span className={`tier-badge tier-badge-${tier}`}>
            <Crown size={12} />
            {tier.charAt(0).toUpperCase() + tier.slice(1)}
        </span>
    );
}

// Vocab size slider config (max: 256K for expanded tokenizer support)
const VOCAB_STOPS = [8000, 16000, 32000, 50000, 65536, 100000, 128000, 256000];
const CONTEXT_STOPS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 5000000];

// Parameter budget stops (100M to 20B) - expanded for fine-grained control
const PARAM_STOPS = [
    100_000_000,    // 100M
    250_000_000,    // 250M
    500_000_000,    // 500M
    750_000_000,    // 750M
    1_000_000_000,  // 1B
    1_500_000_000,  // 1.5B
    2_000_000_000,  // 2B
    3_000_000_000,  // 3B
    5_000_000_000,  // 5B
    7_000_000_000,  // 7B
    10_000_000_000, // 10B
    13_000_000_000, // 13B
    16_000_000_000, // 16B
    20_000_000_000, // 20B (Lite limit)
];

// =============================================================================
// LITE EDITION ARCHITECTURE LIMITS
// =============================================================================

const LITE_LIMITS = {
    maxParams: 20_000_000_000,      // 20B parameters
    maxReasoningBlocks: 24,
    maxMoEExperts: 12,
    maxEmbeddingDim: 4096,
    maxSuperpositionDim: 2,         // Max 2 for Lite edition
    maxContextWindow: 5_000_000,
    maxVocabSize: 65536,
    maxMambaStateDim: 128,
};

// Architecture dropdown options
const REASONING_BLOCK_OPTIONS = [2, 4, 6, 8, 12, 16, 20, 24];
const MOE_EXPERT_OPTIONS = [2, 4, 6, 8, 12];
const EMBEDDING_DIM_OPTIONS = [256, 512, 768, 1024, 2048, 4096];
const SUPERPOSITION_DIM_OPTIONS = [1, 2]; // Max 2 for Lite
const MAMBA_STATE_DIM_OPTIONS = [16, 32, 64, 128];
const WLAM_HEADS_OPTIONS = [4, 8, 12, 16];

// =============================================================================
// ARCHITECTURE CONFIG TYPES
// =============================================================================

interface ArchitectureConfig {
    numReasoningBlocks: number;
    numMoEExperts: number;
    embeddingDim: number;
    superpositionDim: number;
    mambaStateDim: number;
    wlamHeads: number;
    vocabSize: number;
    contextWindow: number;
}

interface ParameterBreakdown {
    embedding: number;
    spatialBlocks: number;
    timeCrystal: number;
    latentReasoning: number;
    wlam: number;
    moe: number;
    output: number;
    total: number;
}

// =============================================================================
// PARAMETER CALCULATION
// =============================================================================

/**
 * Calculate model parameters based on HSMN architecture.
 * Uses the 6-block pattern: SpatialBlock, TimeCrystal, LatentReasoning, SpatialBlock, WLAM, MoE
 */
function calculateModelParams(config: ArchitectureConfig): ParameterBreakdown {
    const { numReasoningBlocks, numMoEExperts, embeddingDim, mambaStateDim, vocabSize } = config;

    // Embedding parameters
    const vocabEmbedding = vocabSize * embeddingDim;

    // Number of complete 6-block patterns
    const blocksPerPattern = 6;
    const numPatterns = Math.ceil(numReasoningBlocks / blocksPerPattern);

    // Per-block parameter calculations
    const dInner = embeddingDim * 2; // expand_factor=2
    const ffDim = embeddingDim * 4;  // ff_expansion=4

    // SpatialBlock (Mamba-2): input proj + conv + SSM params + output proj
    // ~4 * d_model * d_inner + state params
    const spatialParams = 4 * embeddingDim * dInner + mambaStateDim * embeddingDim * 2;

    // TimeCrystalSequenceBlock: HNN weights (W1, W2, W3), output proj, VQC
    // ~8 * embedding_dim² for the full cell
    const timeCrystalParams = 8 * embeddingDim * embeddingDim;

    // LatentReasoningBlock: FFN layers with expansion
    // ~3 * embedding_dim * ff_dim
    const latentParams = 3 * embeddingDim * ffDim;

    // WLAMBlock: wavelet transforms + attention
    // ~4 * embedding_dim²
    const wlamParams = 4 * embeddingDim * embeddingDim;

    // MoELayer: num_experts * (2 * d_model * ff_dim) + router
    const moeParams = numMoEExperts * (2 * embeddingDim * ffDim) + embeddingDim * numMoEExperts;

    // Total per pattern (2 spatial blocks per pattern)
    const paramsPerPattern = spatialParams * 2 + timeCrystalParams + latentParams + wlamParams + moeParams;

    // Output layer (LM head)
    const outputParams = embeddingDim * vocabSize;

    // Calculate per-type totals
    const totalPatterns = Math.min(numPatterns, Math.ceil(numReasoningBlocks / blocksPerPattern));

    return {
        embedding: vocabEmbedding,
        spatialBlocks: spatialParams * 2 * totalPatterns,
        timeCrystal: timeCrystalParams * totalPatterns,
        latentReasoning: latentParams * totalPatterns,
        wlam: wlamParams * totalPatterns,
        moe: moeParams * totalPatterns,
        output: outputParams,
        total: vocabEmbedding + paramsPerPattern * totalPatterns + outputParams,
    };
}

/**
 * Format large numbers with B/M/K suffixes
 */
function formatParams(num: number): string {
    if (num >= 1_000_000_000) return `${(num / 1_000_000_000).toFixed(2)}B`;
    if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
    if (num >= 1_000) return `${(num / 1_000).toFixed(0)}K`;
    return num.toString();
}

/**
 * Get status color based on parameter usage
 */
function getParamStatus(total: number): 'safe' | 'warning' | 'danger' {
    const ratio = total / LITE_LIMITS.maxParams;
    if (ratio > 1) return 'danger';
    if (ratio > 0.8) return 'warning';
    return 'safe';
}

// =============================================================================
// SLIDER COMPONENT
// =============================================================================

interface LogSliderProps {
    label: string;
    value: number;
    stops: number[];
    maxValue: number;
    onChange: (value: number) => void;
    formatValue?: (value: number) => string;
}

function LogSlider({ label, value, stops, maxValue, onChange, formatValue = formatNumber }: LogSliderProps) {
    const availableStops = stops.filter(s => s <= maxValue);
    const currentIndex = availableStops.findIndex(s => s >= value) !== -1
        ? availableStops.findIndex(s => s >= value)
        : availableStops.length - 1;

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const index = parseInt(e.target.value);
        onChange(availableStops[index]);
    };

    return (
        <div className="log-slider">
            <div className="log-slider-header">
                <label className="log-slider-label">{label}</label>
                <span className="log-slider-value">{formatValue(value)}</span>
            </div>
            <input
                type="range"
                min={0}
                max={availableStops.length - 1}
                value={currentIndex}
                onChange={handleChange}
                className="log-slider-input"
            />
            <div className="log-slider-range">
                <span>{formatValue(availableStops[0])}</span>
                <span>{formatValue(availableStops[availableStops.length - 1])}</span>
            </div>
        </div>
    );
}

// =============================================================================
// COMPONENTS
// =============================================================================

interface StepIndicatorProps {
    currentStep: number;
    steps: string[];
}

function StepIndicator({ currentStep, steps }: StepIndicatorProps) {
    return (
        <div className="step-indicator">
            {steps.map((step, index) => (
                <div
                    key={step}
                    className={`step-item ${index < currentStep ? 'step-complete' : ''} ${index === currentStep ? 'step-active' : ''}`}
                >
                    <div className="step-number">
                        {index < currentStep ? <Check size={14} /> : index + 1}
                    </div>
                    <span className="step-label">{step}</span>
                    {index < steps.length - 1 && <ChevronRight size={16} className="step-arrow" />}
                </div>
            ))}
        </div>
    );
}

interface UpgradeModalProps {
    isOpen: boolean;
    onClose: () => void;
    feature: string;
    requiredTier: 'pro' | 'enterprise';
}

function UpgradeModal({ isOpen, onClose, feature, requiredTier }: UpgradeModalProps) {
    return (
        <Modal open={isOpen} onClose={onClose} title={`Upgrade to ${requiredTier.charAt(0).toUpperCase() + requiredTier.slice(1)}`}>
            <div className="upgrade-modal-content">
                <div className="upgrade-feature-highlight">
                    <Crown size={32} className="upgrade-icon" />
                    <h3>Unlock {feature}</h3>
                    <p>
                        {requiredTier === 'pro'
                            ? 'Get access to advanced hyperparameter controls and more optimizers.'
                            : 'Unlock custom optimizers, API access, and enterprise support.'}
                    </p>
                </div>
                <div className="upgrade-actions">
                    <Button variant="primary" onClick={() => window.open('/pricing', '_blank')}>
                        View Plans
                    </Button>
                    <Button variant="ghost" onClick={onClose}>
                        Maybe Later
                    </Button>
                </div>
            </div>
        </Modal>
    );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function HPO() {
    // For demo, simulate license tier (would come from auth context in production)
    const [currentTier] = useState<LicenseTier>('lite');
    const limits = TIER_LIMITS[currentTier];

    // Curriculum data from API (user-created + presets)
    const [curricula, setCurricula] = useState<Curriculum[]>([]);
    const [loadingCurricula, setLoadingCurricula] = useState(true);

    // Fetch both user curricula and preset curricula
    useEffect(() => {
        async function fetchCurricula() {
            try {
                // Fetch user-created curricula
                const userCurricula = await curriculumApi.list();

                // Fetch preset curricula
                let presetCurricula: Curriculum[] = [];
                try {
                    const presets = await curriculumApi.listPresets();
                    presetCurricula = presets.map(p => ({
                        id: p.id,
                        name: `[Preset] ${p.name}`,
                        stages: p.stages || [],
                        created_at: '',
                        updated_at: '',
                    }));
                } catch (presetErr) {
                    console.warn('Could not load presets, using user curricula only:', presetErr);
                }

                // Combine: presets first, then user curricula
                setCurricula([...presetCurricula, ...userCurricula]);
            } catch (err) {
                console.error('Failed to load curricula:', err);
            } finally {
                setLoadingCurricula(false);
            }
        }
        fetchCurricula();
    }, []);

    // Wizard state
    const [currentStep, setCurrentStep] = useState(0);
    const [selectedCurriculum, setSelectedCurriculum] = useState<string>('');

    // Model configuration - Context window slider (vocab is auto-determined)
    const [contextWindow, setContextWindow] = useState(4096);

    // Parameter budget - HPO will find optimal architecture within this constraint
    const [paramBudget, setParamBudget] = useState(1_000_000_000); // Default 1B

    // Advanced options (Pro+)
    const [advancedOpen, setAdvancedOpen] = useState(false);
    const [searchStrategy] = useState('quantum');  // QAHPO is the only strategy
    const [selectedOptimizers, setSelectedOptimizers] = useState<string[]>(['sophiag']);

    // Learning rate is now auto-tuned based on optimizer selection
    // (handled by backend in hpo_manager.py using OPTIMIZER_LR_RANGES)

    // Sweep state
    const [sweepStatus, setSweepStatus] = useState<HPOSweepInfo | null>(null);
    const [trials, setTrials] = useState<HPOTrialInfo[]>([]);
    const [isRunning, setIsRunning] = useState(false);

    // Budget enforcement stats
    const [budgetStats, setBudgetStats] = useState<{
        enabled: boolean;
        param_budget: number;
        total_skipped: number;
        skip_rate: number;
        safe_bounds?: { max_embedding_dim: number; max_reasoning_blocks: number; max_moe_experts: number };
    } | null>(null);

    // Modal state
    const [upgradeModal, setUpgradeModal] = useState<{ open: boolean; feature: string; tier: 'pro' | 'enterprise' }>({
        open: false, feature: '', tier: 'pro'
    });

    // Dev mode state for verbose logging
    const [devMode, setDevMode] = useState(false);

    // Dashboard mode: 'setup' shows wizard, 'dashboard' shows live monitoring
    const [dashboardMode, setDashboardMode] = useState<'setup' | 'dashboard'>('setup');

    // Settings modal for accessing config from dashboard mode
    const [settingsModalOpen, setSettingsModalOpen] = useState(false);

    // WebSocket connection ref for sweep tracking
    const wsRef = useRef<{ close: () => void } | null>(null);

    // Fetch dev mode status on mount
    useEffect(() => {
        async function checkDevMode() {
            try {
                const response = await fetch('/api/dev/status');
                if (response.ok) {
                    const data = await response.json();
                    setDevMode(data.dev_mode || data.debug_build);
                }
            } catch (err) {
                console.log('Dev mode check failed:', err);
            }
        }
        checkDevMode();
    }, []);

    // Track sweep progress via WebSocket when running
    useEffect(() => {
        if (!sweepStatus?.sweep_id || !isRunning) {
            return;
        }

        console.log('[HPO] Connecting to WebSocket for sweep:', sweepStatus.sweep_id);

        const handleMessage = (message: WebSocketMessage) => {
            console.log('[HPO] WebSocket message:', message);

            if (message.type === 'trial_update') {
                // Update trials list
                const trialData = message.data;
                setTrials(prev => {
                    const existing = prev.findIndex(t => t.trial_id === trialData.trial_id);
                    if (existing >= 0) {
                        const updated = [...prev];
                        updated[existing] = trialData;
                        return updated;
                    }
                    return [...prev, trialData];
                });

                // Update sweep status with completed count
                setSweepStatus(prev => prev ? {
                    ...prev,
                    completed_trials: (prev.completed_trials || 0) + 1,
                } as HPOSweepInfo : null);
            } else if (message.type === 'sweep_update') {
                // Full sweep status update (backend sends 'sweep_update')
                const data = message.data;
                setSweepStatus(prev => {
                    if (!prev) return null;
                    return {
                        ...prev,
                        ...data,
                    } as HPOSweepInfo;
                });
            } else if (message.type === 'finished') {
                console.log('[HPO] Sweep completed:', message.state);
                setIsRunning(false);
                setSweepStatus(prev => prev ? { ...prev, state: message.state } as HPOSweepInfo : null);
            } else if (message.type === 'error') {
                console.error('[HPO] Sweep error:', message.message);
                setIsRunning(false);
            }
        };

        const handleStatus = (connected: boolean) => {
            console.log('[HPO] WebSocket connected:', connected);
        };

        wsRef.current = createHPOWebSocket(sweepStatus.sweep_id, handleMessage, handleStatus);

        // Cleanup on unmount or when sweep changes
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, [sweepStatus?.sweep_id, isRunning]);

    // Poll for sweep status as fallback when WebSocket may not be available
    useEffect(() => {
        if (!sweepStatus?.sweep_id || !isRunning) {
            return;
        }

        const pollInterval = setInterval(async () => {
            try {
                const status = await hpoApi.getSweepStatus(sweepStatus.sweep_id);
                setSweepStatus(status);

                // Also fetch trials
                const trialList = await hpoApi.getTrials(sweepStatus.sweep_id);
                setTrials(trialList);

                // Fetch budget stats
                try {
                    const stats = await hpoApi.getBudgetStats(sweepStatus.sweep_id);
                    setBudgetStats(stats);
                } catch (budgetErr) {
                    // Budget stats may not be available for all sweeps
                    console.debug('[HPO] Budget stats not available:', budgetErr);
                }

                // Check if sweep is done
                if (status.state === 'completed' || status.state === 'failed' || status.state === 'cancelled') {
                    setIsRunning(false);
                }
            } catch (err) {
                console.error('[HPO] Failed to poll sweep status:', err);
            }
        }, 5000); // Poll every 5 seconds

        return () => clearInterval(pollInterval);
    }, [sweepStatus?.sweep_id, isRunning]);

    const steps = ['Curriculum', 'Configure', 'Start'];
    const curriculum = curricula.find(c => c.id === selectedCurriculum) || null;

    const canProceed = () => {
        if (currentStep === 0) return !!selectedCurriculum;
        return true;
    };

    const handleNext = () => {
        if (currentStep < steps.length - 1 && canProceed()) {
            setCurrentStep(currentStep + 1);
        }
    };

    const handleBack = () => {
        if (currentStep > 0) {
            setCurrentStep(currentStep - 1);
        }
    };

    const handleStartSweep = async () => {
        setIsRunning(true);
        try {
            const response = await fetch('/api/hpo/sweep/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    curriculum_id: selectedCurriculum,
                    // HPO runs until convergence - no manual limits
                    auto_stop_on_convergence: true,
                    search_strategy: searchStrategy,
                    optimizers: selectedOptimizers,
                    // Tokenizer configuration - vocab_size is auto-determined by IntelligentVocabController
                    context_window: contextWindow,
                    use_quantum_tokenizer: true,  // Enable automatic vocab sizing
                    use_hyperdimensional_embedding: true,  // Use HDE for memory efficiency
                    // Parameter budget constraint - HPO will auto-discover optimal architecture
                    param_budget: paramBudget,
                }),
            });
            const data = await response.json();
            console.log('HPO sweep started:', data);
            if (data.sweep_id) {
                setSweepStatus({
                    sweep_id: data.sweep_id,
                    stage: 'coarse' as const,
                    state: 'running',
                    max_trials: data.max_trials,
                    completed_trials: 0,
                } as unknown as HPOSweepInfo);
                // Switch to dashboard mode for unified monitoring view
                setDashboardMode('dashboard');
            }
        } catch (err) {
            console.error('Failed to start HPO sweep:', err);
            setIsRunning(false);
            setDashboardMode('setup');
        }
    };

    const handleStopSweep = async () => {
        setIsRunning(false);
    };

    const showUpgradeModal = useCallback((feature: string, tier: 'pro' | 'enterprise') => {
        setUpgradeModal({ open: true, feature, tier });
    }, []);

    return (
        <LicenseContext.Provider value={currentTier}>
            <div className="page">
                {/* Dashboard Mode - Unified monitoring interface */}
                {dashboardMode === 'dashboard' ? (
                    <>
                        <div className="page-header">
                            <div className="page-header-content">
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    leftIcon={<ArrowLeft size={16} />}
                                    onClick={() => setDashboardMode('setup')}
                                    className="hpo-back-btn"
                                >
                                    Back to Setup
                                </Button>
                                <h1 className="page-title">
                                    Smart Tuner
                                    {getTierBadge(currentTier)}
                                </h1>
                            </div>
                        </div>
                        <HPODashboard
                            sweepStatus={sweepStatus}
                            trials={trials}
                            isRunning={isRunning}
                            devMode={devMode}
                            onPause={() => {/* TODO: Implement pause */ }}
                            onStop={handleStopSweep}
                            onShowSettings={() => setSettingsModalOpen(true)}
                            config={{
                                curriculumName: curriculum?.name,
                                paramBudget: paramBudget,
                                optimizer: selectedOptimizers[0] === 'sophiag' ? 'SophiaG' : selectedOptimizers[0],
                                optimizationMode: 'Until Convergence',
                            }}
                        />
                    </>
                ) : (
                    /* Setup Mode - Configuration wizard */
                    <>
                        <div className="page-header">
                            <div className="page-header-content">
                                <h1 className="page-title">
                                    Smart Tuner
                                    {getTierBadge(currentTier)}
                                </h1>
                                <p className="page-subtitle">
                                    Automatically find the best hyperparameters for your model
                                </p>
                            </div>
                            <div className="page-header-actions">
                                {isRunning && (
                                    <>
                                        <Button variant="secondary" leftIcon={<Pause size={16} />}>Pause</Button>
                                        <Button variant="danger" leftIcon={<Square size={16} />} onClick={handleStopSweep}>Stop</Button>
                                    </>
                                )}
                            </div>
                        </div>

                        <div className="hpo-layout-simple">
                            {/* Left Column: Wizard */}
                            <div className="hpo-wizard-column">
                                <Card variant="glass" padding="none">
                                    <CardHeader title="Setup" subtitle="Configure optimization in a few simple steps" />
                                    <CardContent>
                                        <StepIndicator currentStep={currentStep} steps={steps} />

                                        <div className="wizard-content">
                                            {/* Step 1: Curriculum Selection */}
                                            {currentStep === 0 && (
                                                <div className="wizard-step">
                                                    <h3 className="wizard-step-title">Select your curriculum</h3>
                                                    <p className="wizard-step-desc">
                                                        Choose the curriculum to optimize hyperparameters for
                                                    </p>
                                                    <div className="curriculum-options">
                                                        {loadingCurricula ? (
                                                            <div className="curriculum-loading">Loading curricula...</div>
                                                        ) : curricula.length === 0 ? (
                                                            <div className="curriculum-empty">
                                                                <Info size={24} />
                                                                <p>No curricula found. Create one in the Curriculum tab first.</p>
                                                            </div>
                                                        ) : (
                                                            curricula.map((curr) => (
                                                                <label
                                                                    key={curr.id}
                                                                    className={`curriculum-option ${selectedCurriculum === curr.id ? 'curriculum-option-selected' : ''}`}
                                                                >
                                                                    <input
                                                                        type="radio"
                                                                        name="curriculum"
                                                                        value={curr.id}
                                                                        checked={selectedCurriculum === curr.id}
                                                                        onChange={() => setSelectedCurriculum(curr.id)}
                                                                    />
                                                                    <div className="curriculum-option-content">
                                                                        <span className="curriculum-option-name">{curr.name}</span>
                                                                        <div className="curriculum-option-meta">
                                                                            <span>{curr.stages.length} stages</span>
                                                                        </div>
                                                                    </div>
                                                                </label>
                                                            ))
                                                        )}
                                                    </div>
                                                </div>
                                            )}

                                            {/* Step 2: Configuration (Model + Time + Advanced) */}
                                            {currentStep === 1 && (
                                                <div className="wizard-step">
                                                    <h3 className="wizard-step-title">Configure your model</h3>
                                                    <p className="wizard-step-desc">
                                                        Set model parameters and optimization time
                                                    </p>

                                                    {/* Model Configuration - SLIDER BARS */}
                                                    <div className="config-section">
                                                        <div className="config-section-header">
                                                            <Cpu size={18} />
                                                            <span>Tokenizer Settings</span>
                                                        </div>
                                                        <div className="slider-group">
                                                            {/* Vocabulary Info - Auto-determined by Quantum Tokenizer */}
                                                            <div className="config-info-box" style={{ marginBottom: '1rem', padding: '0.75rem', background: 'var(--bg-subtle)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                                                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                                                                    <Sparkles size={14} style={{ color: 'var(--accent-primary)' }} />
                                                                    <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>Vocabulary Size: Automatic</span>
                                                                </div>
                                                                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                                                                    Determined by IntelligentVocabController (~362 base + learned n-grams)
                                                                </span>
                                                            </div>
                                                            <LogSlider
                                                                label="Context Window"
                                                                value={contextWindow}
                                                                stops={CONTEXT_STOPS}
                                                                maxValue={limits.maxContextWindow}
                                                                onChange={setContextWindow}
                                                            />
                                                        </div>

                                                        {/* Parameter Budget - HPO finds optimal architecture within this constraint */}
                                                        <div className="config-section config-section-architecture">
                                                            <div className="config-section-header">
                                                                <Layers size={18} />
                                                                <span>Parameter Budget</span>
                                                                <span className="param-indicator">
                                                                    <span className="param-count param-count-safe">
                                                                        {formatParams(paramBudget)} max
                                                                    </span>
                                                                </span>
                                                            </div>
                                                            <div className="param-budget-content">
                                                                <p className="budget-explanation">
                                                                    Set your target model size. HPO will automatically discover the optimal
                                                                    architecture (reasoning blocks, MoE experts, embedding dimension, etc.)
                                                                    within this parameter budget.
                                                                </p>
                                                                <LogSlider
                                                                    label="Maximum Parameters"
                                                                    value={paramBudget}
                                                                    stops={PARAM_STOPS}
                                                                    maxValue={LITE_LIMITS.maxParams}
                                                                    onChange={setParamBudget}
                                                                    formatValue={formatParams}
                                                                />
                                                                <div className="budget-presets">
                                                                    <span className="budget-preset-label">Quick Presets:</span>
                                                                    <button className="budget-preset-btn" onClick={() => setParamBudget(500_000_000)}>500M</button>
                                                                    <button className="budget-preset-btn" onClick={() => setParamBudget(1_000_000_000)}>1B</button>
                                                                    <button className="budget-preset-btn" onClick={() => setParamBudget(3_000_000_000)}>3B</button>
                                                                    <button className="budget-preset-btn" onClick={() => setParamBudget(7_000_000_000)}>7B</button>
                                                                    <button className="budget-preset-btn" onClick={() => setParamBudget(13_000_000_000)}>13B</button>
                                                                    <button className="budget-preset-btn" onClick={() => setParamBudget(20_000_000_000)}>20B</button>
                                                                </div>
                                                            </div>
                                                        </div>

                                                        {/* Advanced Options - Collapsible */}
                                                        <div className="config-section config-section-advanced">
                                                            <button
                                                                className="config-section-header config-section-toggle"
                                                                onClick={() => limits.advancedOptions ? setAdvancedOpen(!advancedOpen) : showUpgradeModal('Advanced Options', 'pro')}
                                                            >
                                                                <Settings size={18} />
                                                                <span>Advanced Options</span>
                                                                {!limits.advancedOptions && (
                                                                    <span className="locked-badge">
                                                                        <Lock size={12} />
                                                                        Pro
                                                                    </span>
                                                                )}
                                                                {limits.advancedOptions && (
                                                                    advancedOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />
                                                                )}
                                                            </button>

                                                            {advancedOpen && limits.advancedOptions && (
                                                                <div className="advanced-options">
                                                                    <div className="config-grid">
                                                                        <div className="config-field">
                                                                            <label className="config-label">Search Strategy</label>
                                                                            <div className="auto-tuned-indicator" style={{ padding: '12px', borderRadius: '8px', background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.15))' }}>
                                                                                <Sparkles size={16} />
                                                                                <span style={{ fontWeight: 600 }}>Quantum Adaptive HPO (QAHPO)</span>
                                                                            </div>
                                                                            <p className="auto-tuned-hint" style={{ marginTop: '4px' }}>
                                                                                Uses quantum-inspired amplitude selection, thermal annealing, and tunneling.
                                                                            </p>
                                                                        </div>
                                                                        <div className="config-field">
                                                                            <label className="config-label">Optimizer</label>
                                                                            <Select
                                                                                options={[
                                                                                    { value: 'sophiag', label: 'SophiaG (Recommended)' },
                                                                                    { value: 'qiao', label: 'QIAO (Quantum-Inspired Alternating)' },
                                                                                    { value: 'grover', label: 'Grover-Q (Quantum-Enhanced)' },
                                                                                    { value: 'sympflow', label: 'SympFlow (Symplectic Hamiltonian)' },
                                                                                    { value: 'sympflowqng', label: 'SympFlowQNG (Quantum Natural Gradient)' },
                                                                                    { value: 'adamw', label: 'AdamW' },
                                                                                    { value: 'adam', label: 'Adam' },
                                                                                    { value: 'lion', label: 'Lion' },
                                                                                ].filter(o => limits.allowedOptimizers.includes(o.value))}
                                                                                value={selectedOptimizers[0]}
                                                                                onChange={(e) => setSelectedOptimizers([e.target.value])}
                                                                                fullWidth
                                                                            />
                                                                        </div>
                                                                    </div>
                                                                    {/* Learning rate auto-tuning indicator */}
                                                                    <div className="config-field auto-tuned-field">
                                                                        <div className="auto-tuned-indicator">
                                                                            <Sparkles size={16} />
                                                                            <span>Learning rate will be auto-tuned for {selectedOptimizers[0] === 'sophiag' ? 'SophiaG' : selectedOptimizers[0] === 'qiao' ? 'QIAO' : selectedOptimizers[0]}</span>
                                                                        </div>
                                                                        <p className="auto-tuned-hint">
                                                                            Uses optimizer-specific ranges with log-uniform sampling for stable training.
                                                                        </p>
                                                                    </div>
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>
                                                </div>
                                            )}

                                            {/* Step 3: Review & Start */}
                                            {currentStep === 2 && (
                                                <div className="wizard-step">
                                                    <h3 className="wizard-step-title">Ready to optimize</h3>
                                                    <p className="wizard-step-desc">
                                                        Review your configuration and start
                                                    </p>

                                                    <div className="review-summary">
                                                        <div className="review-item">
                                                            <span className="review-label">Curriculum</span>
                                                            <span className="review-value">{curriculum?.name}</span>
                                                        </div>
                                                        <div className="review-item review-item-highlight">
                                                            <span className="review-label">Parameter Budget</span>
                                                            <span className="review-value review-value-safe">
                                                                Up to {formatParams(paramBudget)}
                                                            </span>
                                                        </div>
                                                        <div className="review-item">
                                                            <span className="review-label">Vocabulary</span>
                                                            <span className="review-value">Automatic (Quantum)</span>
                                                        </div>
                                                        <div className="review-item">
                                                            <span className="review-label">Context</span>
                                                            <span className="review-value">{formatNumber(contextWindow)} tokens</span>
                                                        </div>
                                                        <div className="review-item">
                                                            <span className="review-label">Termination</span>
                                                            <span className="review-value">Automatic (runs until convergence)</span>
                                                        </div>
                                                        <div className="review-item">
                                                            <span className="review-label">Strategy</span>
                                                            <span className="review-value">
                                                                Quantum Adaptive HPO
                                                            </span>
                                                        </div>
                                                        <div className="review-item">
                                                            <span className="review-label">Optimizer</span>
                                                            <span className="review-value">
                                                                {selectedOptimizers[0] === 'sophiag' ? 'SophiaG' :
                                                                    selectedOptimizers[0] === 'qiao' ? 'QIAO (Quantum-Inspired Alternating)' :
                                                                        selectedOptimizers[0] === 'grover' ? 'Grover-Q (Quantum-Enhanced)' :
                                                                            selectedOptimizers[0] === 'sympflow' ? 'SympFlow (Symplectic Hamiltonian)' :
                                                                                selectedOptimizers[0] === 'sympflowqng' ? 'SympFlowQNG (Quantum Natural Gradient)' :
                                                                                    selectedOptimizers[0] === 'adamw' ? 'AdamW' :
                                                                                        selectedOptimizers[0] === 'adam' ? 'Adam' :
                                                                                            selectedOptimizers[0] === 'lion' ? 'Lion' :
                                                                                                selectedOptimizers[0]}
                                                            </span>
                                                        </div>
                                                    </div>

                                                    <div className="auto-config-note">
                                                        <Sparkles size={16} />
                                                        <span>Architecture, learning rate, batch size, and scheduler will be auto-optimized</span>
                                                    </div>
                                                </div>
                                            )}
                                        </div>

                                        {/* Wizard Navigation */}
                                        <div className="wizard-nav">
                                            {currentStep > 0 && (
                                                <Button variant="ghost" onClick={handleBack}>Back</Button>
                                            )}
                                            <div className="wizard-nav-spacer" />
                                            {currentStep < steps.length - 1 ? (
                                                <Button variant="primary" onClick={handleNext} disabled={!canProceed()}>
                                                    Continue
                                                </Button>
                                            ) : (
                                                <Button variant="primary" leftIcon={<Play size={16} />} onClick={handleStartSweep} disabled={isRunning}>
                                                    Start Optimization
                                                </Button>
                                            )}
                                        </div>
                                    </CardContent>
                                </Card>
                            </div>

                            {/* Right Column: Status & Results */}
                            <div className="hpo-results-column">
                                <Card padding="lg">
                                    <CardHeader title="Status" />
                                    <CardContent>
                                        <div className="sweep-status">
                                            <div className="status-indicator">
                                                <span className={`status-dot ${isRunning ? 'status-running' : 'status-idle'}`}></span>
                                                <span className="status-text">{isRunning ? 'Running' : 'Ready'}</span>
                                                <span className="status-strategy">QAHPO</span>
                                            </div>
                                            <div className="status-metrics">
                                                <div className="status-metric">
                                                    <span className="metric-label">Trials</span>
                                                    <span className="metric-value">{trials.length}</span>
                                                </div>
                                                <div className="status-metric">
                                                    <span className="metric-label">Best Loss</span>
                                                    <span className="metric-value">{sweepStatus?.best_loss?.toFixed(4) ?? '—'}</span>
                                                </div>
                                                <div className="status-metric">
                                                    <span className="metric-label">Best Composite</span>
                                                    <span className="metric-value">{sweepStatus?.best_composite_score?.toFixed(4) ?? '—'}</span>
                                                </div>
                                                <div className="status-metric">
                                                    <span className="metric-label">Best PPL</span>
                                                    <span className="metric-value">{sweepStatus?.best_perplexity?.toFixed(2) ?? '—'}</span>
                                                </div>
                                            </div>
                                            {isRunning && (
                                                <div className="convergence-indicator">
                                                    <span className="convergence-text">Running until convergence...</span>
                                                </div>
                                            )}
                                        </div>
                                    </CardContent>
                                </Card>

                                <Card padding="lg">
                                    <CardHeader
                                        title="Best Found"
                                        action={<Button variant="ghost" size="sm" leftIcon={<Download size={14} />}>Export</Button>}
                                    />
                                    <CardContent>
                                        {sweepStatus?.best_hyperparams ? (
                                            <div className="best-params">
                                                {Object.entries(sweepStatus.best_hyperparams).map(([key, value]) => (
                                                    <div key={key} className="param-row-display">
                                                        <span className="param-key">{key}</span>
                                                        <span className="param-value">{String(value)}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <div className="empty-state-sm">
                                                <Info size={24} />
                                                <p>Start optimization to find best params</p>
                                            </div>
                                        )}
                                    </CardContent>
                                </Card>

                                {/* Budget Enforcement Stats */}
                                {budgetStats && budgetStats.total_skipped > 0 && (
                                    <Card padding="lg" className="budget-stats-card">
                                        <CardHeader
                                            title="Budget Enforcement"
                                            subtitle="Parameter budget optimization"
                                        />
                                        <CardContent>
                                            <div className="budget-stats">
                                                <div className="budget-stat">
                                                    <span className="stat-label">Budget</span>
                                                    <span className="stat-value">{formatNumber(budgetStats.param_budget)}</span>
                                                </div>
                                                <div className="budget-stat">
                                                    <span className="stat-label">Skipped</span>
                                                    <span className="stat-value stat-warning">{budgetStats.total_skipped}</span>
                                                </div>
                                                <div className="budget-stat">
                                                    <span className="stat-label">Skip Rate</span>
                                                    <span className="stat-value">{(budgetStats.skip_rate * 100).toFixed(1)}%</span>
                                                </div>
                                            </div>
                                            {budgetStats.safe_bounds && (
                                                <div className="safe-bounds">
                                                    <span className="safe-bounds-title">Safe Architecture Bounds:</span>
                                                    <div className="bounds-grid">
                                                        <span>Embed: {budgetStats.safe_bounds.max_embedding_dim}</span>
                                                        <span>Blocks: {budgetStats.safe_bounds.max_reasoning_blocks}</span>
                                                        <span>Experts: {budgetStats.safe_bounds.max_moe_experts}</span>
                                                    </div>
                                                </div>
                                            )}
                                        </CardContent>
                                    </Card>
                                )}

                                <Card padding="none" className="trials-card">
                                    <CardHeader
                                        title="Trial Results"
                                        action={<Button variant="ghost" size="sm" leftIcon={<RefreshCw size={14} />}>Refresh</Button>}
                                    />
                                    <div className="trials-table-wrapper">
                                        <table className="trials-table">
                                            <thead>
                                                <tr>
                                                    <th>Trial</th>
                                                    <th>Status</th>
                                                    <th>LR</th>
                                                    <th>Loss</th>
                                                    <th>PPL</th>
                                                    <th>Conf</th>
                                                    <th>Composite</th>
                                                    <th>Memory</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {trials.length === 0 ? (
                                                    <tr>
                                                        <td colSpan={8} className="trials-empty">
                                                            No trials yet
                                                        </td>
                                                    </tr>
                                                ) : (
                                                    trials.map((trial) => (
                                                        <tr key={trial.trial_id}>
                                                            <td>#{trial.trial_id}</td>
                                                            <td>
                                                                <span className={`trial-status trial-status-${trial.status}`}>
                                                                    {trial.status}
                                                                </span>
                                                            </td>
                                                            <td>{trial.learning_rate ? trial.learning_rate.toExponential(2) : '—'}</td>
                                                            <td>{trial.loss?.toFixed(4) ?? '—'}</td>
                                                            <td>{trial.perplexity?.toFixed(2) ?? '—'}</td>
                                                            <td>{trial.mean_confidence?.toFixed(3) ?? '—'}</td>
                                                            <td>{trial.composite_score?.toFixed(4) ?? '—'}</td>
                                                            <td>{trial.memory_mb ? `${trial.memory_mb.toFixed(0)} MB` : '—'}</td>
                                                        </tr>
                                                    ))
                                                )}
                                            </tbody>
                                        </table>
                                    </div>
                                </Card>

                                {/* Training Console - Real-time logs */}
                                <TrainingConsole
                                    sweepId={sweepStatus?.sweep_id || null}
                                    isRunning={isRunning}
                                    devMode={devMode}
                                />
                            </div>
                        </div>

                        {/* Upgrade Modal */}
                        <UpgradeModal
                            isOpen={upgradeModal.open}
                            onClose={() => setUpgradeModal({ ...upgradeModal, open: false })}
                            feature={upgradeModal.feature}
                            requiredTier={upgradeModal.tier}
                        />
                    </>
                )}
            </div>
        </LicenseContext.Provider>
    );
}
