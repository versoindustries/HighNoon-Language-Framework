// HighNoon Dashboard - API Types
// Type definitions for all API endpoints

// ============================================================================
// Common Types
// ============================================================================

export interface ApiResponse<T> {
    data: T;
    error?: string;
}

export type JobState = 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
export type TrainingMode = 'quick_train' | 'auto_tune' | 'full_sweep';
export type HPOStage = 'coarse' | 'refine' | 'fine';
export type SchedulerType = 'cosine' | 'linear' | 'warmup_stable_decay' | 'polynomial';
export type ChunkingStrategy = 'fixed' | 'sentence' | 'paragraph';

// ============================================================================
// HPO Configuration Types (Forest-style grouped parameters)
// ============================================================================

/** Learning rate configuration group */
export interface LearningRateConfig {
    min: string;
    max: string;
    logScale: boolean;
    distribution: 'log_uniform' | 'uniform';
}

/** Scheduler configuration - grouped with LR for forest tuning */
export interface SchedulerConfig {
    type: SchedulerType;
    warmupSteps: number;
    warmupRatio: number;
    numCycles: number;  // For cosine
    minLrRatio: number; // End LR as fraction of peak
}

/** Batch size and gradient accumulation - grouped for memory optimization */
export interface BatchConfig {
    options: number[];
    gradientAccumulationSteps: number[];
    effectiveBatchSizeTarget?: number;
    memoryEstimates: Record<number, string>;
}

/** Optimizer and regularization - grouped for convergence */
export interface OptimizerConfig {
    options: string[];
    selected: string[];
    weightDecayRange: { min: number; max: number };
    gradientClipping: { maxNorm: number | null; maxValue: number | null };
    beta1Range: { min: number; max: number };
    beta2Range: { min: number; max: number };
    epsilon: number;
}

/** Tokenizer configuration */
export interface TokenizerConfig {
    vocabSize: number;
    maxContextLength: number;
    padToken: string;
    eosToken: string;
    bosToken: string;
    unkToken: string;
    addSpecialTokens: boolean;
}

/** Context window configuration */
export interface ContextConfig {
    maxSequenceLength: number;
    slidingWindowSize: number;
    chunkingStrategy: ChunkingStrategy;
    overlapTokens: number;
    positionEmbeddingType: 'rope' | 'alibi' | 'learned';
}

/** Model architecture tuning */
export interface ArchitectureConfig {
    hiddenSize: number;
    numLayers: number;
    numHeads: number;
    intermediateSize: number;
    numExperts: number;  // For MoE
    topK: number;        // For MoE
    dropoutRate: number;
}

/** User-defined sweep caps */
export interface SweepCaps {
    maxTrials: number;
    maxTimePerTrialMinutes: number;
    totalSweepBudgetHours: number;
    earlyStoppingPatience: number;
    convergenceThreshold: number;
    pruneThreshold: number;
    resourceCap: {
        maxMemoryGB: number;
        maxCpuCores: number;
    };
}

/** Complete HPO search space configuration */
export interface SearchSpaceConfig {
    // Forest-style parameter groups
    learningRate: LearningRateConfig;
    scheduler: SchedulerConfig;
    batch: BatchConfig;
    optimizer: OptimizerConfig;
    tokenizer: TokenizerConfig;
    context: ContextConfig;
    architecture?: ArchitectureConfig;

    // Sweep control
    caps: SweepCaps;
    searchStrategy: 'random' | 'bayesian' | 'hyperband' | 'forest';
    parallelTrials: number;
}

// ============================================================================
// Training Types
// ============================================================================

export interface TrainingJobInfo {
    job_id: string;
    state: JobState;
    mode: TrainingMode;
    model_size: string;
    current_stage: number;
    current_epoch: number;
    global_step: number;
    loss: number;
    learning_rate: number;
    throughput: number;
    progress_percent: number;
    started_at: string | null;
    updated_at: string | null;
    hpo_trial_current: number;
    hpo_trial_total: number;
    best_hyperparams: Record<string, unknown> | null;
}

export interface TrainingMetrics {
    global_step: number;
    current_epoch: number;
    loss: number;
    learning_rate: number;
    throughput: number;
    progress_percent: number;
    memory_used_gb?: number;
    gradient_norm?: number;
}

export interface StartTrainingRequest {
    mode: TrainingMode;
    // For HPO mode
    sweep_id?: string;
    // For Quick Train mode
    curriculum_id?: string;
    model_size?: string;  // '1b', '3b', '7b', '12b', '20b'
    // Common options
    output_dir?: string;
    resume_from_checkpoint?: boolean;
}

// ============================================================================
// HPO Types
// ============================================================================

export interface HPOTrialInfo {
    trial_id: string | number;  // Backend uses string (e.g., 'trial_0')
    status: 'running' | 'completed' | 'pruned' | 'failed';
    hyperparams?: Record<string, unknown>;
    learning_rate: number;
    batch_size?: number;
    optimizer?: string;
    loss: number | null;
    best_loss?: number | null;
    duration_seconds?: number | null;
    step?: number;
    pruned_at_step?: number;
    memory_mb?: number;       // Current RSS memory in MB
    peak_memory_mb?: number;  // Peak RSS memory in MB
    // Multi-objective quality metrics
    perplexity?: number | null;
    mean_confidence?: number | null;
    expected_calibration_error?: number | null;
    composite_score?: number | null;
    // Performance metrics
    throughput_tokens_per_sec?: number | null;
}

/** Model configuration from HPO sweep */
export interface HPOModelConfig {
    vocab_size: number;
    context_window: number;
    embedding_dim: number;
    num_reasoning_blocks: number;
    num_moe_experts: number;
    position_embedding: string;
    param_budget: number;  // Source of truth for model size
}

export interface HPOSweepInfo {
    sweep_id: string;
    stage: HPOStage;
    state: JobState;
    search_space: SearchSpaceConfig;
    max_trials: number;
    completed_trials: number;
    pruned_trials: number;
    best_trial_id: string | number | null;
    best_loss: number | null;
    best_composite_score?: number | null;  // Multi-objective best score
    best_perplexity?: number | null;        // Best trial's perplexity
    best_confidence?: number | null;        // Best trial's mean confidence
    best_memory_mb?: number | null;         // Best trial's peak memory
    best_hyperparams: Record<string, unknown> | null;
    started_at: string | null;
    estimated_completion: string | null;
    trials: HPOTrialInfo[];
    // Model configuration - HPO is the source of truth
    model_config?: HPOModelConfig;
    config?: {
        curriculum_id?: string;
        search_strategy?: string;
    };
}

export interface StartHPORequest {
    search_space: SearchSpaceConfig;
    stage?: HPOStage;
    curriculum_id?: string;
}

// ============================================================================
// Dataset Types
// ============================================================================

export interface DatasetInfo {
    id: string;
    name: string;
    source: 'huggingface' | 'local' | 'remote';
    description: string;
    num_examples: number;
    size_bytes?: number;
    features?: Record<string, string>;
    splits?: string[];
    download_status?: 'pending' | 'downloading' | 'ready' | 'error';
}

export interface HuggingFaceDataset {
    id: string;
    author: string;
    name: string;
    description: string;
    downloads: number;
    likes: number;
    tags: string[];
    lastModified: string;
    gated?: boolean | string;  // false, true, "auto", or "manual"
    private?: boolean;
}

// ============================================================================
// Curriculum Types
// ============================================================================

export interface CurriculumStage {
    name: string;
    display_name: string;
    module: string;
    datasets: CurriculumDataset[];
    epochs: number;
    learning_rate: string;
    batch_size: number;
    weight: number;
}

export interface CurriculumDataset {
    dataset_id: string;
    weight: number;
    splits?: string[];
}

export interface Curriculum {
    id: string;
    name: string;
    stages: CurriculumStage[];
    created_at: string;
    updated_at: string;
}

// ============================================================================
// WebSocket Message Types
// ============================================================================

export type WebSocketMessage =
    | { type: 'metrics'; data: TrainingMetrics }
    | { type: 'trial_update'; data: HPOTrialInfo }
    | { type: 'sweep_update'; data: Partial<HPOSweepInfo> }
    | { type: 'log'; data: { level: string; message: string; timestamp: string } }
    | { type: 'finished'; state: JobState }
    | { type: 'error'; message: string }
    | { type: 'pong'; ts: number };

// ============================================================================
// Lite Edition Limits
// ============================================================================

export interface LiteEditionLimits {
    maxParameters: number;       // 20B
    maxContextLength: number;    // 5M tokens
    maxReasoningBlocks: number;  // 24
    maxMoEExperts: number;       // 12
    maxHPOTrials: number;        // 100
}

export const LITE_LIMITS: LiteEditionLimits = {
    maxParameters: 20_000_000_000,
    maxContextLength: 5_000_000,
    maxReasoningBlocks: 24,
    maxMoEExperts: 12,
    maxHPOTrials: 100,
};

// ============================================================================
// Unified Smart Tuner Types
// ============================================================================

/** Smart tuner configuration */
export interface SmartTunerConfig {
    enabled: boolean;
    memory_enabled: boolean;
    coordination_mode: 'aggressive' | 'balanced' | 'conservative';
    exploration_decay: number;
    lr_initial: number;
    lr_min: number;
    lr_max: number;
    galore_rank: number;
    galore_adaptive_rank: boolean;
    barren_plateau_threshold: number;
    barren_plateau_aggressive: boolean;
    max_grad_norm: number;
    warmup_steps: number;
    exploration_steps: number;
    emergency_grad_threshold: number;
}

/** Smart tuner current status */
export interface SmartTunerStatus {
    enabled: boolean;
    current_phase: 'idle' | 'warmup' | 'exploration' | 'exploitation' | 'emergency';
    exploration_factor: number;
    emergency_mode: boolean;
    global_step: number;
    coordination_mode: string;
    lr_controller_stats: Record<string, unknown>;
    bp_monitor_stats: Record<string, unknown>;
    galore_stats: Record<string, unknown>;
}

/** Cross-trial memory statistics */
export interface TunerMemoryStats {
    trial_count: number;
    best_loss: number | null;
    best_trial_id: string | null;
    unique_architectures: number;
    mean_loss?: number;
    std_loss?: number;
    converged_count?: number;
    error?: string;
}

/** Suggested tuner configuration from memory */
export interface TunerSuggestion {
    initial_lr?: number;
    galore_rank?: number;
    exploration_factor?: number;
    target_lr?: number;
}

// ============================================================================
// Hyperparameter Importance Analysis Types (fANOVA)
// ============================================================================

/** Individual hyperparameter importance */
export interface ParameterImportance {
    name: string;
    importance: number;  // 0-1, fraction of variance explained
    rank: number;
    std: number;
    is_categorical: boolean;
}

/** Pairwise interaction importance */
export interface InteractionImportance {
    param1: string;
    param2: string;
    importance: number;
    is_significant: boolean;
}

/** Marginal effect curve for a parameter */
export interface MarginalCurve {
    x_values: (number | string)[];
    y_mean: number[];
    y_std: number[];
    is_categorical: boolean;
}

/** Complete importance analysis result */
export interface ImportanceResult {
    individual: ParameterImportance[];
    interactions: InteractionImportance[];
    total_variance: number;
    explained_variance: number;
    n_trials: number;
    param_names: string[];
}

/** API response for importance analysis */
export interface ImportanceAnalysisResponse {
    sweep_id: string;
    importance: ImportanceResult;
    marginal_curves: Record<string, MarginalCurve>;
    error?: string;
    message?: string;
}
