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
    trial_id: number;
    status: 'running' | 'completed' | 'pruned' | 'failed';
    hyperparams: Record<string, unknown>;
    learning_rate: number;
    batch_size: number;
    optimizer: string;
    loss: number | null;
    best_loss: number | null;
    duration_seconds: number | null;
    step: number;
    pruned_at_step?: number;
    memory_mb?: number;       // Current RSS memory in MB
    peak_memory_mb?: number;  // Peak RSS memory in MB
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
    best_trial_id: number | null;
    best_loss: number | null;
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
    | { type: 'error'; message: string };

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
