// HighNoon Dashboard - API Client
// Centralized API client with type safety

import type {
    SearchSpaceConfig,
    StartHPORequest,
    StartTrainingRequest,
    HPOSweepInfo,
    HPOTrialInfo,
    TrainingJobInfo,
    TrainingMetrics,
    DatasetInfo,
    HuggingFaceDataset,
    Curriculum,
    CurriculumStage,
    LiteEditionLimits,
} from './types';

const API_BASE = '/api';

/**
 * Generic fetch wrapper with error handling.
 */
async function fetchApi<T>(
    endpoint: string,
    options?: RequestInit
): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        headers: {
            'Content-Type': 'application/json',
            ...options?.headers,
        },
        ...options,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
}

// ============================================================================
// HPO Endpoints
// ============================================================================

export const hpoApi = {
    /** Get default search space configuration with Lite limits applied. */
    async getSearchSpaceDefaults(): Promise<SearchSpaceConfig> {
        return fetchApi('/hpo/search-space/defaults');
    },

    /** Validate search space configuration against Lite limits. */
    async validateSearchSpace(config: SearchSpaceConfig): Promise<{ valid: boolean; errors: string[] }> {
        return fetchApi('/hpo/search-space/validate', {
            method: 'POST',
            body: JSON.stringify(config),
        });
    },

    /** Start a new HPO sweep. */
    async startSweep(request: StartHPORequest): Promise<HPOSweepInfo> {
        return fetchApi('/hpo/sweep/start', {
            method: 'POST',
            body: JSON.stringify(request),
        });
    },

    /** Get HPO sweep status. */
    async getSweepStatus(sweepId: string): Promise<HPOSweepInfo> {
        return fetchApi(`/hpo/sweep/${sweepId}/status`);
    },

    /** Get all trials for a sweep. */
    async getTrials(sweepId: string): Promise<HPOTrialInfo[]> {
        const response = await fetchApi<{ trials: HPOTrialInfo[] }>(`/hpo/sweep/${sweepId}/trials`);
        return response.trials || [];
    },

    /** Get best hyperparameters from a sweep. */
    async getBestParams(sweepId: string): Promise<Record<string, unknown>> {
        return fetchApi(`/hpo/sweep/${sweepId}/best`);
    },

    /** Cancel an HPO sweep. */
    async cancelSweep(sweepId: string): Promise<void> {
        await fetchApi(`/hpo/sweep/${sweepId}/cancel`, { method: 'POST' });
    },

    /** List all HPO sweeps. */
    async listSweeps(): Promise<HPOSweepInfo[]> {
        const response = await fetchApi<{ sweeps: HPOSweepInfo[] }>('/hpo/sweeps');
        return response.sweeps || [];
    },

    /** List available HPO configurations/presets. */
    async listConfigs(): Promise<{ name: string; description: string }[]> {
        return fetchApi('/hpo/configs');
    },

    /** Get parameter budget enforcement statistics. */
    async getBudgetStats(sweepId: string): Promise<{
        enabled: boolean;
        param_budget: number;
        total_skipped: number;
        skip_rate: number;
        safe_bounds?: {
            max_embedding_dim: number;
            max_reasoning_blocks: number;
            max_moe_experts: number;
        };
    }> {
        return fetchApi(`/hpo/sweep/${sweepId}/budget`);
    },

    /** Get skipped trials due to budget constraints. */
    async getSkippedTrials(sweepId: string): Promise<{
        skipped_trials: Array<{
            trial_id: string;
            reason: string;
            estimated_params: number;
            param_budget: number;
        }>;
        statistics: {
            total_skipped: number;
            skip_rate: number;
        };
    }> {
        return fetchApi(`/hpo/sweep/${sweepId}/skipped`);
    },

    /** Export best hyperparameter configuration as JSON or YAML. */
    async exportConfig(sweepId: string, format: 'json' | 'yaml' = 'json'): Promise<void> {
        // Use direct window.open for file download since fetchApi expects JSON
        window.open(`${API_BASE}/hpo/sweep/${sweepId}/export?format=${format}`, '_blank');
    },

    /** Get training config from completed sweep. */
    async getTrainingConfig(sweepId: string): Promise<{
        sweep_id: string;
        training_config: Record<string, unknown>;
        model_build_config: Record<string, unknown>;
        optimizer_config: Record<string, unknown>;
        best_loss?: number;
    }> {
        return fetchApi(`/hpo/sweep/${sweepId}/training-config`);
    },
};

// ============================================================================
// Training Endpoints
// ============================================================================

export const trainingApi = {
    /** Start training with HPO-optimized parameters. */
    async start(request: StartTrainingRequest): Promise<TrainingJobInfo> {
        return fetchApi('/training/start', {
            method: 'POST',
            body: JSON.stringify(request),
        });
    },

    /** Get training job status. */
    async getStatus(jobId: string): Promise<TrainingJobInfo> {
        return fetchApi(`/training/${jobId}/status`);
    },

    /** Get current training metrics. */
    async getMetrics(jobId: string): Promise<TrainingMetrics> {
        return fetchApi(`/training/${jobId}/metrics`);
    },

    /** Pause training. */
    async pause(jobId: string): Promise<void> {
        await fetchApi(`/training/${jobId}/pause`, { method: 'POST' });
    },

    /** Resume training. */
    async resume(jobId: string): Promise<void> {
        await fetchApi(`/training/${jobId}/resume`, { method: 'POST' });
    },

    /** Cancel training. */
    async cancel(jobId: string): Promise<void> {
        await fetchApi(`/training/${jobId}/cancel`, { method: 'POST' });
    },

    /** List checkpoints for a job. */
    async listCheckpoints(jobId: string): Promise<{ path: string; step: number; created_at: string }[]> {
        return fetchApi(`/training/${jobId}/checkpoints`);
    },

    /** List all training jobs. */
    async listJobs(): Promise<TrainingJobInfo[]> {
        const response = await fetchApi<{ jobs: TrainingJobInfo[] }>('/training/jobs');
        return response.jobs || [];
    },

    /** Get training presets with resource estimates. */
    async getPresets(): Promise<{ name: string; description: string; ram_estimate: string }[]> {
        return fetchApi('/training/presets');
    },

    /** Get training logs with pagination. */
    async getLogs(jobId: string, sinceIndex = 0, limit = 100): Promise<{
        logs: Array<{
            timestamp: string;
            level: string;
            message: string;
            step?: number;
            loss?: number;
            learning_rate?: number;
            throughput?: number;
            memory_mb?: number;
            gradient_norm?: number;
        }>;
        next_index: number;
        total: number;
    }> {
        return fetchApi(`/training/${jobId}/logs?since_index=${sinceIndex}&limit=${limit}`);
    },

    /** Get system health for a training job. */
    async getHealth(jobId: string): Promise<{
        cpu_percent: number;
        memory_used_gb: number;
        memory_total_gb: number;
        disk_used_gb: number;
        disk_total_gb: number;
    }> {
        return fetchApi(`/training/${jobId}/health`);
    },

    /** Clear training logs. */
    async clearLogs(jobId: string): Promise<void> {
        await fetchApi(`/training/${jobId}/logs`, { method: 'DELETE' });
    },
};

// ============================================================================
// Dataset Endpoints
// ============================================================================

interface BackendDatasetResponse {
    datasets: Array<{
        dataset_id: string;
        provider: string;
        description: string;
        downloads?: number;
        total_size_bytes?: number;
        media_types?: string[];
    }>;
}

interface BackendHuggingFaceSearchResponse {
    datasets: Array<{
        dataset_id: string;
        provider: string;
        description: string;
        downloads: number;
        likes: number;
        tags: string[];
        last_modified: string;
        gated?: boolean | string;
        private?: boolean;
    }>;
    total: number;
    token_configured?: boolean;
}

interface BackendAddDatasetResponse {
    status: string;
    dataset: {
        dataset_id: string;
        provider: string;
        description: string;
        downloads?: number;
        total_size_bytes?: number;
    };
}

export const datasetApi = {
    /** List all local datasets. */
    async list(): Promise<DatasetInfo[]> {
        const response = await fetchApi<BackendDatasetResponse>('/datasets');
        // Transform backend format to frontend format
        return response.datasets.map((ds) => ({
            id: ds.dataset_id,
            name: ds.dataset_id.split('/').pop() || ds.dataset_id,
            source: ds.provider as 'huggingface' | 'local' | 'remote',
            description: ds.description || '',
            num_examples: (ds as any).num_examples ?? (ds.total_size_bytes ? Math.floor(ds.total_size_bytes / 1000) : 0),
            download_status: 'ready' as const,
        }));
    },

    /** Get dataset info. */
    async get(datasetId: string): Promise<DatasetInfo> {
        const ds = await fetchApi<{
            dataset_id: string;
            provider: string;
            description: string;
            total_size_bytes?: number;
        }>(`/datasets/${encodeURIComponent(datasetId)}`);
        return {
            id: ds.dataset_id,
            name: ds.dataset_id.split('/').pop() || ds.dataset_id,
            source: ds.provider as 'huggingface' | 'local' | 'remote',
            description: ds.description || '',
            num_examples: ds.total_size_bytes ? Math.floor(ds.total_size_bytes / 1000) : 0,
        };
    },

    /** Inspect dataset structure. */
    async inspect(datasetId: string, objectIndex = 0): Promise<Record<string, unknown>> {
        return fetchApi(`/datasets/${encodeURIComponent(datasetId)}/inspect?object_index=${objectIndex}`);
    },

    /** Search HuggingFace Hub. */
    async searchHuggingFace(query: string, limit = 20): Promise<HuggingFaceDataset[]> {
        const response = await fetchApi<BackendHuggingFaceSearchResponse>(
            `/huggingface/search?query=${encodeURIComponent(query)}&limit=${limit}`
        );
        // Transform backend format to frontend format
        return response.datasets.map((ds) => {
            const parts = ds.dataset_id.split('/');
            return {
                id: ds.dataset_id,
                author: parts.length > 1 ? parts[0] : 'unknown',
                name: parts.length > 1 ? parts[1] : ds.dataset_id,
                description: ds.description || '',
                downloads: ds.downloads || 0,
                likes: ds.likes || 0,
                tags: ds.tags || [],
                lastModified: ds.last_modified || '',
                gated: ds.gated,
                private: ds.private,
            };
        });
    },

    /** Get HuggingFace dataset info. */
    async getHuggingFaceInfo(datasetId: string): Promise<HuggingFaceDataset> {
        const ds = await fetchApi<{
            dataset_id: string;
            description: string;
            downloads: number;
            likes: number;
            tags: string[];
        }>(`/huggingface/datasets/${encodeURIComponent(datasetId)}`);
        const parts = ds.dataset_id.split('/');
        return {
            id: ds.dataset_id,
            author: parts.length > 1 ? parts[0] : 'unknown',
            name: parts.length > 1 ? parts[1] : ds.dataset_id,
            description: ds.description || '',
            downloads: ds.downloads || 0,
            likes: ds.likes || 0,
            tags: ds.tags || [],
            lastModified: '',
        };
    },

    /** Add HuggingFace dataset to local catalog. */
    async addHuggingFace(datasetId: string, config?: string, splits?: string[]): Promise<DatasetInfo> {
        const response = await fetchApi<BackendAddDatasetResponse>('/datasets/add-huggingface', {
            method: 'POST',
            body: JSON.stringify({ dataset_id: datasetId, config_name: config, splits: splits || ['train'] }),
        });
        const ds = response.dataset;
        return {
            id: ds.dataset_id,
            name: ds.dataset_id.split('/').pop() || ds.dataset_id,
            source: 'huggingface',
            description: ds.description || '',
            num_examples: (ds as any).num_examples ?? (ds.total_size_bytes ? Math.floor(ds.total_size_bytes / 1000) : 0),
            download_status: 'pending',
        };
    },

    /** Remove dataset from catalog. */
    async remove(datasetId: string): Promise<void> {
        await fetchApi(`/datasets/${encodeURIComponent(datasetId)}`, { method: 'DELETE' });
    },
};

// ============================================================================
// Curriculum Endpoints
// ============================================================================

export const curriculumApi = {
    /** List all saved curricula. */
    async list(): Promise<Curriculum[]> {
        // The backend returns the curricula array directly
        return fetchApi('/curriculum');
    },

    /** Save a complete curriculum (create or update). */
    async save(curriculum: Curriculum): Promise<{ status: string; curriculum: Curriculum }> {
        return fetchApi('/curriculum', {
            method: 'POST',
            body: JSON.stringify({
                id: curriculum.id,
                name: curriculum.name,
                stages: curriculum.stages,
                created_at: curriculum.created_at,
                updated_at: curriculum.updated_at,
            }),
        });
    },

    /** Get a specific curriculum by ID. */
    async get(curriculumId: string): Promise<Curriculum> {
        return fetchApi(`/curriculum/${encodeURIComponent(curriculumId)}`);
    },

    /** Delete a curriculum. */
    async delete(curriculumId: string): Promise<void> {
        await fetchApi(`/curriculum/${encodeURIComponent(curriculumId)}`, { method: 'DELETE' });
    },

    /** Get curriculum stages (legacy). */
    async getStages(): Promise<CurriculumStage[]> {
        const response = await fetchApi<{ stages: CurriculumStage[] }>('/curriculum/stages');
        return response.stages;
    },

    /** Create a new stage (legacy). */
    async createStage(stage: Partial<CurriculumStage>): Promise<CurriculumStage> {
        return fetchApi('/curriculum/stages', {
            method: 'POST',
            body: JSON.stringify(stage),
        });
    },

    /** Update a stage (legacy). */
    async updateStage(stageName: string, updates: Partial<CurriculumStage>): Promise<CurriculumStage> {
        return fetchApi(`/curriculum/stages/${encodeURIComponent(stageName)}`, {
            method: 'PUT',
            body: JSON.stringify(updates),
        });
    },

    /** Delete a stage (legacy). */
    async deleteStage(stageName: string): Promise<void> {
        await fetchApi(`/curriculum/stages/${encodeURIComponent(stageName)}`, { method: 'DELETE' });
    },

    /** Add dataset to stage (legacy). */
    async addDatasetToStage(stageName: string, datasetId: string, weight = 1.0): Promise<void> {
        await fetchApi(`/curriculum/stages/${encodeURIComponent(stageName)}/datasets`, {
            method: 'POST',
            body: JSON.stringify({ dataset_id: datasetId, weight }),
        });
    },

    /** Remove dataset from stage (legacy). */
    async removeDatasetFromStage(stageName: string, datasetId: string): Promise<void> {
        await fetchApi(`/curriculum/stages/${encodeURIComponent(stageName)}/datasets/${encodeURIComponent(datasetId)}`, {
            method: 'DELETE',
        });
    },

    /** List predefined curriculum presets with HuggingFace dataset mappings. */
    async listPresets(): Promise<{ id: string; name: string; description: string; hf_datasets: string[]; stages: any[] }[]> {
        const response = await fetchApi<{ presets: { id: string; name: string; description: string; hf_datasets: string[]; stages: any[] }[] }>('/curriculum/presets');
        return response.presets;
    },
};


// ============================================================================
// System Endpoints
// ============================================================================

export const systemApi = {
    /** Health check. */
    async health(): Promise<{ status: string }> {
        return fetchApi('/health');
    },

    /** Get Lite edition limits. */
    async getLimits(): Promise<LiteEditionLimits> {
        return fetchApi('/system/limits');
    },

    /** Estimate resource requirements. */
    async estimateResources(modelSize: string, batchSize: number): Promise<{ ram_gb: number; vram_gb: number }> {
        return fetchApi(`/system/estimate?model_size=${modelSize}&batch_size=${batchSize}`);
    },
};

// ============================================================================
// Settings Endpoints
// ============================================================================

export const settingsApi = {
    /** Get HuggingFace token configuration status. */
    async getHfTokenStatus(): Promise<{ configured: boolean; token_prefix?: string | null }> {
        return fetchApi('/settings/hf-token/status');
    },

    /** Save HuggingFace token. */
    async saveHfToken(token: string): Promise<{ status: string; token_prefix: string }> {
        return fetchApi('/settings/hf-token', {
            method: 'PUT',
            body: JSON.stringify({ token }),
        });
    },

    /** Clear HuggingFace token. */
    async clearHfToken(): Promise<{ status: string }> {
        return fetchApi('/settings/hf-token', { method: 'DELETE' });
    },

    /** Validate HuggingFace token against HF API. */
    async validateHfToken(token?: string): Promise<{
        valid: boolean;
        username?: string;
        email?: string;
        error?: string;
    }> {
        return fetchApi('/settings/hf-token/validate', {
            method: 'POST',
            body: token ? JSON.stringify({ token }) : JSON.stringify({}),
        });
    },
};

// ============================================================================
// Activation Visualization Endpoints
// ============================================================================

/** Surface data returned by activation surface API */
export interface ActivationSurfaceData {
    x: number[];
    y: number[];
    z: number[][];
    colorscale: string;
    layer_name: string;
    original_shape: [number, number];
    stats: {
        min: number;
        max: number;
        mean: number;
        std: number;
    };
    demo_mode: boolean;
}

export const activationApi = {
    /** Get available layers for activation visualization. */
    async getLayers(jobId: string): Promise<{ layers: string[]; demo_mode: boolean }> {
        return fetchApi(`/activations/layers/${encodeURIComponent(jobId)}`);
    },

    /** Get activation surface data for a specific layer. */
    async getSurface(
        layerName: string,
        jobId: string,
        options?: {
            applyEnvelope?: boolean;
            gridSize?: number;
        }
    ): Promise<ActivationSurfaceData> {
        const params = new URLSearchParams({
            job_id: jobId,
            apply_envelope: String(options?.applyEnvelope ?? true),
            grid_size: String(options?.gridSize ?? 50),
        });
        return fetchApi(`/activations/surface/${encodeURIComponent(layerName)}?${params}`);
    },

    /** Clear activation cache for a job. */
    async clearCache(jobId: string): Promise<{ status: string; job_id: string }> {
        return fetchApi(`/activations/cache/${encodeURIComponent(jobId)}`, {
            method: 'DELETE',
        });
    },
};
