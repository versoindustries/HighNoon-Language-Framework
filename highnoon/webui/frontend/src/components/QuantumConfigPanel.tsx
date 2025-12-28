// QuantumConfigPanel.tsx - Quantum Training Enhancement Configuration UI
// Part 10: Cleanup.md WebUI Integration
// Copyright 2025 Verso Industries

import { useState, useEffect } from 'react';
import './QuantumConfigPanel.css';

interface TrainingEnhancementConfig {
    use_tensor_galore: boolean;
    galore_rank: number;
    galore_vqc_aware: boolean;
    use_quantum_natural_gradient: boolean;
    qng_damping: number;
    use_sympflow: boolean;
    sympflow_mass: number;
    sympflow_friction: number;
    barren_plateau_monitor: boolean;
    barren_plateau_threshold: number;
    use_qhpm_crystallization: boolean;
    crystallization_threshold: number;
    use_neural_zne: boolean;
    use_entropy_regularization: boolean;
    entropy_reg_weight: number;
}

interface SynergyConfig {
    s1_unified_qssm_gating: boolean;
    s2_coconut_qmamba_amplitudes: boolean;
    s5_qhpm_hopfield_threshold: boolean;
    s8_neural_zne_qssm_stats: boolean;
    s11_alphaqubit_decoder: boolean;
    s12_sympflow_qng_geodesic: boolean;
    s18_cayley_dense: boolean;
    s19_coconut_crystallize: boolean;
    s20_galore_bp_aware: boolean;
    s21_qalrc_quls_entropy: boolean;
    s22_bp_qalrc_escape: boolean;
}

interface ArchitectureConfig {
    coconut_max_thought_steps: number;
    coconut_bfs_branches: number;
    coconut_crystallize_threshold: number;
    use_holographic_state_binding: boolean;
    use_state_bus: boolean;
    use_unified_quantum_bus: boolean;
    token_shift_mode: string;
    reasoning_block_pattern: string;
}

export function QuantumConfigPanel() {
    const [activeTab, setActiveTab] = useState<'training' | 'synergies' | 'architecture'>('training');
    const [trainingConfig, setTrainingConfig] = useState<TrainingEnhancementConfig | null>(null);
    const [synergyConfig, setSynergyConfig] = useState<SynergyConfig | null>(null);
    const [archConfig, setArchConfig] = useState<ArchitectureConfig | null>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        loadConfigs();
    }, []);

    const loadConfigs = async () => {
        setLoading(true);
        try {
            const [training, synergies, arch] = await Promise.all([
                fetch('/api/config/training').then(r => r.json()),
                fetch('/api/config/synergies').then(r => r.json()),
                fetch('/api/config/architecture').then(r => r.json()),
            ]);
            setTrainingConfig(training.config);
            setSynergyConfig(synergies.config);
            setArchConfig(arch.config);
        } catch (error) {
            console.error('Failed to load quantum configs:', error);
        }
        setLoading(false);
    };

    const saveConfig = async (type: 'training' | 'synergies' | 'architecture') => {
        setSaving(true);
        try {
            const configMap = {
                training: trainingConfig,
                synergies: synergyConfig,
                architecture: archConfig,
            };
            await fetch(`/api/config/${type}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(configMap[type]),
            });
        } catch (error) {
            console.error(`Failed to save ${type} config:`, error);
        }
        setSaving(false);
    };

    if (loading) {
        return <div className="quantum-config-loading">Loading quantum configuration...</div>;
    }

    return (
        <div className="quantum-config-panel">
            <div className="quantum-config-header">
                <h2>⚛️ Quantum Enhancement Configuration</h2>
                <p>Configure training enhancements, synergies, and architecture settings</p>
            </div>

            <div className="quantum-config-tabs">
                <button
                    className={`tab ${activeTab === 'training' ? 'active' : ''}`}
                    onClick={() => setActiveTab('training')}
                >
                    Training
                </button>
                <button
                    className={`tab ${activeTab === 'synergies' ? 'active' : ''}`}
                    onClick={() => setActiveTab('synergies')}
                >
                    Synergies
                </button>
                <button
                    className={`tab ${activeTab === 'architecture' ? 'active' : ''}`}
                    onClick={() => setActiveTab('architecture')}
                >
                    Architecture
                </button>
            </div>

            <div className="quantum-config-content">
                {activeTab === 'training' && trainingConfig && (
                    <TrainingConfigForm
                        config={trainingConfig}
                        onChange={setTrainingConfig}
                        onSave={() => saveConfig('training')}
                        saving={saving}
                    />
                )}
                {activeTab === 'synergies' && synergyConfig && (
                    <SynergyConfigForm
                        config={synergyConfig}
                        onChange={setSynergyConfig}
                        onSave={() => saveConfig('synergies')}
                        saving={saving}
                    />
                )}
                {activeTab === 'architecture' && archConfig && (
                    <ArchitectureConfigForm
                        config={archConfig}
                        onChange={setArchConfig}
                        onSave={() => saveConfig('architecture')}
                        saving={saving}
                    />
                )}
            </div>
        </div>
    );
}

function TrainingConfigForm({
    config,
    onChange,
    onSave,
    saving,
}: {
    config: TrainingEnhancementConfig;
    onChange: (c: TrainingEnhancementConfig) => void;
    onSave: () => void;
    saving: boolean;
}) {
    const updateField = <K extends keyof TrainingEnhancementConfig>(
        field: K,
        value: TrainingEnhancementConfig[K]
    ) => {
        onChange({ ...config, [field]: value });
    };

    return (
        <div className="config-form">
            <div className="config-group">
                <h3>GaLore Compression</h3>
                <label>
                    <input
                        type="checkbox"
                        checked={config.use_tensor_galore}
                        onChange={(e) => updateField('use_tensor_galore', e.target.checked)}
                    />
                    Enable Tensor GaLore
                </label>
                <label>
                    Rank:
                    <input
                        type="number"
                        value={config.galore_rank}
                        onChange={(e) => updateField('galore_rank', parseInt(e.target.value))}
                        min={4}
                        max={256}
                    />
                </label>
                <label>
                    <input
                        type="checkbox"
                        checked={config.galore_vqc_aware}
                        onChange={(e) => updateField('galore_vqc_aware', e.target.checked)}
                    />
                    VQC-Aware Rank Boosting
                </label>
            </div>

            <div className="config-group">
                <h3>Quantum Natural Gradient</h3>
                <label>
                    <input
                        type="checkbox"
                        checked={config.use_quantum_natural_gradient}
                        onChange={(e) => updateField('use_quantum_natural_gradient', e.target.checked)}
                    />
                    Enable QNG
                </label>
                <label>
                    Damping:
                    <input
                        type="number"
                        value={config.qng_damping}
                        onChange={(e) => updateField('qng_damping', parseFloat(e.target.value))}
                        step={0.0001}
                        min={0}
                    />
                </label>
            </div>

            <div className="config-group">
                <h3>SympFlow Optimizer</h3>
                <label>
                    <input
                        type="checkbox"
                        checked={config.use_sympflow}
                        onChange={(e) => updateField('use_sympflow', e.target.checked)}
                    />
                    Enable SympFlow
                </label>
            </div>

            <div className="config-group">
                <h3>Barren Plateau Detection</h3>
                <label>
                    <input
                        type="checkbox"
                        checked={config.barren_plateau_monitor}
                        onChange={(e) => updateField('barren_plateau_monitor', e.target.checked)}
                    />
                    Enable BP Monitor
                </label>
                <label>
                    Threshold:
                    <input
                        type="number"
                        value={config.barren_plateau_threshold}
                        onChange={(e) => updateField('barren_plateau_threshold', parseFloat(e.target.value))}
                        step={1e-7}
                    />
                </label>
            </div>

            <div className="config-group">
                <h3>Neural ZNE</h3>
                <label>
                    <input
                        type="checkbox"
                        checked={config.use_neural_zne}
                        onChange={(e) => updateField('use_neural_zne', e.target.checked)}
                    />
                    Enable Neural ZNE Error Mitigation
                </label>
            </div>

            <button className="save-btn" onClick={onSave} disabled={saving}>
                {saving ? 'Saving...' : 'Save Training Config'}
            </button>
        </div>
    );
}

function SynergyConfigForm({
    config,
    onChange,
    onSave,
    saving,
}: {
    config: SynergyConfig;
    onChange: (c: SynergyConfig) => void;
    onSave: () => void;
    saving: boolean;
}) {
    const synergies = [
        { key: 's1_unified_qssm_gating', label: 'S1: QMamba ↔ Q-SSM Gating' },
        { key: 's2_coconut_qmamba_amplitudes', label: 'S2: QMamba → COCONUT Amplitudes' },
        { key: 's5_qhpm_hopfield_threshold', label: 'S5: Hopfield → QHPM Threshold' },
        { key: 's8_neural_zne_qssm_stats', label: 'S8: Q-SSM → Neural ZNE' },
        { key: 's11_alphaqubit_decoder', label: 'S11: AlphaQubit Decoder' },
        { key: 's12_sympflow_qng_geodesic', label: 'S12: SympFlow → QNG' },
        { key: 's18_cayley_dense', label: 'S18: CayleyDense Weights' },
        { key: 's19_coconut_crystallize', label: 'S19: COCONUT Crystallization' },
        { key: 's20_galore_bp_aware', label: 'S20: GaLore ↔ BP Awareness' },
        { key: 's21_qalrc_quls_entropy', label: 'S21: QALRC → QULS Entropy' },
        { key: 's22_bp_qalrc_escape', label: 'S22: BP → QALRC Escape' },
    ] as const;

    return (
        <div className="config-form synergy-form">
            <div className="config-group">
                <h3>Quantum Synergies (S1-S22)</h3>
                <div className="synergy-grid">
                    {synergies.map(({ key, label }) => (
                        <label key={key} className="synergy-toggle">
                            <input
                                type="checkbox"
                                checked={config[key as keyof SynergyConfig]}
                                onChange={(e) => onChange({ ...config, [key]: e.target.checked })}
                            />
                            <span>{label}</span>
                        </label>
                    ))}
                </div>
            </div>
            <button className="save-btn" onClick={onSave} disabled={saving}>
                {saving ? 'Saving...' : 'Save Synergy Config'}
            </button>
        </div>
    );
}

function ArchitectureConfigForm({
    config,
    onChange,
    onSave,
    saving,
}: {
    config: ArchitectureConfig;
    onChange: (c: ArchitectureConfig) => void;
    onSave: () => void;
    saving: boolean;
}) {
    return (
        <div className="config-form">
            <div className="config-group">
                <h3>COCONUT Parameters</h3>
                <label>
                    Max Thought Steps:
                    <input
                        type="number"
                        value={config.coconut_max_thought_steps}
                        onChange={(e) => onChange({ ...config, coconut_max_thought_steps: parseInt(e.target.value) })}
                        min={1}
                        max={32}
                    />
                </label>
                <label>
                    BFS Branches:
                    <input
                        type="number"
                        value={config.coconut_bfs_branches}
                        onChange={(e) => onChange({ ...config, coconut_bfs_branches: parseInt(e.target.value) })}
                        min={1}
                        max={16}
                    />
                </label>
                <label>
                    Crystallize Threshold:
                    <input
                        type="number"
                        value={config.coconut_crystallize_threshold}
                        onChange={(e) => onChange({ ...config, coconut_crystallize_threshold: parseFloat(e.target.value) })}
                        step={0.05}
                        min={0.5}
                        max={0.99}
                    />
                </label>
            </div>

            <div className="config-group">
                <h3>Memory & State</h3>
                <label>
                    <input
                        type="checkbox"
                        checked={config.use_holographic_state_binding}
                        onChange={(e) => onChange({ ...config, use_holographic_state_binding: e.target.checked })}
                    />
                    Holographic State Binding
                </label>
                <label>
                    <input
                        type="checkbox"
                        checked={config.use_state_bus}
                        onChange={(e) => onChange({ ...config, use_state_bus: e.target.checked })}
                    />
                    State Bus
                </label>
                <label>
                    <input
                        type="checkbox"
                        checked={config.use_unified_quantum_bus}
                        onChange={(e) => onChange({ ...config, use_unified_quantum_bus: e.target.checked })}
                    />
                    Unified Quantum Bus
                </label>
            </div>

            <div className="config-group">
                <h3>Token Shift Mode</h3>
                <select
                    value={config.token_shift_mode}
                    onChange={(e) => onChange({ ...config, token_shift_mode: e.target.value })}
                >
                    <option value="data_dependent">Data-Dependent</option>
                    <option value="simplified">Simplified</option>
                    <option value="fourier">Fourier</option>
                    <option value="delta">Delta</option>
                    <option value="hierarchical">Hierarchical</option>
                    <option value="multiposition">Multi-Position</option>
                </select>
            </div>

            <button className="save-btn" onClick={onSave} disabled={saving}>
                {saving ? 'Saving...' : 'Save Architecture Config'}
            </button>
        </div>
    );
}

export default QuantumConfigPanel;
