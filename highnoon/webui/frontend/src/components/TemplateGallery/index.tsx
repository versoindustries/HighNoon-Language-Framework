// TemplateGallery - Pre-certified training recipes (Baseline Curriculums)
// Allows users to quickly start with optimized configurations for various use cases

import { useState, useEffect } from 'react';
import {
    Code, Brain, Zap, Sparkles, Check, Clock, Database,
    ChevronRight, Star, MessageCircle, BookOpen, Wand2, Target, PenTool
} from 'lucide-react';
import { Card, Button, Modal } from '../ui';
import './TemplateGallery.css';

// =============================================================================
// TEMPLATE DEFINITIONS - BASELINE CURRICULUMS
// =============================================================================

export interface CurriculumStageConfig {
    name: string;
    module: string;
    epochs: number;
    learningRate: string;
    datasets: string[];
}

export interface TrainingTemplate {
    id: string;
    name: string;
    description: string;
    longDescription?: string;
    icon: React.ReactNode;
    category: 'chat' | 'code' | 'reasoning' | 'instruction' | 'creative' | 'general' | 'quick';
    difficulty: 'beginner' | 'intermediate' | 'advanced';
    estimatedTime: string;
    datasets: string[];
    stages?: CurriculumStageConfig[];
    config: {
        vocabSize: number;
        contextWindow: number;
        epochs: number;
        batchSize: number;
        learningRate: string;
        optimizer: string;
        hpoTrials: number;
    };
    recommended?: boolean;
    tags?: string[];
}

/**
 * Helper function to convert a training template to CurriculumStage format.
 * Used when loading baseline curriculums into the Curriculum Builder.
 */
export function templateToCurriculumStages(template: TrainingTemplate): Array<{
    name: string;
    display_name: string;
    module: string;
    datasets: Array<{ dataset_id: string; weight: number }>;
    epochs: number;
    learning_rate: string;
    batch_size: number;
    weight: number;
}> {
    if (!template.stages || template.stages.length === 0) {
        // If no stages defined, create a single stage from the template config
        return [{
            name: template.id.replace(/-/g, '_'),
            display_name: template.name,
            module: template.category === 'quick' ? 'language_modeling' : template.category,
            datasets: template.datasets.map(ds => ({ dataset_id: ds, weight: 1.0 })),
            epochs: template.config.epochs,
            learning_rate: template.config.learningRate,
            batch_size: template.config.batchSize,
            weight: 1.0,
        }];
    }

    // Convert each stage from the template
    return template.stages.map((stage, index) => ({
        name: stage.name,
        display_name: stage.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        module: stage.module,
        datasets: stage.datasets.map(ds => ({ dataset_id: ds, weight: 1.0 })),
        epochs: stage.epochs,
        learning_rate: stage.learningRate,
        batch_size: template.config.batchSize,
        weight: 1.0 - (index * 0.1), // Slightly decrease weights for later stages
    }));
}

export const TEMPLATES: TrainingTemplate[] = [
    // =========================================================================
    // VERSO BASELINE - THE GOLD STANDARD CURRICULUM (FRONTIER MODEL PARITY)
    // =========================================================================
    {
        id: 'verso-baseline',
        name: 'Verso Baseline',
        description: 'Ultimate Capability: 180+ datasets spanning code, engineering, quantum, finance, medical, legal, and more.',
        longDescription: 'The Verso Baseline represents ultimate capability in LLM training curricula. It incorporates 180+ meticulously curated datasets across 12 progressive training stages, providing comprehensive coverage of: (1) 50T+ pretraining tokens (FineWeb, Dolma, Common Corpus), (2) multilingual (500+ languages), (3) complete Stack coding + DeepSeek, (4) engineering (electrical, mechanical, aerospace, robotics, CAD), (5) quantum computing & quantum mechanics (QST, ChemBench, quantum-ML), (6) comprehensive chat (WizardLM, ShareGPT, Capybara), (7) synthetic data (Magpie, OpenMathInstruct 14M), (8) advanced reasoning + R1 distillation, (9) 20M+ math problems, (10) medical/healthcare (MedQA, MedMCQA), (11) legal (case-law, legalbench), (12) finance (FinBERT, FinanceBench), (13) creative writing/roleplay, (14) long-context (64K+), (15) safety/red-teaming (WildJailbreak, Constitutional AI), and (16) frontier DPO/RLHF. This curriculum produces models with capabilities comparable to GPT-4, Claude, and Gemini.',
        icon: <Sparkles size={24} />,
        category: 'general',
        difficulty: 'advanced',
        estimatedTime: '200-400 hours',
        tags: ['ultimate', 'frontier', 'gold-standard', 'flagship', 'engineering', 'quantum', 'finance', 'medical', 'legal', 'agentic', 'tool-calling', 'claude-code', 'codex-cli', 'multilingual', 'dpo', 'rlhf'],
        datasets: [], // Loaded dynamically from API
        stages: [],   // Loaded dynamically from API
        config: {
            vocabSize: 64000,
            contextWindow: 4194304, // 4M context for Lite max
            epochs: 23,
            batchSize: 4,
            learningRate: '5e-5',
            optimizer: 'sophiag',
            hpoTrials: 50,
        },
        recommended: true,
    },

    // =========================================================================
    // CHAT / CONVERSATIONAL
    // =========================================================================
    {
        id: 'chat-conversational',
        name: 'Chat / Conversational',
        description: 'Multi-turn dialogue with persona consistency and empathetic responses.',
        longDescription: 'Train a model optimized for natural conversation. Includes stages for dialogue understanding, persona consistency, emotional intelligence, and multi-turn context management.',
        icon: <MessageCircle size={24} />,
        category: 'chat',
        difficulty: 'intermediate',
        estimatedTime: '3-6 hours',
        tags: ['dialogue', 'persona', 'empathy', 'multi-turn'],
        datasets: [
            'OpenAssistant/oasst1',
            'HuggingFaceH4/ultrachat_200k',
            'Anthropic/hh-rlhf',
        ],
        stages: [
            { name: 'foundation', module: 'language_modeling', epochs: 1, learningRate: '1e-4', datasets: ['openwebtext'] },
            { name: 'dialogue', module: 'chat', epochs: 2, learningRate: '5e-5', datasets: ['OpenAssistant/oasst1'] },
            { name: 'empathy', module: 'alignment', epochs: 1, learningRate: '2e-5', datasets: ['Anthropic/hh-rlhf'] },
        ],
        config: {
            vocabSize: 32000,
            contextWindow: 4096,
            epochs: 4,
            batchSize: 8,
            learningRate: '5e-5',
            optimizer: 'sophiag',
            hpoTrials: 20,
        },
        recommended: true,
    },

    // =========================================================================
    // CODE GENERATION
    // =========================================================================
    {
        id: 'code-generation',
        name: 'Code Generation',
        description: 'Programming, code completion, documentation, and debugging assistance.',
        longDescription: 'Comprehensive code training curriculum covering multiple programming languages, code completion, documentation generation, bug detection, and code explanation.',
        icon: <Code size={24} />,
        category: 'code',
        difficulty: 'intermediate',
        estimatedTime: '4-8 hours',
        tags: ['programming', 'completion', 'debugging', 'documentation'],
        datasets: [
            'bigcode/the-stack-dedup',
            'bigcode/starcoderdata',
            'codeparrot/github-code',
        ],
        stages: [
            { name: 'code_foundation', module: 'code_generation', epochs: 2, learningRate: '1e-4', datasets: ['bigcode/the-stack-dedup'] },
            { name: 'code_completion', module: 'code_generation', epochs: 2, learningRate: '5e-5', datasets: ['bigcode/starcoderdata'] },
            { name: 'code_instruction', module: 'instruction_tuning', epochs: 1, learningRate: '2e-5', datasets: ['codeparrot/github-code'] },
        ],
        config: {
            vocabSize: 50000,
            contextWindow: 8192,
            epochs: 5,
            batchSize: 8,
            learningRate: '5e-5',
            optimizer: 'sophiag',
            hpoTrials: 25,
        },
        recommended: true,
    },

    // =========================================================================
    // REASONING & LOGIC
    // =========================================================================
    {
        id: 'reasoning-logic',
        name: 'Reasoning & Logic',
        description: 'Chain-of-thought, math problems, logical deduction, and step-by-step problem solving.',
        longDescription: 'Develop strong reasoning capabilities through chain-of-thought training, mathematical reasoning, logical deduction, and systematic problem decomposition.',
        icon: <Brain size={24} />,
        category: 'reasoning',
        difficulty: 'advanced',
        estimatedTime: '6-10 hours',
        tags: ['chain-of-thought', 'math', 'logic', 'problem-solving'],
        datasets: [
            'gsm8k',
            'meta-math/MetaMathQA',
            'nvidia/OpenMathInstruct-1',
            'allenai/ai2_arc',
        ],
        stages: [
            { name: 'foundation', module: 'language_modeling', epochs: 1, learningRate: '1e-4', datasets: ['openwebtext'] },
            { name: 'math_reasoning', module: 'reasoning', epochs: 3, learningRate: '5e-5', datasets: ['gsm8k', 'meta-math/MetaMathQA'] },
            { name: 'logic_deduction', module: 'reasoning', epochs: 2, learningRate: '3e-5', datasets: ['allenai/ai2_arc'] },
            { name: 'cot_refinement', module: 'reasoning', epochs: 1, learningRate: '2e-5', datasets: ['nvidia/OpenMathInstruct-1'] },
        ],
        config: {
            vocabSize: 32000,
            contextWindow: 4096,
            epochs: 7,
            batchSize: 4,
            learningRate: '3e-5',
            optimizer: 'sophiag',
            hpoTrials: 40,
        },
    },

    // =========================================================================
    // INSTRUCTION FOLLOWING
    // =========================================================================
    {
        id: 'instruction-following',
        name: 'Instruction Following',
        description: 'Task completion, following complex instructions, and tool use capabilities.',
        longDescription: 'Train for robust instruction following across diverse task types. Includes function calling, tool use, complex multi-step instructions, and format adherence.',
        icon: <Target size={24} />,
        category: 'instruction',
        difficulty: 'intermediate',
        estimatedTime: '3-5 hours',
        tags: ['task-completion', 'tool-use', 'function-calling', 'format'],
        datasets: [
            'databricks/dolly-15k',
            'OpenAssistant/oasst1',
            'HuggingFaceH4/no_robots',
            'glaive-coder/function-calling-v2',
        ],
        stages: [
            { name: 'instruction_base', module: 'instruction_tuning', epochs: 2, learningRate: '1e-4', datasets: ['databricks/dolly-15k'] },
            { name: 'task_diversity', module: 'instruction_tuning', epochs: 2, learningRate: '5e-5', datasets: ['HuggingFaceH4/no_robots'] },
            { name: 'function_calling', module: 'instruction_tuning', epochs: 1, learningRate: '3e-5', datasets: ['glaive-coder/function-calling-v2'] },
        ],
        config: {
            vocabSize: 32000,
            contextWindow: 4096,
            epochs: 5,
            batchSize: 8,
            learningRate: '5e-5',
            optimizer: 'sophiag',
            hpoTrials: 20,
        },
    },

    // =========================================================================
    // CREATIVE WRITING
    // =========================================================================
    {
        id: 'creative-writing',
        name: 'Creative Writing',
        description: 'Storytelling, poetry, creative text generation, and narrative coherence.',
        longDescription: 'Develop creative writing capabilities including long-form storytelling, poetry generation, dialogue writing, world-building, and maintaining narrative coherence across extended texts.',
        icon: <PenTool size={24} />,
        category: 'creative',
        difficulty: 'intermediate',
        estimatedTime: '4-6 hours',
        tags: ['storytelling', 'poetry', 'narrative', 'fiction'],
        datasets: [
            'roneneldan/TinyStories',
            'euclaise/writingprompts',
            'Gustavosta/Stable-Diffusion-Prompts',
        ],
        stages: [
            { name: 'narrative_foundation', module: 'language_modeling', epochs: 2, learningRate: '1e-4', datasets: ['roneneldan/TinyStories'] },
            { name: 'creative_expansion', module: 'language_modeling', epochs: 2, learningRate: '5e-5', datasets: ['euclaise/writingprompts'] },
            { name: 'style_refinement', module: 'language_modeling', epochs: 1, learningRate: '2e-5', datasets: ['Gustavosta/Stable-Diffusion-Prompts'] },
        ],
        config: {
            vocabSize: 32000,
            contextWindow: 8192,
            epochs: 5,
            batchSize: 4,
            learningRate: '5e-5',
            optimizer: 'sophiag',
            hpoTrials: 15,
        },
    },

    // =========================================================================
    // GENERAL PURPOSE
    // =========================================================================
    {
        id: 'general-purpose',
        name: 'General Purpose',
        description: 'Balanced curriculum for versatile language understanding across all domains.',
        longDescription: 'A comprehensive, balanced training curriculum that develops capabilities across multiple domains including conversation, instruction following, reasoning, and general knowledge.',
        icon: <Sparkles size={24} />,
        category: 'general',
        difficulty: 'intermediate',
        estimatedTime: '5-8 hours',
        tags: ['balanced', 'versatile', 'all-purpose', 'foundation'],
        datasets: [
            'openwebtext',
            'databricks/dolly-15k',
            'OpenAssistant/oasst1',
            'gsm8k',
        ],
        stages: [
            { name: 'foundation', module: 'language_modeling', epochs: 2, learningRate: '1e-4', datasets: ['openwebtext'] },
            { name: 'instruction', module: 'instruction_tuning', epochs: 2, learningRate: '5e-5', datasets: ['databricks/dolly-15k'] },
            { name: 'dialogue', module: 'chat', epochs: 1, learningRate: '3e-5', datasets: ['OpenAssistant/oasst1'] },
            { name: 'reasoning', module: 'reasoning', epochs: 1, learningRate: '2e-5', datasets: ['gsm8k'] },
        ],
        config: {
            vocabSize: 32000,
            contextWindow: 4096,
            epochs: 6,
            batchSize: 8,
            learningRate: '5e-5',
            optimizer: 'sophiag',
            hpoTrials: 25,
        },
        recommended: true,
    },

    // =========================================================================
    // QUICK TEST
    // =========================================================================
    {
        id: 'quick-test',
        name: 'Quick Test',
        description: 'Minimal configuration for fast iteration and testing. Validate your setup quickly.',
        longDescription: 'A lightweight configuration designed for rapid testing and validation. Perfect for checking your environment, data pipeline, and training setup before committing to longer runs.',
        icon: <Zap size={24} />,
        category: 'quick',
        difficulty: 'beginner',
        estimatedTime: '15-30 min',
        tags: ['testing', 'validation', 'quick', 'minimal'],
        datasets: [],
        stages: [
            { name: 'quick_train', module: 'language_modeling', epochs: 1, learningRate: '1e-4', datasets: [] },
        ],
        config: {
            vocabSize: 16000,
            contextWindow: 2048,
            epochs: 1,
            batchSize: 16,
            learningRate: '1e-4',
            optimizer: 'sophiag',
            hpoTrials: 5,
        },
    },
];

// =============================================================================
// TEMPLATE CARD COMPONENT
// =============================================================================

interface TemplateCardProps {
    template: TrainingTemplate;
    selected: boolean;
    onSelect: () => void;
}

function TemplateCard({ template, selected, onSelect }: TemplateCardProps) {
    const difficultyColors = {
        beginner: 'var(--color-success)',
        intermediate: 'var(--color-warning)',
        advanced: 'var(--color-danger)',
    };

    return (
        <button
            className={`template-card ${selected ? 'template-card-selected' : ''}`}
            onClick={onSelect}
        >
            {template.recommended && (
                <div className="template-badge">
                    <Star size={12} /> Recommended
                </div>
            )}
            <div className="template-icon" style={{ color: selected ? '#fff' : 'var(--color-primary)' }}>
                {template.icon}
            </div>
            <div className="template-content">
                <h4 className="template-name">{template.name}</h4>
                <p className="template-desc">{template.description}</p>
                <div className="template-meta">
                    <span className="template-time">
                        <Clock size={12} /> {template.estimatedTime}
                    </span>
                    <span
                        className="template-difficulty"
                        style={{ color: difficultyColors[template.difficulty] }}
                    >
                        {template.difficulty}
                    </span>
                </div>
            </div>
            {selected && (
                <div className="template-check">
                    <Check size={16} />
                </div>
            )}
        </button>
    );
}

// =============================================================================
// TEMPLATE DETAILS PANEL
// =============================================================================

interface TemplateDetailsProps {
    template: TrainingTemplate;
    onUse: () => void;
}

function TemplateDetails({ template, onUse }: TemplateDetailsProps) {
    return (
        <div className="template-details">
            <div className="template-details-header">
                <div className="template-details-icon">{template.icon}</div>
                <div>
                    <h3>{template.name}</h3>
                    <p>{template.longDescription || template.description}</p>
                </div>
            </div>

            {/* Tags */}
            {template.tags && template.tags.length > 0 && (
                <div className="template-tags">
                    {template.tags.map((tag) => (
                        <span key={tag} className="template-tag">{tag}</span>
                    ))}
                </div>
            )}

            {/* Training Stages */}
            {template.stages && template.stages.length > 0 && (
                <div className="template-stages">
                    <h4>Training Stages</h4>
                    <div className="stages-timeline">
                        {template.stages.map((stage, index) => (
                            <div key={stage.name} className="stage-timeline-item">
                                <div className="stage-timeline-marker">
                                    <span className="stage-number">{index + 1}</span>
                                </div>
                                <div className="stage-timeline-content">
                                    <span className="stage-name">{stage.name.replace(/_/g, ' ')}</span>
                                    <span className="stage-module">{stage.module.replace(/_/g, ' ')}</span>
                                    <div className="stage-meta">
                                        <span>{stage.epochs} epoch{stage.epochs > 1 ? 's' : ''}</span>
                                        <span>LR: {stage.learningRate}</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <div className="template-config">
                <h4>Configuration</h4>
                <div className="config-grid">
                    <div className="config-item">
                        <span className="config-label">Vocab Size</span>
                        <span className="config-value">{template.config.vocabSize.toLocaleString()}</span>
                    </div>
                    <div className="config-item">
                        <span className="config-label">Context Window</span>
                        <span className="config-value">{template.config.contextWindow.toLocaleString()}</span>
                    </div>
                    <div className="config-item">
                        <span className="config-label">Total Epochs</span>
                        <span className="config-value">{template.config.epochs}</span>
                    </div>
                    <div className="config-item">
                        <span className="config-label">Batch Size</span>
                        <span className="config-value">{template.config.batchSize}</span>
                    </div>
                    <div className="config-item">
                        <span className="config-label">Base LR</span>
                        <span className="config-value">{template.config.learningRate}</span>
                    </div>
                    <div className="config-item">
                        <span className="config-label">Optimizer</span>
                        <span className="config-value">{template.config.optimizer}</span>
                    </div>
                    <div className="config-item">
                        <span className="config-label">HPO Trials</span>
                        <span className="config-value">{template.config.hpoTrials}</span>
                    </div>
                    <div className="config-item">
                        <span className="config-label">Est. Time</span>
                        <span className="config-value">{template.estimatedTime}</span>
                    </div>
                </div>
            </div>

            {template.datasets.length > 0 && (
                <div className="template-datasets">
                    <h4>Included Datasets</h4>
                    <div className="dataset-list">
                        {template.datasets.map((ds) => (
                            <div key={ds} className="dataset-chip">
                                <Database size={12} /> {ds}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <Button variant="primary" fullWidth onClick={onUse}>
                Use This Curriculum <ChevronRight size={16} />
            </Button>
        </div>
    );
}

// =============================================================================
// MAIN GALLERY COMPONENT
// =============================================================================

interface TemplateGalleryProps {
    open: boolean;
    onClose: () => void;
    onSelectTemplate: (template: TrainingTemplate) => void;
}

// Filter categories for the gallery
const FILTER_CATEGORIES = [
    { id: 'all', label: 'All Curriculums' },
    { id: 'chat', label: 'Chat' },
    { id: 'code', label: 'Code' },
    { id: 'reasoning', label: 'Reasoning' },
    { id: 'instruction', label: 'Instruction' },
    { id: 'creative', label: 'Creative' },
    { id: 'general', label: 'General' },
    { id: 'quick', label: 'Quick Test' },
];

export function TemplateGallery({ open, onClose, onSelectTemplate }: TemplateGalleryProps) {
    const [templates, setTemplates] = useState<TrainingTemplate[]>(TEMPLATES);
    const [selectedId, setSelectedId] = useState<string | null>(TEMPLATES[0].id);
    const [filter, setFilter] = useState<string>('all');

    // Fetch baseline presets from backend API on mount
    useEffect(() => {
        const fetchPresets = async () => {
            try {
                // @ts-ignore - Assuming fetch or global API client is available
                const response = await fetch('/api/curricula/presets');
                if (response.ok) {
                    const data = await response.json();
                    if (data.presets && Array.isArray(data.presets)) {
                        const presets = data.presets;

                        // Merge API presets into local templates
                        // Specifically look for 'verso-baseline' to update it
                        setTemplates(prevTemplates => {
                            return prevTemplates.map(t => {
                                const apiPreset = presets.find((p: any) => p.id === t.id);
                                if (apiPreset) {
                                    // Update datasets and reconstruct stages if needed
                                    // Note: The API returns flat datasets list + stages info
                                    // We'll trust the API as the single source of truth for datasets

                                    // Helper to convert API stages to UI stages config if needed
                                    // For now, we mainly want to ensure the datasets list is correct
                                    return {
                                        ...t,
                                        datasets: apiPreset.hf_datasets || [],
                                        description: apiPreset.description || t.description,
                                        // We could also parse 'stage_count' to verify match
                                    };
                                }
                                return t;
                            });
                        });
                        console.log(`[TemplateGallery] Loaded ${presets.length} presets from API`);
                    }
                }
            } catch (error) {
                console.error("Failed to fetch curriculum presets:", error);
            }
        };

        fetchPresets();
    }, []);

    const selectedTemplate = templates.find((t) => t.id === selectedId) || null;

    const filteredTemplates = filter === 'all'
        ? templates
        : templates.filter((t) => t.category === filter);

    const handleUse = () => {
        if (selectedTemplate) {
            onSelectTemplate(selectedTemplate);
            onClose();
        }
    };

    return (
        <Modal open={open} onClose={onClose} title="Baseline Curriculum Gallery" size="xl">
            <div className="template-gallery">
                <div className="template-gallery-sidebar">
                    <div className="template-filters">
                        {FILTER_CATEGORIES.map((cat) => (
                            <button
                                key={cat.id}
                                className={`filter-chip ${filter === cat.id ? 'filter-chip-active' : ''}`}
                                onClick={() => setFilter(cat.id)}
                            >
                                {cat.label}
                            </button>
                        ))}
                    </div>

                    <div className="template-list">
                        {filteredTemplates.map((template) => (
                            <TemplateCard
                                key={template.id}
                                template={template}
                                selected={selectedId === template.id}
                                onSelect={() => setSelectedId(template.id)}
                            />
                        ))}
                    </div>
                </div>

                <div className="template-gallery-main">
                    {selectedTemplate ? (
                        <TemplateDetails template={selectedTemplate} onUse={handleUse} />
                    ) : (
                        <div className="template-empty">
                            <p>Select a template to view details</p>
                        </div>
                    )}
                </div>
            </div>
        </Modal>
    );
}

export default TemplateGallery;
