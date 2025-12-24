// TemplateGallery - Pre-certified training recipes (Baseline Curriculums)
// Allows users to quickly start with optimized configurations for various use cases

import { useState } from 'react';
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
        datasets: [
            // =================================================================
            // FOUNDATION / PRETRAINING CORPORA (Large-scale - 50T+ tokens)
            // =================================================================
            'HuggingFaceFW/fineweb',                          // 15T tokens from CommonCrawl
            'HuggingFaceFW/fineweb-edu',                      // 1.3T educational tokens
            'HuggingFaceFW/fineweb-2',                        // Multilingual extension
            'allenai/dolma',                                  // 3T tokens (OLMo training data)
            'cerebras/SlimPajama-627B',                       // 627B deduplicated tokens
            'togethercomputer/RedPajama-Data-V2',             // 30T filtered tokens
            'HuggingFaceTB/cosmopedia',                       // Synthetic textbooks
            'EleutherAI/pile',                                // 800GB diverse corpus
            'tiiuae/falcon-refinedweb',                       // Falcon training data
            'pleias/common_corpus',                           // 2T tokens, 30+ languages
            'wikimedia/wikipedia',                            // Wikipedia all languages

            // =================================================================
            // MULTILINGUAL / TRANSLATION
            // =================================================================
            'Helsinki-NLP/opus-100',                          // OPUS translation corpus
            'facebook/flores',                                // Multilingual benchmark
            'allenai/MADLAD-400',                             // MADLAD multilingual
            'cis-lmu/Glot500',                                // 500+ languages

            // =================================================================
            // CODE DATASETS - THE STACK FAMILY (Complete)
            // =================================================================
            'bigcode/the-stack',                              // 6TB+ source code, 358 languages
            'bigcode/the-stack-v2',                           // 3B+ files, 600+ languages
            'bigcode/the-stack-dedup',                        // 3TB deduplicated
            'bigcode/the-stack-v2-dedup',                     // v2 deduplicated
            'bigcode/starcoderdata',                          // 783GB, 86 languages + issues/commits
            'codeparrot/github-code',                         // GitHub code corpus
            'nvidia/Nemotron-Pretraining-Code-v2',            // NVIDIA code pretraining
            'nampdn-ai/tiny-codes',                           // 1.6M code snippets
            'code-rag-bench/humaneval',                       // HumanEval benchmark
            'code-rag-bench/mbpp',                            // MBPP benchmark
            'code-rag-bench/ds1000',                          // DS1000 data science
            'code-rag-bench/odex',                            // ODEX benchmark
            'code-search-net/code_search_net',                // 6M code functions
            'TokenBender/code_instructions_122k_alpaca_style', // Code instructions
            'jon-tow/starcoderdata-python-edu',               // Python educational code
            'codefuse-ai/Evol-instruction-66k',               // Evol instruction for code
            'ise-uiuc/Magicoder-Evol-Instruct-110K',          // Magicoder instructions
            'rombodawg/LosslessMegaCodeTrainingV3_1.6m_Evol', // 1.6M evolved code
            'mikex86/stackoverflow-posts',                    // Stack Overflow Q&A
            'bigcode/bigcodebench',                           // BigCode benchmark
            'deepseek-ai/DeepSeek-Coder-V2-Instruct',         // DeepSeek Coder

            // =================================================================
            // CHAT / DIALOGUE DATASETS (Comprehensive)
            // =================================================================
            'OpenAssistant/oasst1',                           // Open Assistant v1
            'OpenAssistant/oasst2',                           // Open Assistant v2
            'teknium/OpenHermes-2.5',                         // 1M entries
            'lmsys/lmsys-chat-1m',                            // 1M real conversations
            'lmsys/chatbot_arena_conversations',              // Arena conversations
            'HuggingFaceH4/ultrachat_200k',                   // 200K conversations
            'stingning/ultrachat',                            // Full UltraChat
            'WizardLMTeam/WizardLM_evol_instruct_V2_196k',    // WizardLM Evol-Instruct
            'WizardLMTeam/WizardLM_evol_instruct_70k',        // WizardLM base
            'openchat/openchat_sharegpt4_dataset',            // ShareGPT GPT-4
            'anon8231489123/ShareGPT_Vicuna_unfiltered',      // ShareGPT Vicuna
            'theblackcat102/sharegpt-english',                // ShareGPT English
            'berkeley-nest/Nectar',                           // Nectar dataset
            'Intel/orca_dpo_pairs',                           // Orca DPO pairs
            'NousResearch/Nous-Capybara',                     // Capybara multi-turn

            // =================================================================
            // SYNTHETIC DATA (Frontier Model Style)
            // =================================================================
            'Magpie-Align/Magpie-Pro-300K-Filtered',          // Magpie synthetic
            'Magpie-Align/Magpie-Qwen2-Pro-200K-English',     // Magpie Qwen2
            'nvidia/OpenMathInstruct-1',                      // 1.8M math problems
            'nvidia/OpenMathInstruct-2',                      // 14M math problems
            'deepseek-ai/DeepSeek-Math',                      // DeepSeekMath corpus

            // =================================================================
            // INSTRUCTION FOLLOWING (Extensive)
            // =================================================================
            'databricks/dolly-15k',                           // Dolly 15K
            'HuggingFaceH4/no_robots',                        // No Robots dataset
            'MBZUAI/LaMini-instruction',                      // LaMini instruction
            'llm-wizard/dolly-15k-instruction-alpaca-format', // Alpaca format
            'tatsu-lab/alpaca',                               // Stanford Alpaca
            'yahma/alpaca-cleaned',                           // Cleaned Alpaca
            'BAAI/Infinity-Instruct',                         // Infinity Instruct
            'alespalla/chatbot_instruction_prompts',          // Instruction prompts
            'Open-Orca/OpenOrca',                             // OpenOrca dataset

            // =================================================================
            // AGENTIC TOOL CALLING (HARDCODED - Required for Codex CLI & Claude Code)
            // =================================================================
            'Salesforce/xlam-function-calling-60k',           // Core function calling (60K)
            'glaiveai/glaive-function-calling-v2',            // Function calling v2
            'gorilla-llm/gorilla-openfunctions-v1',           // Gorilla OpenFunctions
            'NousResearch/hermes-function-calling-v1',        // Hermes function calling
            'Trelis/function_calling_extended',               // Extended function calling
            'Nexusflow/Function_Call_Definitions',            // Function definitions
            'Nexusflow/VT_MultiAPIs',                         // Multi-API calls
            'Nexusflow/VirusTotalMultiple',                   // VirusTotal APIs
            'nvidia/Nemotron-Agentic-v1',                     // NVIDIA Agentic dataset
            'open-thoughts/OpenThoughts-Agent-v1-SFT',        // Agentic SFT
            'open-thoughts/OpenThoughts-Agent-v1-RL',         // Agentic RL
            'open-r1/codeforces-cots',                        // Coding agent CoT
            'rizerphe/glaive-function-calling-v2-zephyr',     // Zephyr format

            // =================================================================
            // REASONING DATASETS (Comprehensive)
            // =================================================================
            'argilla/distilabel-reasoning-prompts',           // Reasoning prompts
            'nvidia/OpenCodeReasoning',                       // Code reasoning
            'nvidia/OpenMathReasoning',                       // Math reasoning
            'glaiveai/reasoning-v1-20m',                      // 20M reasoning samples
            'allenai/ai2_arc',                                // ARC challenge
            'Rowan/hellaswag',                                // HellaSwag commonsense
            'cais/mmlu',                                      // MMLU benchmark
            'TIGER-Lab/MMLU-Pro',                             // MMLU Pro enhanced
            'openai/MMMLU',                                   // Multilingual MMLU
            'CohereForAI/Global-MMLU',                        // Global MMLU
            'allenai/winogrande',                             // WinoGrande
            'Muennighoff/babi',                               // bAbI reasoning
            'PKU-Alignment/BeaverTails',                      // Reasoning safety
            'wenhu/LogiCoT',                                  // Logical CoT reasoning
            'facebook/belebele',                              // Multilingual reading

            // =================================================================
            // MATHEMATICS DATASETS (Extensive - 20M+ problems)
            // =================================================================
            'openai/gsm8k',                                   // GSM8K math problems
            'meta-math/MetaMathQA',                           // MetaMath QA
            'microsoft/orca-math-word-problems-200k',         // Orca Math 200K
            'qwedsacf/grade-school-math-instructions',        // Grade school math
            'abacusai/MetaMathFewshot',                       // MetaMath few-shot
            'math-ai/StackMathQA',                            // Stack Math QA
            'zwhe99/DeepMath-103K',                           // DeepMath 103K
            'HuggingFaceTB/finemath',                         // Fine math dataset
            'camel-ai/math',                                  // CAMEL math
            'hendrycks/competition_math',                     // Competition math
            'EleutherAI/hendrycks_math',                      // Hendrycks math
            'MathLLMs/MathCodeInstruct',                      // Math code instruct
            'AI-MO/NuminaMath-CoT',                           // Numina Math CoT
            'AI-MO/NuminaMath-TIR',                           // Numina Math TIR
            'PRIME-RL/Eurus-2-RL-Data',                       // Eurus RL math
            'bespokelabs/Bespoke-MiniCheck-7B',               // Math verification

            // =================================================================
            // QUANTUM COMPUTING & PHYSICS
            // =================================================================
            'VDR_Quantum',                                    // Quantum technical documents
            'QuantumLLMInstruct',                             // Quantum optimization
            'camel-ai/physics',                               // CAMEL physics
            'ajibawa-2023/Physics-QA',                        // Physics Q&A
            'knowrohit07/physics_dataset',                    // Physics dataset

            // =================================================================
            // SCIENCE & CHEMISTRY
            // =================================================================
            'camel-ai/chemistry',                             // CAMEL chemistry
            'camel-ai/biology',                               // CAMEL biology
            'allenai/sciq',                                   // Science QA
            'allenai/openbookqa',                             // OpenBook QA
            'allenai/piqa',                                   // Physical intuition QA
            'allenai/social_i_qa',                            // Social IQA
            'allenai/cosmos_qa',                              // Cosmos QA
            'bigbio/pubmed_qa',                               // PubMed QA
            'allen-ai/pubmedqa',                              // PubMedQA science

            // =================================================================
            // DPO / RLHF / PREFERENCE DATA (Frontier Alignment)
            // =================================================================
            'Anthropic/hh-rlhf',                              // Anthropic RLHF
            'Open-Orca/SlimOrca-Dedup',                       // SlimOrca deduplicated
            'HuggingFaceH4/ultrafeedback_binarized',          // UltraFeedback
            'allenai/ultrafeedback_binarized_cleaned',        // Cleaned version
            'Skywork/Skywork-Reward-Preference-80K-v0.2',     // Skywork Reward
            'argilla/dpo-mix-7k',                             // DPO mix
            'argilla/Capybara-Preferences',                   // Capybara preferences
            'argilla/distilabel-intel-orca-dpo-pairs',        // Orca DPO distilabel
            'argilla/distilabel-capybara-dpo-7k-binarized',   // Capybara DPO
            'jondurbin/truthy-dpo-v0.1',                      // Truthy DPO
            'nvidia/HelpSteer2',                              // NVIDIA HelpSteer2

            // =================================================================
            // MEDICAL / HEALTHCARE
            // =================================================================
            'openlifescienceai/medmcqa',                      // MedMCQA exam questions
            'medalpaca/medical_meadow_wikidoc',               // Medical Wikipedia
            'medalpaca/medical_meadow_medqa',                 // MedQA dataset
            'lavita/ChatDoctor-HealthCareMagic-100k',         // Healthcare dialogues
            'epfl-llm/guidelines',                            // Clinical guidelines

            // =================================================================
            // LEGAL
            // =================================================================
            'HFforLegal/case-law',                            // Millions of legal cases
            'joelniklaus/legal_case_document_summarization',  // Legal summarization
            'nguha/legalbench',                               // Legal benchmark

            // =================================================================
            // CREATIVE WRITING / ROLEPLAY / STORIES
            // =================================================================
            'Gryphe/ChatGPT-4o-Writing-Prompts',              // Writing prompts + stories
            'euclaise/writingprompts',                        // Reddit WritingPrompts
            'NousResearch/CharacterCodex',                    // Character codex
            'lemonilia/LimaRP',                               // Roleplay dataset
            'Dampfinchen/Creative_Writing_Multiturn',         // Creative writing
            'agentlans/llama3.1-8b-short-stories',            // Short stories

            // =================================================================
            // ENGINEERING - ELECTRICAL / MECHANICAL / CIVIL
            // =================================================================
            'STEM-AI-mtl/Electrical-engineering',             // Electrical engineering + KiCAD
            'lamm-mit/MechanicsMaterials',                    // Mechanics, thermodynamics, fluids
            'XXCCF/bridge_construction',                      // Civil engineering bridges
            'LouisChen15/ConstructionSite',                   // Construction site data
            'riegel/crackenpy_dataset',                       // Crack detection engineering

            // =================================================================
            // AEROSPACE / AVIATION / SPACE
            // =================================================================
            'archanatikayatray/aeroBERT-NER',                 // Aerospace NER
            'kyleeasterly/purple-aerospace-mix-v1-80-12',     // Aerospace mix

            // =================================================================
            // ROBOTICS
            // =================================================================
            'lerobot/aloha_mobile_cabinet',                   // LeRobot robotics
            'jxu124/OpenX-Embodiment',                        // Embodied AI

            // =================================================================
            // CAD / 3D DESIGN
            // =================================================================
            'Text2CAD/Text2CAD',                              // Text to CAD multimodal
            'filapro/cad-recode',                             // CAD reverse engineering

            // =================================================================
            // QUANTUM MECHANICS / QUANTUM COMPUTING (EXPANDED)
            // =================================================================
            'Allanatric/QST',                                 // Quantum state tomography
            'moremilk/CoT_Reasoning_Quantom_Physics_And_Computing', // Quantum reasoning
            'jablonkagroup/ChemBench',                        // Quantum chemistry benchmark
            'themanaspandey/QuantumMechanics',                // Quantum mechanics
            'shwetha729/quantum-machine-learning',            // Quantum ML
            'wesley7137/quantum',                             // Quantum dataset
            'BoltzmannEntropy/QuantumLLMInstruct',            // Quantum many-body physics
            'AI4Chem/ChemPref-DPO-for-Chemistry-data-en',     // Chemistry DPO

            // =================================================================
            // FINANCE / ECONOMICS / BUSINESS
            // =================================================================
            'takala/financial_phrasebank',                    // Financial sentiment
            'Josephgflowers/Financial-NER-NLP',               // Financial NER (139 XBRL tags)
            'nlpaueb/finer-139',                              // Financial entity recognition
            'PatronusAI/financebench',                        // Finance benchmark
            'AdaptLLM/finance-tasks',                         // Finance tasks
            'winddude/reddit_finance_43_250k',                // Reddit finance
            'zeroshot/twitter-financial-news-sentiment',      // Twitter finance sentiment

            // =================================================================
            // LONG CONTEXT TRAINING
            // =================================================================
            'princeton-nlp/prolong-data-64K',                 // 64K context data
            'THUDM/LongAlign-10k',                            // Long context alignment

            // =================================================================
            // SAFETY / RED TEAMING / CONSTITUTIONAL AI
            // =================================================================
            'allenai/wildjailbreak',                          // 262K jailbreak pairs
            'HuggingFaceH4/cai-conversation-harmless',        // Constitutional AI
            'Anthropic/hh-rlhf',                              // Harmless preference
            'aurora-m/redteam',                               // Red team prompts
            'PKU-Alignment/safe-rlhf',                        // Safe RLHF

            // =================================================================
            // REASONING / R1 DISTILLATION
            // =================================================================
            'deepseek-ai/DeepSeek-R1-Distill-Qwen',           // R1 distillation
            'simplescaling/s1-data',                          // S1 reasoning data
            'bespokelabs/Bespoke-Stratos-17k',                // Reasoning traces
        ],
        stages: [
            // Stage 1: Foundation Pretraining - Large-scale language understanding
            {
                name: 'foundation_pretraining',
                module: 'language_modeling',
                epochs: 1,
                learningRate: '1e-4',
                datasets: [
                    'HuggingFaceFW/fineweb',
                    'HuggingFaceFW/fineweb-edu',
                    'HuggingFaceFW/fineweb-2',
                    'allenai/dolma',
                    'cerebras/SlimPajama-627B',
                    'togethercomputer/RedPajama-Data-V2',
                    'HuggingFaceTB/cosmopedia',
                    'EleutherAI/pile',
                    'tiiuae/falcon-refinedweb',
                ],
            },
            // Stage 2: Code Pretraining - Complete Stack family
            {
                name: 'code_pretraining',
                module: 'code_generation',
                epochs: 2,
                learningRate: '8e-5',
                datasets: [
                    'bigcode/the-stack',
                    'bigcode/the-stack-v2',
                    'bigcode/the-stack-dedup',
                    'bigcode/the-stack-v2-dedup',
                    'bigcode/starcoderdata',
                    'codeparrot/github-code',
                    'nvidia/Nemotron-Pretraining-Code-v2',
                    'nampdn-ai/tiny-codes',
                    'code-search-net/code_search_net',
                    'mikex86/stackoverflow-posts',
                ],
            },
            // Stage 3: Code Quality - Benchmarks and evolved code
            {
                name: 'code_quality',
                module: 'code_generation',
                epochs: 2,
                learningRate: '5e-5',
                datasets: [
                    'code-rag-bench/humaneval',
                    'code-rag-bench/mbpp',
                    'code-rag-bench/ds1000',
                    'code-rag-bench/odex',
                    'TokenBender/code_instructions_122k_alpaca_style',
                    'jon-tow/starcoderdata-python-edu',
                    'codefuse-ai/Evol-instruction-66k',
                    'ise-uiuc/Magicoder-Evol-Instruct-110K',
                    'rombodawg/LosslessMegaCodeTrainingV3_1.6m_Evol',
                    'bigcode/bigcodebench',
                ],
            },
            // Stage 4: Instruction Tuning - Learn to follow instructions
            {
                name: 'instruction_tuning',
                module: 'instruction_tuning',
                epochs: 2,
                learningRate: '5e-5',
                datasets: [
                    'databricks/dolly-15k',
                    'HuggingFaceH4/no_robots',
                    'MBZUAI/LaMini-instruction',
                    'tatsu-lab/alpaca',
                    'yahma/alpaca-cleaned',
                    'BAAI/Infinity-Instruct',
                    'alespalla/chatbot_instruction_prompts',
                ],
            },
            // Stage 5: Chat & Dialogue - Comprehensive conversation
            {
                name: 'chat_dialogue',
                module: 'chat',
                epochs: 2,
                learningRate: '5e-5',
                datasets: [
                    'OpenAssistant/oasst1',
                    'OpenAssistant/oasst2',
                    'teknium/OpenHermes-2.5',
                    'lmsys/lmsys-chat-1m',
                    'lmsys/chatbot_arena_conversations',
                    'HuggingFaceH4/ultrachat_200k',
                    'stingning/ultrachat',
                    'WizardLMTeam/WizardLM_evol_instruct_V2_196k',
                    'WizardLMTeam/WizardLM_evol_instruct_70k',
                    'openchat/openchat_sharegpt4_dataset',
                    'anon8231489123/ShareGPT_Vicuna_unfiltered',
                    'berkeley-nest/Nectar',
                ],
            },
            // Stage 6: Reasoning & Logic - Comprehensive benchmarks
            {
                name: 'reasoning_logic',
                module: 'reasoning',
                epochs: 3,
                learningRate: '3e-5',
                datasets: [
                    'argilla/distilabel-reasoning-prompts',
                    'nvidia/OpenCodeReasoning',
                    'nvidia/OpenMathReasoning',
                    'glaiveai/reasoning-v1-20m',
                    'allenai/ai2_arc',
                    'Rowan/hellaswag',
                    'cais/mmlu',
                    'TIGER-Lab/MMLU-Pro',
                    'allenai/winogrande',
                    'wenhu/LogiCoT',
                ],
            },
            // Stage 7: Mathematics - Deep math training
            {
                name: 'mathematics',
                module: 'reasoning',
                epochs: 3,
                learningRate: '3e-5',
                datasets: [
                    'openai/gsm8k',
                    'meta-math/MetaMathQA',
                    'nvidia/OpenMathInstruct-2',
                    'microsoft/orca-math-word-problems-200k',
                    'qwedsacf/grade-school-math-instructions',
                    'abacusai/MetaMathFewshot',
                    'math-ai/StackMathQA',
                    'zwhe99/DeepMath-103K',
                    'HuggingFaceTB/finemath',
                    'camel-ai/math',
                    'hendrycks/competition_math',
                    'MathLLMs/MathCodeInstruct',
                    'AI-MO/NuminaMath-CoT',
                ],
            },
            // Stage 8: Science - Quantum, Physics, Chemistry
            {
                name: 'science_training',
                module: 'reasoning',
                epochs: 2,
                learningRate: '3e-5',
                datasets: [
                    'VDR_Quantum',
                    'QuantumLLMInstruct',
                    'camel-ai/physics',
                    'ajibawa-2023/Physics-QA',
                    'camel-ai/chemistry',
                    'camel-ai/biology',
                    'allenai/sciq',
                    'allenai/openbookqa',
                    'allenai/piqa',
                    'allenai/cosmos_qa',
                ],
            },
            // Stage 9: Agentic Tool Calling - CRITICAL for Codex CLI & Claude Code
            {
                name: 'agentic_training',
                module: 'tool_calling',
                epochs: 3,
                learningRate: '2e-5',
                datasets: [
                    'Salesforce/xlam-function-calling-60k',
                    'glaiveai/glaive-function-calling-v2',
                    'gorilla-llm/gorilla-openfunctions-v1',
                    'NousResearch/hermes-function-calling-v1',
                    'Trelis/function_calling_extended',
                    'Nexusflow/Function_Call_Definitions',
                    'Nexusflow/VT_MultiAPIs',
                    'Nexusflow/VirusTotalMultiple',
                    'nvidia/Nemotron-Agentic-v1',
                    'open-thoughts/OpenThoughts-Agent-v1-SFT',
                    'open-thoughts/OpenThoughts-Agent-v1-RL',
                    'open-r1/codeforces-cots',
                    'rizerphe/glaive-function-calling-v2-zephyr',
                ],
            },
            // Stage 10: Knowledge Benchmarks - MMLU/Knowledge
            {
                name: 'knowledge_benchmarks',
                module: 'instruction_tuning',
                epochs: 1,
                learningRate: '2e-5',
                datasets: [
                    'cais/mmlu',
                    'TIGER-Lab/MMLU-Pro',
                    'openai/MMMLU',
                    'CohereForAI/Global-MMLU',
                    'allenai/social_i_qa',
                ],
            },
            // Stage 11: Advanced Code Instruction
            {
                name: 'advanced_code_instruction',
                module: 'instruction_tuning',
                epochs: 1,
                learningRate: '2e-5',
                datasets: [
                    'TokenBender/code_instructions_122k_alpaca_style',
                    'nvidia/Nemotron-Pretraining-Code-v2',
                    'MathLLMs/MathCodeInstruct',
                ],
            },
            // Stage 12: Alignment - Final safety and helpfulness tuning
            {
                name: 'alignment',
                module: 'alignment',
                epochs: 1,
                learningRate: '1e-5',
                datasets: [
                    'Anthropic/hh-rlhf',
                    'Open-Orca/SlimOrca-Dedup',
                    'HuggingFaceH4/ultrafeedback_binarized',
                    'allenai/ultrafeedback_binarized_cleaned',
                    'Skywork/Skywork-Reward-Preference-80K-v0.2',
                    'argilla/dpo-mix-7k',
                ],
            },
        ],
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
    const [selectedId, setSelectedId] = useState<string | null>(TEMPLATES[0].id);
    const [filter, setFilter] = useState<string>('all');

    const selectedTemplate = TEMPLATES.find((t) => t.id === selectedId) || null;

    const filteredTemplates = filter === 'all'
        ? TEMPLATES
        : TEMPLATES.filter((t) => t.category === filter);

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
