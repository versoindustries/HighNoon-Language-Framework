// TutorialMode - Interactive guided walkthrough for new users
// Helps users get familiar with the HighNoon webui quickly

import { useState, useEffect, useCallback } from 'react';
import {
    X, ChevronRight, ChevronLeft, Check, Rocket, Database,
    GraduationCap, Sliders, Zap, LayoutDashboard, HelpCircle,
    Sparkles, Play, BookOpen
} from 'lucide-react';
import { Button, Modal } from '../ui';
import './TutorialMode.css';

// =============================================================================
// TUTORIAL STEP DEFINITIONS
// =============================================================================

export interface TutorialStep {
    id: string;
    title: string;
    description: string;
    content: React.ReactNode;
    icon: React.ReactNode;
    highlight?: string; // CSS selector to highlight
    action?: {
        label: string;
        path?: string;
        onClick?: () => void;
    };
}

const TUTORIAL_STEPS: TutorialStep[] = [
    {
        id: 'welcome',
        title: 'Welcome to HighNoon',
        description: 'Your gateway to training custom language models',
        icon: <Rocket size={32} />,
        content: (
            <div className="tutorial-welcome">
                <p>
                    <strong>HighNoon Language Framework</strong> is your all-in-one solution for training
                    custom language models with the HSMN architecture.
                </p>
                <p>
                    This tutorial will walk you through the key features and help you get started
                    with training your first model in just a few steps.
                </p>
                <div className="tutorial-highlight-box">
                    <Sparkles size={20} />
                    <span>Lite Edition supports models up to 20B parameters with 5M token context</span>
                </div>
            </div>
        ),
    },
    {
        id: 'dashboard',
        title: 'Dashboard Overview',
        description: 'Your training command center',
        icon: <LayoutDashboard size={32} />,
        highlight: '[data-section="dashboard"]',
        content: (
            <div className="tutorial-content-section">
                <p>
                    The <strong>Dashboard</strong> is your home base. Here you can:
                </p>
                <ul className="tutorial-feature-list">
                    <li><Check size={14} /> Monitor active training jobs and HPO sweeps</li>
                    <li><Check size={14} /> View system status and Lite Edition limits</li>
                    <li><Check size={14} /> Access quick actions for common tasks</li>
                    <li><Check size={14} /> Launch training with the QuickStart wizard</li>
                </ul>
                <p className="tutorial-tip">
                    <strong>Pro Tip:</strong> Use the "Start Training" button for the fastest path to training.
                </p>
            </div>
        ),
        action: {
            label: 'Go to Dashboard',
            path: '/',
        },
    },
    {
        id: 'datasets',
        title: 'Managing Datasets',
        description: 'Browse and add training data',
        icon: <Database size={32} />,
        highlight: '[data-section="datasets"]',
        content: (
            <div className="tutorial-content-section">
                <p>
                    The <strong>Datasets</strong> page lets you manage your training data:
                </p>
                <ul className="tutorial-feature-list">
                    <li><Check size={14} /> Browse datasets from HuggingFace Hub</li>
                    <li><Check size={14} /> Register local or remote datasets</li>
                    <li><Check size={14} /> Preview dataset contents and statistics</li>
                    <li><Check size={14} /> Validate data quality before training</li>
                </ul>
                <div className="tutorial-highlight-box">
                    <Database size={20} />
                    <span>Datasets are automatically preprocessed for optimal training performance</span>
                </div>
            </div>
        ),
        action: {
            label: 'Explore Datasets',
            path: '/datasets',
        },
    },
    {
        id: 'curriculum',
        title: 'Building Curriculums',
        description: 'Design multi-stage training plans',
        icon: <GraduationCap size={32} />,
        highlight: '[data-section="curriculum"]',
        content: (
            <div className="tutorial-content-section">
                <p>
                    The <strong>Curriculum Builder</strong> is where you design your training strategy:
                </p>
                <ul className="tutorial-feature-list">
                    <li><Check size={14} /> Create multi-stage training curriculums</li>
                    <li><Check size={14} /> Choose from baseline presets (Chat, Code, Reasoning, etc.)</li>
                    <li><Check size={14} /> Mix datasets with custom weights</li>
                    <li><Check size={14} /> Configure learning rates and epochs per stage</li>
                </ul>
                <p className="tutorial-tip">
                    <strong>Pro Tip:</strong> Start with a baseline curriculum and customize from there.
                </p>
            </div>
        ),
        action: {
            label: 'Build Curriculum',
            path: '/curriculum',
        },
    },
    {
        id: 'hpo',
        title: 'HPO Orchestrator',
        description: 'Hyperparameter optimization made easy',
        icon: <Sliders size={32} />,
        highlight: '[data-section="hpo"]',
        content: (
            <div className="tutorial-content-section">
                <p>
                    The <strong>HPO Orchestrator</strong> automatically finds optimal hyperparameters:
                </p>
                <ul className="tutorial-feature-list">
                    <li><Check size={14} /> Configure search spaces for learning rate, batch size, etc.</li>
                    <li><Check size={14} /> Run multiple trials in parallel</li>
                    <li><Check size={14} /> Track trial progress and metrics in real-time</li>
                    <li><Check size={14} /> Automatically select the best configuration</li>
                </ul>
                <div className="tutorial-highlight-box">
                    <Sliders size={20} />
                    <span>HPO uses SophiaG optimizer by default for faster convergence</span>
                </div>
            </div>
        ),
        action: {
            label: 'Configure HPO',
            path: '/hpo',
        },
    },
    {
        id: 'training',
        title: 'Training Your Model',
        description: 'Launch and monitor training runs',
        icon: <Zap size={32} />,
        highlight: '[data-section="training"]',
        content: (
            <div className="tutorial-content-section">
                <p>
                    The <strong>Training</strong> page is where your model comes to life:
                </p>
                <ul className="tutorial-feature-list">
                    <li><Check size={14} /> Start training with your configured curriculum</li>
                    <li><Check size={14} /> Monitor loss and metrics in real-time</li>
                    <li><Check size={14} /> View training logs and diagnostics</li>
                    <li><Check size={14} /> Save checkpoints and export trained models</li>
                </ul>
                <p className="tutorial-tip">
                    <strong>Pro Tip:</strong> The training process respects Lite Edition limits automatically.
                </p>
            </div>
        ),
        action: {
            label: 'Start Training',
            path: '/training',
        },
    },
    {
        id: 'complete',
        title: 'You\'re Ready!',
        description: 'Start training your first model',
        icon: <Sparkles size={32} />,
        content: (
            <div className="tutorial-complete">
                <div className="tutorial-complete-icon">
                    <Check size={48} />
                </div>
                <h3>Tutorial Complete!</h3>
                <p>
                    You now know the basics of the HighNoon Language Framework.
                    Here's the recommended workflow to get started:
                </p>
                <ol className="tutorial-workflow">
                    <li><span className="step-number">1</span> Add or select datasets from the Datasets page</li>
                    <li><span className="step-number">2</span> Choose a baseline curriculum or build your own</li>
                    <li><span className="step-number">3</span> Run HPO to find optimal hyperparameters</li>
                    <li><span className="step-number">4</span> Launch training with the best configuration</li>
                </ol>
                <div className="tutorial-cta">
                    <Play size={20} />
                    <span>Ready to train your first model?</span>
                </div>
            </div>
        ),
        action: {
            label: 'Get Started',
            path: '/',
        },
    },
];

// =============================================================================
// TUTORIAL PROGRESS INDICATOR
// =============================================================================

interface ProgressIndicatorProps {
    currentStep: number;
    totalSteps: number;
    steps: TutorialStep[];
}

function ProgressIndicator({ currentStep, totalSteps, steps }: ProgressIndicatorProps) {
    return (
        <div className="tutorial-progress">
            <div className="tutorial-progress-bar">
                <div
                    className="tutorial-progress-fill"
                    style={{ width: `${((currentStep + 1) / totalSteps) * 100}%` }}
                />
            </div>
            <div className="tutorial-progress-steps">
                {steps.map((step, index) => (
                    <div
                        key={step.id}
                        className={`tutorial-progress-step ${
                            index < currentStep ? 'completed' : ''
                        } ${index === currentStep ? 'active' : ''}`}
                        title={step.title}
                    >
                        {index < currentStep ? (
                            <Check size={12} />
                        ) : (
                            <span>{index + 1}</span>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}

// =============================================================================
// MAIN TUTORIAL COMPONENT
// =============================================================================

interface TutorialModeProps {
    open: boolean;
    onClose: () => void;
    onNavigate?: (path: string) => void;
}

export function TutorialMode({ open, onClose, onNavigate }: TutorialModeProps) {
    const [currentStep, setCurrentStep] = useState(0);
    const [hasSeenTutorial, setHasSeenTutorial] = useState(false);

    const step = TUTORIAL_STEPS[currentStep];
    const isFirstStep = currentStep === 0;
    const isLastStep = currentStep === TUTORIAL_STEPS.length - 1;

    // Check if user has seen tutorial before
    useEffect(() => {
        const seen = localStorage.getItem('highnoon-tutorial-seen');
        setHasSeenTutorial(seen === 'true');
    }, []);

    const handleNext = () => {
        if (currentStep < TUTORIAL_STEPS.length - 1) {
            setCurrentStep(currentStep + 1);
        }
    };

    const handlePrev = () => {
        if (currentStep > 0) {
            setCurrentStep(currentStep - 1);
        }
    };

    const handleComplete = () => {
        localStorage.setItem('highnoon-tutorial-seen', 'true');
        setCurrentStep(0);
        onClose();
    };

    const handleAction = () => {
        if (step.action?.path && onNavigate) {
            onNavigate(step.action.path);
        }
        if (step.action?.onClick) {
            step.action.onClick();
        }
        if (isLastStep) {
            handleComplete();
        } else {
            handleNext();
        }
    };

    const handleSkip = () => {
        localStorage.setItem('highnoon-tutorial-seen', 'true');
        setCurrentStep(0);
        onClose();
    };

    return (
        <Modal
            open={open}
            onClose={handleSkip}
            title=""
            size="lg"
        >
            <div className="tutorial-modal">
                <button className="tutorial-close" onClick={handleSkip} aria-label="Close tutorial">
                    <X size={20} />
                </button>

                <ProgressIndicator
                    currentStep={currentStep}
                    totalSteps={TUTORIAL_STEPS.length}
                    steps={TUTORIAL_STEPS}
                />

                <div className="tutorial-header">
                    <div className="tutorial-icon">{step.icon}</div>
                    <div className="tutorial-titles">
                        <h2>{step.title}</h2>
                        <p>{step.description}</p>
                    </div>
                </div>

                <div className="tutorial-body">
                    {step.content}
                </div>

                <div className="tutorial-footer">
                    <div className="tutorial-footer-left">
                        {!isFirstStep && (
                            <Button variant="ghost" onClick={handlePrev}>
                                <ChevronLeft size={16} /> Previous
                            </Button>
                        )}
                    </div>

                    <div className="tutorial-footer-center">
                        <span className="tutorial-step-counter">
                            {currentStep + 1} of {TUTORIAL_STEPS.length}
                        </span>
                    </div>

                    <div className="tutorial-footer-right">
                        {!isLastStep && (
                            <Button variant="ghost" onClick={handleSkip}>
                                Skip Tutorial
                            </Button>
                        )}
                        {step.action ? (
                            <Button variant="primary" onClick={handleAction}>
                                {step.action.label} <ChevronRight size={16} />
                            </Button>
                        ) : (
                            <Button variant="primary" onClick={handleNext}>
                                Continue <ChevronRight size={16} />
                            </Button>
                        )}
                    </div>
                </div>
            </div>
        </Modal>
    );
}

// =============================================================================
// TUTORIAL TRIGGER BUTTON (for Sidebar)
// =============================================================================

interface TutorialTriggerProps {
    onClick: () => void;
}

export function TutorialTrigger({ onClick }: TutorialTriggerProps) {
    return (
        <button className="tutorial-trigger" onClick={onClick} title="Start Tutorial">
            <HelpCircle size={18} />
            <span>Tutorial</span>
        </button>
    );
}

export default TutorialMode;
