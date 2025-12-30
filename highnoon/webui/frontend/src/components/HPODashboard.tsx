// HPODashboard.tsx - Unified live dashboard for HPO optimization
// Enterprise-grade monitoring interface with F1/Jet cockpit HUD

import { useState } from 'react';
import {
    Pause,
    Square,
    Settings,
    ChevronDown,
    ChevronUp,
} from 'lucide-react';
import { Button } from './ui';
import { CockpitHUD } from './CockpitHUD';
import { TrainingConsole } from './TrainingConsole';
import type { HPOSweepInfo, HPOTrialInfo } from '../api/types';
import './HPODashboard.css';

interface HPODashboardProps {
    sweepStatus: HPOSweepInfo | null;
    trials: HPOTrialInfo[];
    isRunning: boolean;
    devMode?: boolean;
    onPause?: () => void;
    onStop?: () => void;
    onShowSettings?: () => void;
    config?: {
        curriculumName?: string;
        paramBudget?: number;
        optimizer?: string;
        optimizationMode?: string;
        maxTrials?: number;
    };
}

export function HPODashboard({
    sweepStatus,
    trials,
    isRunning,
    devMode = false,
    onPause,
    onStop,
    onShowSettings,
    config,
}: HPODashboardProps) {
    const [consoleExpanded, setConsoleExpanded] = useState(false);

    return (
        <div className="hpo-dashboard hpo-dashboard--cockpit">
            {/* Action buttons - floating top-right */}
            <div className="hpo-dashboard__actions-floating">
                {isRunning && (
                    <>
                        <Button
                            variant="secondary"
                            size="sm"
                            leftIcon={<Pause size={16} />}
                            onClick={onPause}
                        >
                            Pause
                        </Button>
                        <Button
                            variant="danger"
                            size="sm"
                            leftIcon={<Square size={16} />}
                            onClick={onStop}
                        >
                            Stop
                        </Button>
                    </>
                )}
                <Button
                    variant="ghost"
                    size="sm"
                    leftIcon={<Settings size={16} />}
                    onClick={onShowSettings}
                >
                    Settings
                </Button>
            </div>

            {/* Main Cockpit HUD - replaces all scattered sections */}
            <CockpitHUD
                sweepStatus={sweepStatus}
                trials={trials}
                isRunning={isRunning}
                devMode={devMode}
                onPause={onPause}
                onStop={onStop}
                config={config}
            />

            {/* Training Console - kept as collapsible at bottom */}
            <div className={`hpo-dashboard__console ${consoleExpanded ? 'hpo-dashboard__console--expanded' : ''}`}>
                <button
                    className="hpo-dashboard__console-toggle"
                    onClick={() => setConsoleExpanded(!consoleExpanded)}
                >
                    <span>Training Console</span>
                    {consoleExpanded ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
                </button>
                {consoleExpanded && (
                    <TrainingConsole
                        sweepId={sweepStatus?.sweep_id || null}
                        isRunning={isRunning}
                        devMode={devMode}
                    />
                )}
            </div>
        </div>
    );
}

export default HPODashboard;
