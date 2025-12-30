// HUDStatusRing.tsx - Animated status indicator ring for optimization phase
// Shows current phase with rotating animation and phase labels

import { useEffect, useState } from 'react';
import { Zap, Compass, Target, AlertTriangle, Battery, Loader2 } from 'lucide-react';
import './HUDGauges.css';

type OptimizationPhase = 'idle' | 'warmup' | 'exploration' | 'exploitation' | 'emergency';

interface HUDStatusRingProps {
    phase: OptimizationPhase;
    trialNumber?: number;
    elapsedTime?: string;
    isRunning?: boolean;
    emergencyMode?: boolean;
    tunneling?: number;  // 0-1 tunneling probability
    temperature?: number;
}

const PHASE_CONFIG: Record<OptimizationPhase, {
    icon: React.ReactNode;
    color: string;
    label: string;
    description: string;
}> = {
    idle: {
        icon: <Loader2 size={20} />,
        color: 'var(--text-muted)',
        label: 'IDLE',
        description: 'Ready to start',
    },
    warmup: {
        icon: <Battery size={20} />,
        color: 'var(--color-warning)',
        label: 'WARMUP',
        description: 'Initializing search',
    },
    exploration: {
        icon: <Compass size={20} />,
        color: 'var(--color-info)',
        label: 'EXPLORE',
        description: 'Searching space',
    },
    exploitation: {
        icon: <Target size={20} />,
        color: 'var(--color-success)',
        label: 'EXPLOIT',
        description: 'Refining best',
    },
    emergency: {
        icon: <AlertTriangle size={20} />,
        color: 'var(--color-danger)',
        label: 'EMERGENCY',
        description: 'Recovery mode',
    },
};

export function HUDStatusRing({
    phase,
    trialNumber,
    elapsedTime,
    isRunning = false,
    emergencyMode = false,
    tunneling = 0,
    temperature = 1.0,
}: HUDStatusRingProps) {
    const [rotation, setRotation] = useState(0);

    // Animate rotation when running
    useEffect(() => {
        if (!isRunning) {
            return;
        }

        const speed = emergencyMode ? 2 : phase === 'exploration' ? 0.5 : 0.2;
        const interval = setInterval(() => {
            setRotation(prev => (prev + speed) % 360);
        }, 50);

        return () => clearInterval(interval);
    }, [isRunning, phase, emergencyMode]);

    const config = emergencyMode ? PHASE_CONFIG.emergency : PHASE_CONFIG[phase];
    const displayPhase = emergencyMode ? 'emergency' : phase;

    // Calculate ring segment fills based on metrics
    const tunnelingPct = tunneling * 100;
    const tempNormalized = Math.min(temperature / 2, 1) * 100;

    return (
        <div className={`hud-status-ring hud-status-ring--${displayPhase} ${isRunning ? 'hud-status-ring--running' : ''}`}>
            {/* Outer rotating ring */}
            <div
                className="hud-status-ring__outer"
                style={{
                    transform: `rotate(${rotation}deg)`,
                    borderColor: config.color,
                }}
            >
                <div className="hud-status-ring__dash" style={{ background: config.color }} />
                <div className="hud-status-ring__dash" style={{ background: config.color }} />
                <div className="hud-status-ring__dash" style={{ background: config.color }} />
                <div className="hud-status-ring__dash" style={{ background: config.color }} />
            </div>

            {/* Middle static ring */}
            <div className="hud-status-ring__middle" style={{ borderColor: `${config.color}40` }}>
                {/* Tunneling indicator arc */}
                <svg className="hud-status-ring__tunneling-svg" viewBox="0 0 100 100">
                    <circle
                        className="hud-status-ring__tunneling-track"
                        cx="50"
                        cy="50"
                        r="46"
                    />
                    <circle
                        className="hud-status-ring__tunneling-fill"
                        cx="50"
                        cy="50"
                        r="46"
                        style={{
                            stroke: 'var(--color-secondary)',
                            strokeDasharray: `${tunnelingPct * 2.89} 289`,
                        }}
                    />
                </svg>
            </div>

            {/* Inner content */}
            <div className="hud-status-ring__content">
                <div className="hud-status-ring__icon" style={{ color: config.color }}>
                    {config.icon}
                </div>
                <span className="hud-status-ring__phase" style={{ color: config.color }}>
                    {config.label}
                </span>

                {trialNumber !== undefined && (
                    <div className="hud-status-ring__trial">
                        <Zap size={12} />
                        <span>Trial {trialNumber}</span>
                    </div>
                )}

                {elapsedTime && (
                    <span className="hud-status-ring__time">{elapsedTime}</span>
                )}
            </div>

            {/* Status indicators */}
            <div className="hud-status-ring__indicators">
                <div className="hud-status-ring__indicator">
                    <span className="hud-status-ring__indicator-label">TUNNEL</span>
                    <span className="hud-status-ring__indicator-value">
                        {(tunneling * 100).toFixed(0)}%
                    </span>
                </div>
                <div className="hud-status-ring__indicator">
                    <span className="hud-status-ring__indicator-label">TEMP</span>
                    <span className="hud-status-ring__indicator-value">
                        {temperature.toFixed(2)}
                    </span>
                </div>
            </div>
        </div>
    );
}

export default HUDStatusRing;
