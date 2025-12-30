// HUDAlertPanel.tsx - Consolidated error/warning panel for critical alerts
// Displays barren plateau detection, memory pressure, gradient issues

import { useState, useEffect } from 'react';
import { AlertTriangle, AlertCircle, Info, X, ChevronDown, ChevronUp } from 'lucide-react';
import './HUDGauges.css';

export type AlertSeverity = 'critical' | 'warning' | 'info';

export interface HUDAlert {
    id: string;
    severity: AlertSeverity;
    title: string;
    message: string;
    timestamp?: Date;
    dismissible?: boolean;
    autoDismiss?: number;  // Auto-dismiss after N seconds
}

interface HUDAlertPanelProps {
    alerts: HUDAlert[];
    onDismiss?: (id: string) => void;
    maxVisible?: number;
    compact?: boolean;
}

const SEVERITY_CONFIG = {
    critical: {
        icon: <AlertTriangle size={16} />,
        bgColor: 'var(--color-danger-muted)',
        borderColor: 'var(--color-danger)',
        textColor: 'var(--color-danger)',
    },
    warning: {
        icon: <AlertCircle size={16} />,
        bgColor: 'var(--color-warning-muted)',
        borderColor: 'var(--color-warning)',
        textColor: 'var(--color-warning)',
    },
    info: {
        icon: <Info size={16} />,
        bgColor: 'var(--color-info-muted)',
        borderColor: 'var(--color-info)',
        textColor: 'var(--color-info)',
    },
};

export function HUDAlertPanel({
    alerts,
    onDismiss,
    maxVisible = 3,
    compact = false,
}: HUDAlertPanelProps) {
    const [expanded, setExpanded] = useState(false);
    const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());

    // Filter out dismissed alerts and sort by severity
    const activeAlerts = alerts
        .filter(a => !dismissedIds.has(a.id))
        .sort((a, b) => {
            const severityOrder = { critical: 0, warning: 1, info: 2 };
            return severityOrder[a.severity] - severityOrder[b.severity];
        });

    const criticalCount = activeAlerts.filter(a => a.severity === 'critical').length;
    const warningCount = activeAlerts.filter(a => a.severity === 'warning').length;
    const visibleAlerts = expanded ? activeAlerts : activeAlerts.slice(0, maxVisible);
    const hiddenCount = activeAlerts.length - visibleAlerts.length;

    // Handle dismiss
    const handleDismiss = (id: string) => {
        setDismissedIds(prev => new Set([...prev, id]));
        onDismiss?.(id);
    };

    // Auto-dismiss handling
    useEffect(() => {
        const timers: ReturnType<typeof setTimeout>[] = [];

        activeAlerts.forEach(alert => {
            if (alert.autoDismiss && alert.autoDismiss > 0) {
                const timer = setTimeout(() => {
                    handleDismiss(alert.id);
                }, alert.autoDismiss * 1000);
                timers.push(timer);
            }
        });

        return () => {
            timers.forEach(t => clearTimeout(t));
        };
    }, [activeAlerts]);

    // No alerts - show all-clear indicator
    if (activeAlerts.length === 0) {
        return (
            <div className="hud-alert-panel hud-alert-panel--clear">
                <div className="hud-alert-panel__status">
                    <span className="hud-alert-panel__status-dot" />
                    <span className="hud-alert-panel__status-text">All Systems Nominal</span>
                </div>
            </div>
        );
    }

    return (
        <div className={`hud-alert-panel ${compact ? 'hud-alert-panel--compact' : ''}`}>
            {/* Header with summary */}
            <div className="hud-alert-panel__header">
                <div className="hud-alert-panel__summary">
                    <AlertTriangle size={14} className="hud-alert-panel__header-icon" />
                    <span className="hud-alert-panel__title">
                        {activeAlerts.length} Alert{activeAlerts.length !== 1 ? 's' : ''}
                    </span>
                    {criticalCount > 0 && (
                        <span className="hud-alert-panel__badge hud-alert-panel__badge--critical">
                            {criticalCount} Critical
                        </span>
                    )}
                    {warningCount > 0 && (
                        <span className="hud-alert-panel__badge hud-alert-panel__badge--warning">
                            {warningCount} Warning
                        </span>
                    )}
                </div>
                {activeAlerts.length > maxVisible && (
                    <button
                        className="hud-alert-panel__expand"
                        onClick={() => setExpanded(!expanded)}
                    >
                        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                        {expanded ? 'Show less' : `+${hiddenCount} more`}
                    </button>
                )}
            </div>

            {/* Alert list */}
            <div className="hud-alert-panel__list">
                {visibleAlerts.map((alert, index) => {
                    const config = SEVERITY_CONFIG[alert.severity];
                    return (
                        <div
                            key={alert.id}
                            className={`hud-alert-panel__item hud-alert-panel__item--${alert.severity}`}
                            style={{
                                '--alert-bg': config.bgColor,
                                '--alert-border': config.borderColor,
                                '--alert-color': config.textColor,
                                animationDelay: `${index * 50}ms`,
                            } as React.CSSProperties}
                        >
                            <div className="hud-alert-panel__item-icon" style={{ color: config.textColor }}>
                                {config.icon}
                            </div>
                            <div className="hud-alert-panel__item-content">
                                <span className="hud-alert-panel__item-title">{alert.title}</span>
                                {!compact && (
                                    <span className="hud-alert-panel__item-message">{alert.message}</span>
                                )}
                            </div>
                            {alert.dismissible !== false && (
                                <button
                                    className="hud-alert-panel__item-dismiss"
                                    onClick={() => handleDismiss(alert.id)}
                                >
                                    <X size={12} />
                                </button>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

// Helper function to create common alerts
export function createBarrenPlateauAlert(vqcVariance: number): HUDAlert {
    return {
        id: 'barren-plateau',
        severity: 'critical',
        title: 'Barren Plateau Detected',
        message: `VQC gradient variance dropped to ${vqcVariance.toExponential(2)}. QALRC mitigation active.`,
        dismissible: false,
    };
}

export function createMemoryAlert(usagePct: number): HUDAlert {
    const severity = usagePct >= 95 ? 'critical' : 'warning';
    return {
        id: 'memory-pressure',
        severity,
        title: 'Memory Pressure',
        message: `System memory at ${usagePct.toFixed(1)}% utilization.`,
        dismissible: true,
    };
}

export function createGradientAlert(type: 'explosion' | 'vanishing', norm: number): HUDAlert {
    return {
        id: `gradient-${type}`,
        severity: 'warning',
        title: type === 'explosion' ? 'Gradient Explosion' : 'Vanishing Gradients',
        message: `Gradient norm: ${norm.toExponential(2)}. Clipping applied.`,
        dismissible: true,
        autoDismiss: 10,
    };
}

export default HUDAlertPanel;
