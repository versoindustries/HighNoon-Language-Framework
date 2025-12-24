// HighNoon Dashboard - Progress Bar Component
import type { HTMLAttributes } from 'react';
import './ProgressBar.css';

export interface ProgressBarProps extends HTMLAttributes<HTMLDivElement> {
    value: number; // 0-100
    max?: number;
    variant?: 'default' | 'success' | 'warning' | 'danger' | 'gradient';
    size?: 'sm' | 'md' | 'lg';
    showLabel?: boolean;
    label?: string;
    animated?: boolean;
}

export function ProgressBar({
    value,
    max = 100,
    variant = 'default',
    size = 'md',
    showLabel = false,
    label,
    animated = true,
    className = '',
    ...props
}: ProgressBarProps) {
    const percentage = Math.min(100, Math.max(0, (value / max) * 100));

    return (
        <div
            className={`progress-wrapper ${className}`}
            role="progressbar"
            aria-valuenow={value}
            aria-valuemin={0}
            aria-valuemax={max}
            {...props}
        >
            {(showLabel || label) && (
                <div className="progress-label-row">
                    <span className="progress-label">{label || 'Progress'}</span>
                    {showLabel && (
                        <span className="progress-value">{percentage.toFixed(1)}%</span>
                    )}
                </div>
            )}
            <div className={`progress-track progress-${size}`}>
                <div
                    className={`progress-fill progress-fill-${variant} ${animated ? 'progress-animated' : ''}`}
                    style={{ width: `${percentage}%` }}
                />
            </div>
        </div>
    );
}

// Stat display with optional progress
export interface StatCardProps {
    label: string;
    value: string | number;
    description?: string;
    icon?: React.ReactNode;
    trend?: { value: number; positive: boolean };
}

export function StatCard({ label, value, description, icon, trend }: StatCardProps) {
    return (
        <div className="stat-card">
            {icon && <div className="stat-icon">{icon}</div>}
            <div className="stat-content">
                <span className="stat-label">{label}</span>
                <span className="stat-value">{value}</span>
                {description && <span className="stat-desc">{description}</span>}
                {trend && (
                    <span
                        className={`stat-trend ${trend.positive ? 'stat-trend-up' : 'stat-trend-down'}`}
                    >
                        {trend.positive ? '↑' : '↓'} {Math.abs(trend.value)}%
                    </span>
                )}
            </div>
        </div>
    );
}
