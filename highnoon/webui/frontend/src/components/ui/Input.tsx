// HighNoon Dashboard - Input Component
import { forwardRef, type InputHTMLAttributes } from 'react';
import './Input.css';

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
    label?: string;
    error?: string;
    hint?: string;
    leftElement?: React.ReactNode;
    rightElement?: React.ReactNode;
    fullWidth?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
    (
        {
            label,
            error,
            hint,
            leftElement,
            rightElement,
            fullWidth = false,
            className = '',
            id,
            ...props
        },
        ref
    ) => {
        const inputId = id || label?.toLowerCase().replace(/\s+/g, '-');

        return (
            <div className={`input-wrapper ${fullWidth ? 'input-full' : ''} ${className}`}>
                {label && (
                    <label htmlFor={inputId} className="input-label">
                        {label}
                    </label>
                )}
                <div className={`input-container ${error ? 'input-error' : ''}`}>
                    {leftElement && <span className="input-left">{leftElement}</span>}
                    <input
                        ref={ref}
                        id={inputId}
                        className="input"
                        {...props}
                    />
                    {rightElement && <span className="input-right">{rightElement}</span>}
                </div>
                {(error || hint) && (
                    <span className={`input-hint ${error ? 'input-hint-error' : ''}`}>
                        {error || hint}
                    </span>
                )}
            </div>
        );
    }
);

Input.displayName = 'Input';

// Range input for HPO parameter tuning
export interface RangeInputProps {
    label: string;
    minLabel: string;
    maxLabel: string;
    minValue: string;
    maxValue: string;
    onMinChange: (value: string) => void;
    onMaxChange: (value: string) => void;
    logScale?: boolean;
    onLogScaleChange?: (value: boolean) => void;
    error?: string;
}

export function RangeInput({
    label,
    minLabel,
    maxLabel,
    minValue,
    maxValue,
    onMinChange,
    onMaxChange,
    logScale,
    onLogScaleChange,
    error,
}: RangeInputProps) {
    return (
        <div className="range-input-wrapper">
            <label className="input-label">{label}</label>
            <div className="range-input-row">
                <div className="range-input-field">
                    <span className="range-label">{minLabel}</span>
                    <input
                        type="text"
                        className={`input ${error ? 'input-has-error' : ''}`}
                        value={minValue}
                        onChange={(e) => onMinChange(e.target.value)}
                    />
                </div>
                <span className="range-separator">to</span>
                <div className="range-input-field">
                    <span className="range-label">{maxLabel}</span>
                    <input
                        type="text"
                        className={`input ${error ? 'input-has-error' : ''}`}
                        value={maxValue}
                        onChange={(e) => onMaxChange(e.target.value)}
                    />
                </div>
                {onLogScaleChange !== undefined && (
                    <label className="range-log-toggle">
                        <input
                            type="checkbox"
                            checked={logScale}
                            onChange={(e) => onLogScaleChange(e.target.checked)}
                        />
                        <span>Log scale</span>
                    </label>
                )}
            </div>
            {error && <span className="input-hint input-hint-error">{error}</span>}
        </div>
    );
}
