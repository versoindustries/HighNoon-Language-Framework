// HighNoon Dashboard - Select Component
import { forwardRef, type SelectHTMLAttributes } from 'react';
import { ChevronDown } from 'lucide-react';
import './Select.css';

export interface SelectOption {
    value: string;
    label: string;
    description?: string;
    disabled?: boolean;
}

export interface SelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'children'> {
    label?: string;
    error?: string;
    hint?: string;
    options: SelectOption[];
    placeholder?: string;
    fullWidth?: boolean;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
    (
        {
            label,
            error,
            hint,
            options,
            placeholder,
            fullWidth = false,
            className = '',
            id,
            ...props
        },
        ref
    ) => {
        const selectId = id || label?.toLowerCase().replace(/\s+/g, '-');

        return (
            <div className={`select-wrapper ${fullWidth ? 'select-full' : ''} ${className}`}>
                {label && (
                    <label htmlFor={selectId} className="select-label">
                        {label}
                    </label>
                )}
                <div className={`select-container ${error ? 'select-error' : ''}`}>
                    <select ref={ref} id={selectId} className="select" {...props}>
                        {placeholder && (
                            <option value="" disabled>
                                {placeholder}
                            </option>
                        )}
                        {options.map((option) => (
                            <option
                                key={option.value}
                                value={option.value}
                                disabled={option.disabled}
                            >
                                {option.label}
                            </option>
                        ))}
                    </select>
                    <ChevronDown className="select-icon" size={16} />
                </div>
                {(error || hint) && (
                    <span className={`select-hint ${error ? 'select-hint-error' : ''}`}>
                        {error || hint}
                    </span>
                )}
            </div>
        );
    }
);

Select.displayName = 'Select';

// Multi-select checkbox group (for HPO batch sizes, optimizers)
export interface CheckboxGroupProps {
    label: string;
    options: SelectOption[];
    selected: string[];
    onChange: (selected: string[]) => void;
    columns?: number;
    error?: string;
}

export function CheckboxGroup({
    label,
    options,
    selected,
    onChange,
    columns = 3,
    error,
}: CheckboxGroupProps) {
    const handleChange = (value: string, checked: boolean) => {
        if (checked) {
            onChange([...selected, value]);
        } else {
            onChange(selected.filter((v) => v !== value));
        }
    };

    return (
        <div className="checkbox-group-wrapper">
            <label className="select-label">{label}</label>
            <div
                className="checkbox-grid"
                style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}
            >
                {options.map((option) => (
                    <label
                        key={option.value}
                        className={`checkbox-option ${option.disabled ? 'checkbox-disabled' : ''}`}
                    >
                        <input
                            type="checkbox"
                            checked={selected.includes(option.value)}
                            onChange={(e) => handleChange(option.value, e.target.checked)}
                            disabled={option.disabled}
                        />
                        <span className="checkbox-label">{option.label}</span>
                        {option.description && (
                            <span className="checkbox-desc">{option.description}</span>
                        )}
                    </label>
                ))}
            </div>
            {error && <span className="select-hint select-hint-error">{error}</span>}
        </div>
    );
}
