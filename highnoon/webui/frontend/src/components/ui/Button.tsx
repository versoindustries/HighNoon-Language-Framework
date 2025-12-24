// HighNoon Dashboard - Button Component
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';
import { Loader2 } from 'lucide-react';
import './Button.css';

export type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost' | 'outline';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: ButtonVariant;
    size?: ButtonSize;
    loading?: boolean;
    leftIcon?: ReactNode;
    rightIcon?: ReactNode;
    fullWidth?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
    (
        {
            variant = 'primary',
            size = 'md',
            loading = false,
            leftIcon,
            rightIcon,
            fullWidth = false,
            disabled,
            children,
            className = '',
            ...props
        },
        ref
    ) => {
        const isDisabled = disabled || loading;

        return (
            <button
                ref={ref}
                className={[
                    'btn',
                    `btn-${variant}`,
                    `btn-${size}`,
                    fullWidth && 'btn-full',
                    loading && 'btn-loading',
                    className,
                ]
                    .filter(Boolean)
                    .join(' ')}
                disabled={isDisabled}
                {...props}
            >
                {loading ? (
                    <Loader2 className="btn-spinner" size={size === 'sm' ? 14 : 16} />
                ) : leftIcon ? (
                    <span className="btn-icon-left">{leftIcon}</span>
                ) : null}
                <span className="btn-text">{children}</span>
                {rightIcon && !loading && <span className="btn-icon-right">{rightIcon}</span>}
            </button>
        );
    }
);

Button.displayName = 'Button';
