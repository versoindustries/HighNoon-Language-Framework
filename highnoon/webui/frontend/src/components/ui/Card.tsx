// HighNoon Dashboard - Card Component
import type { HTMLAttributes, ReactNode } from 'react';
import './Card.css';

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
    variant?: 'default' | 'glass' | 'outlined';
    padding?: 'none' | 'sm' | 'md' | 'lg';
    hover?: boolean;
}

export function Card({
    variant = 'default',
    padding = 'md',
    hover = false,
    className = '',
    children,
    ...props
}: CardProps) {
    return (
        <div
            className={[
                'card',
                `card-${variant}`,
                `card-p-${padding}`,
                hover && 'card-hover',
                className,
            ]
                .filter(Boolean)
                .join(' ')}
            {...props}
        >
            {children}
        </div>
    );
}

export interface CardHeaderProps extends HTMLAttributes<HTMLDivElement> {
    title: string;
    subtitle?: string;
    action?: ReactNode;
}

export function CardHeader({
    title,
    subtitle,
    action,
    className = '',
    ...props
}: CardHeaderProps) {
    return (
        <div className={`card-header ${className}`} {...props}>
            <div className="card-header-text">
                <h3 className="card-title">{title}</h3>
                {subtitle && <p className="card-subtitle">{subtitle}</p>}
            </div>
            {action && <div className="card-header-action">{action}</div>}
        </div>
    );
}

export function CardContent({
    className = '',
    children,
    ...props
}: HTMLAttributes<HTMLDivElement>) {
    return (
        <div className={`card-content ${className}`} {...props}>
            {children}
        </div>
    );
}

export function CardFooter({
    className = '',
    children,
    ...props
}: HTMLAttributes<HTMLDivElement>) {
    return (
        <div className={`card-footer ${className}`} {...props}>
            {children}
        </div>
    );
}
