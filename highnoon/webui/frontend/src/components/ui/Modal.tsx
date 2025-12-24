// HighNoon Dashboard - Modal Component
import { useEffect, useCallback, type ReactNode, type HTMLAttributes } from 'react';
import { X } from 'lucide-react';
import { Button } from './Button';
import './Modal.css';

export interface ModalProps extends HTMLAttributes<HTMLDivElement> {
    open: boolean;
    onClose: () => void;
    title: string;
    description?: string;
    size?: 'sm' | 'md' | 'lg' | 'xl';
    footer?: ReactNode;
    closeOnOverlay?: boolean;
    closeOnEscape?: boolean;
}

export function Modal({
    open,
    onClose,
    title,
    description,
    size = 'md',
    footer,
    closeOnOverlay = true,
    closeOnEscape = true,
    children,
    className = '',
    ...props
}: ModalProps) {
    const handleEscape = useCallback(
        (e: KeyboardEvent) => {
            if (e.key === 'Escape' && closeOnEscape) {
                onClose();
            }
        },
        [closeOnEscape, onClose]
    );

    useEffect(() => {
        if (open) {
            document.addEventListener('keydown', handleEscape);
            document.body.style.overflow = 'hidden';
        }

        return () => {
            document.removeEventListener('keydown', handleEscape);
            document.body.style.overflow = '';
        };
    }, [open, handleEscape]);

    if (!open) return null;

    return (
        <div className="modal-overlay" onClick={closeOnOverlay ? onClose : undefined}>
            <div
                className={`modal modal-${size} animate-scaleIn ${className}`}
                onClick={(e) => e.stopPropagation()}
                role="dialog"
                aria-modal="true"
                aria-labelledby="modal-title"
                {...props}
            >
                <div className="modal-header">
                    <div className="modal-header-text">
                        <h2 id="modal-title" className="modal-title">
                            {title}
                        </h2>
                        {description && <p className="modal-description">{description}</p>}
                    </div>
                    <button
                        className="modal-close"
                        onClick={onClose}
                        aria-label="Close modal"
                    >
                        <X size={20} />
                    </button>
                </div>

                <div className="modal-content">{children}</div>

                {footer && <div className="modal-footer">{footer}</div>}
            </div>
        </div>
    );
}

// Confirmation modal helper
export interface ConfirmModalProps {
    open: boolean;
    onClose: () => void;
    onConfirm: () => void;
    title: string;
    description: string;
    confirmText?: string;
    cancelText?: string;
    variant?: 'danger' | 'primary';
    loading?: boolean;
}

export function ConfirmModal({
    open,
    onClose,
    onConfirm,
    title,
    description,
    confirmText = 'Confirm',
    cancelText = 'Cancel',
    variant = 'primary',
    loading = false,
}: ConfirmModalProps) {
    return (
        <Modal
            open={open}
            onClose={onClose}
            title={title}
            size="sm"
            footer={
                <>
                    <Button variant="ghost" onClick={onClose} disabled={loading}>
                        {cancelText}
                    </Button>
                    <Button
                        variant={variant === 'danger' ? 'danger' : 'primary'}
                        onClick={onConfirm}
                        loading={loading}
                    >
                        {confirmText}
                    </Button>
                </>
            }
        >
            <p className="modal-confirm-text">{description}</p>
        </Modal>
    );
}
