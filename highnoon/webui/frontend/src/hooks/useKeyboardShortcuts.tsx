// HighNoon Dashboard - Keyboard Shortcuts Hook
// Provides vim-style navigation shortcuts for power users

import { useEffect, useCallback, useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface ShortcutHelp {
    key: string;
    description: string;
    category: string;
}

const SHORTCUTS: ShortcutHelp[] = [
    { key: 'g d', description: 'Go to Dashboard', category: 'Navigation' },
    { key: 'g h', description: 'Go to HPO Orchestrator', category: 'Navigation' },
    { key: 'g t', description: 'Go to Training', category: 'Navigation' },
    { key: 'g c', description: 'Go to Curriculum', category: 'Navigation' },
    { key: 'g s', description: 'Go to Settings', category: 'Navigation' },
    { key: 'g b', description: 'Go to Datasets', category: 'Navigation' },
    { key: '?', description: 'Show keyboard shortcuts', category: 'Help' },
    { key: 'Escape', description: 'Close modal / Cancel', category: 'Actions' },
];

export function useKeyboardShortcuts() {
    const navigate = useNavigate();
    const [showHelp, setShowHelp] = useState(false);
    const [lastKey, setLastKey] = useState<string | null>(null);
    const [lastKeyTime, setLastKeyTime] = useState<number>(0);

    const handleKeyDown = useCallback((event: KeyboardEvent) => {
        // Ignore if user is typing in an input field
        const target = event.target as HTMLElement;
        if (
            target.tagName === 'INPUT' ||
            target.tagName === 'TEXTAREA' ||
            target.tagName === 'SELECT' ||
            target.isContentEditable
        ) {
            return;
        }

        const currentTime = Date.now();
        const key = event.key.toLowerCase();

        // Reset last key if more than 500ms has passed
        const effectiveLastKey = currentTime - lastKeyTime < 500 ? lastKey : null;

        // Handle 'g' prefix navigation
        if (effectiveLastKey === 'g') {
            event.preventDefault();
            switch (key) {
                case 'd':
                    navigate('/');
                    break;
                case 'h':
                    navigate('/hpo');
                    break;
                case 't':
                    navigate('/training');
                    break;
                case 'c':
                    navigate('/curriculum');
                    break;
                case 's':
                    navigate('/settings');
                    break;
                case 'b':
                    navigate('/datasets');
                    break;
            }
            setLastKey(null);
            return;
        }

        // Handle single-key shortcuts
        switch (key) {
            case '?':
                event.preventDefault();
                setShowHelp(prev => !prev);
                break;
            case 'escape':
                setShowHelp(false);
                // Also close any open modals (dispatch custom event)
                window.dispatchEvent(new CustomEvent('keyboard:escape'));
                break;
            case 'g':
                // Start of navigation sequence
                setLastKey('g');
                setLastKeyTime(currentTime);
                return;
        }

        setLastKey(key);
        setLastKeyTime(currentTime);
    }, [navigate, lastKey, lastKeyTime]);

    useEffect(() => {
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [handleKeyDown]);

    return {
        showHelp,
        setShowHelp,
        shortcuts: SHORTCUTS,
    };
}

// Keyboard shortcut indicator component
export function ShortcutIndicator({ shortcut }: { shortcut: string }) {
    const keys = shortcut.split(' ');

    return (
        <span className="shortcut-indicator">
            {keys.map((key, index) => (
                <span key={index}>
                    <kbd className="shortcut-key">{key}</kbd>
                    {index < keys.length - 1 && <span className="shortcut-separator">+</span>}
                </span>
            ))}
        </span>
    );
}

// Keyboard help modal component
export function KeyboardHelpModal({
    isOpen,
    onClose
}: {
    isOpen: boolean;
    onClose: () => void;
}) {
    useEffect(() => {
        const handleEscape = () => onClose();
        window.addEventListener('keyboard:escape', handleEscape);
        return () => window.removeEventListener('keyboard:escape', handleEscape);
    }, [onClose]);

    if (!isOpen) return null;

    const categories = [...new Set(SHORTCUTS.map(s => s.category))];

    return (
        <div className="keyboard-help-overlay" onClick={onClose}>
            <div
                className="keyboard-help-modal"
                onClick={e => e.stopPropagation()}
                role="dialog"
                aria-label="Keyboard shortcuts"
            >
                <div className="keyboard-help-header">
                    <h2>Keyboard Shortcuts</h2>
                    <button
                        className="keyboard-help-close"
                        onClick={onClose}
                        aria-label="Close"
                    >
                        ×
                    </button>
                </div>

                <div className="keyboard-help-content">
                    {categories.map(category => (
                        <div key={category} className="keyboard-help-category">
                            <h3>{category}</h3>
                            <div className="keyboard-help-list">
                                {SHORTCUTS.filter(s => s.category === category).map(shortcut => (
                                    <div key={shortcut.key} className="keyboard-help-item">
                                        <ShortcutIndicator shortcut={shortcut.key} />
                                        <span className="keyboard-help-description">
                                            {shortcut.description}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>

                <div className="keyboard-help-footer">
                    <span>Press <kbd>?</kbd> to toggle this help</span>
                </div>
            </div>
        </div>
    );
}
