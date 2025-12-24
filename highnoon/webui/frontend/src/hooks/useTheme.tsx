// HighNoon Dashboard - Theme Hook
import { useState, useEffect, useCallback, createContext, useContext, type ReactNode } from 'react';

type Theme = 'light' | 'dark';

interface ThemeContextValue {
    theme: Theme;
    setTheme: (theme: Theme) => void;
    toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function getInitialTheme(): Theme {
    // Check localStorage first
    const stored = localStorage.getItem('highnoon-theme');
    if (stored === 'light' || stored === 'dark') {
        return stored;
    }

    // Fall back to system preference
    if (window.matchMedia?.('(prefers-color-scheme: light)').matches) {
        return 'light';
    }

    // Default to dark
    return 'dark';
}

export function ThemeProvider({ children }: { children: ReactNode }) {
    const [theme, setThemeState] = useState<Theme>(getInitialTheme);

    const setTheme = useCallback((newTheme: Theme) => {
        setThemeState(newTheme);
        localStorage.setItem('highnoon-theme', newTheme);
        document.documentElement.setAttribute('data-theme', newTheme);
    }, []);

    const toggleTheme = useCallback(() => {
        setTheme(theme === 'dark' ? 'light' : 'dark');
    }, [theme, setTheme]);

    // Apply theme on mount
    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
    }, [theme]);

    // Listen for system theme changes
    useEffect(() => {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

        const handleChange = (e: MediaQueryListEvent) => {
            // Only auto-change if user hasn't set a preference
            if (!localStorage.getItem('highnoon-theme')) {
                setTheme(e.matches ? 'dark' : 'light');
            }
        };

        mediaQuery.addEventListener('change', handleChange);
        return () => mediaQuery.removeEventListener('change', handleChange);
    }, [setTheme]);

    return (
        <ThemeContext.Provider value= {{ theme, setTheme, toggleTheme }
}>
    { children }
    </ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextValue {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error('useTheme must be used within a ThemeProvider');
    }
    return context;
}
