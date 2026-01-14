// HighNoon Dashboard - Main App Component
import { lazy, Suspense, useEffect } from 'react';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from './hooks/useTheme';
import { useKeyboardShortcuts, KeyboardHelpModal } from './hooks/useKeyboardShortcuts';
import { useAnnouncer, useFocusManagement } from './hooks/useAnnouncer';
import { Sidebar } from './components/layout';
import './styles/globals.css';
import './styles/keyboard-help.css';
import './App.css';

// Lazy load page components for better initial load performance
const Dashboard = lazy(() => import('./pages/Dashboard').then(m => ({ default: m.Dashboard })));
const HPO = lazy(() => import('./pages/HPO').then(m => ({ default: m.HPO })));
const Training = lazy(() => import('./pages/Training').then(m => ({ default: m.Training })));
const Curriculum = lazy(() => import('./pages/Curriculum').then(m => ({ default: m.Curriculum })));
const Datasets = lazy(() => import('./pages/Datasets').then(m => ({ default: m.Datasets })));
const Settings = lazy(() => import('./pages/Settings').then(m => ({ default: m.Settings })));

// Loading fallback component
function PageLoader() {
  return (
    <div className="page-loader" role="progressbar" aria-label="Loading page">
      <div className="page-loader__spinner" />
      <span className="sr-only">Loading...</span>
    </div>
  );
}

// Create a client for React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30 * 1000, // 30 seconds
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// Route change announcer for accessibility
function RouteAnnouncer() {
  const location = useLocation();
  const { announcePageChange } = useAnnouncer();
  const { focusMainContent } = useFocusManagement();

  useEffect(() => {
    // Map paths to readable page names
    const pageNames: Record<string, string> = {
      '/': 'Dashboard',
      '/hpo': 'Smart Tuner',
      '/training': 'Training',
      '/curriculum': 'Curriculum Builder',
      '/datasets': 'Datasets',
      '/settings': 'Settings',
    };

    const pageName = pageNames[location.pathname] || 'Page';

    // Announce page change to screen readers
    announcePageChange(pageName);

    // Move focus to main content for keyboard users
    // Small delay to ensure content has rendered
    const timeout = setTimeout(() => {
      focusMainContent();
    }, 100);

    return () => clearTimeout(timeout);
  }, [location.pathname, announcePageChange, focusMainContent]);

  return null;
}

function AppContent() {
  const { showHelp, setShowHelp } = useKeyboardShortcuts();

  return (
    <>
      {/* Skip Navigation Link for keyboard users */}
      <a href="#main-content" className="skip-nav">
        Skip to main content
      </a>

      <div className="app-layout">
        <Sidebar />
        <main
          id="main-content"
          className="app-main"
          tabIndex={-1}
          role="main"
          aria-label="Main content"
        >
          {/* Announce route changes */}
          <RouteAnnouncer />

          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/hpo" element={<HPO />} />
              <Route path="/training" element={<Training />} />
              <Route path="/curriculum" element={<Curriculum />} />
              <Route path="/datasets" element={<Datasets />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Suspense>
        </main>
      </div>

      {/* Keyboard Shortcuts Help Modal */}
      <KeyboardHelpModal
        isOpen={showHelp}
        onClose={() => setShowHelp(false)}
      />
    </>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <BrowserRouter>
          <AppContent />
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
