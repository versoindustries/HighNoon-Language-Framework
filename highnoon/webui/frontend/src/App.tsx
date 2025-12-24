// HighNoon Dashboard - Main App Component
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from './hooks/useTheme';
import { Sidebar } from './components/layout';
import { Dashboard } from './pages/Dashboard';
import { HPO } from './pages/HPO';
import { Training } from './pages/Training';
import { Curriculum } from './pages/Curriculum';
import { Datasets } from './pages/Datasets';
import { Settings } from './pages/Settings';
import './styles/globals.css';
import './App.css';

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

function AppContent() {
  return (
    <div className="app-layout">
      <Sidebar />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/hpo" element={<HPO />} />
          <Route path="/training" element={<Training />} />
          <Route path="/curriculum" element={<Curriculum />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
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
