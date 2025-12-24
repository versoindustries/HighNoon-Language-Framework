// HighNoon Dashboard - Sidebar Navigation
import { useState, useEffect } from 'react';
import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import {
    LayoutDashboard,
    Sliders,
    GraduationCap,
    Database,
    Zap,
    Moon,
    Sun,
    Settings,
    HelpCircle,
} from 'lucide-react';
import { useTheme } from '../../hooks/useTheme';
import { TutorialMode } from '../TutorialMode';
import './Sidebar.css';

interface NavItem {
    path: string;
    label: string;
    icon: React.ReactNode;
    badge?: number;
}

const navItems: NavItem[] = [
    { path: '/', label: 'Dashboard', icon: <LayoutDashboard size={20} /> },
    { path: '/datasets', label: 'Datasets', icon: <Database size={20} /> },
    { path: '/curriculum', label: 'Curriculum', icon: <GraduationCap size={20} /> },
    { path: '/hpo', label: 'HPO Orchestrator', icon: <Sliders size={20} /> },
    { path: '/training', label: 'Training', icon: <Zap size={20} /> },
];

export function Sidebar() {
    const { theme, toggleTheme } = useTheme();
    const location = useLocation();
    const navigate = useNavigate();
    const [showTutorial, setShowTutorial] = useState(false);

    // Check if user has seen tutorial on first visit
    useEffect(() => {
        const hasSeenTutorial = localStorage.getItem('highnoon-tutorial-seen');
        if (!hasSeenTutorial) {
            // Auto-show tutorial for first-time users after a short delay
            const timer = setTimeout(() => {
                setShowTutorial(true);
            }, 500);
            return () => clearTimeout(timer);
        }
    }, []);

    const handleTutorialNavigate = (path: string) => {
        navigate(path);
    };

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div className="sidebar-logo">
                    <span className="logo-icon">🌵</span>
                    <div className="logo-text">
                        <span className="logo-title">HighNoon</span>
                        <span className="logo-subtitle">Lite Edition</span>
                    </div>
                </div>
            </div>

            <nav className="sidebar-nav">
                <ul className="nav-list">
                    {navItems.map((item) => (
                        <li key={item.path}>
                            <NavLink
                                to={item.path}
                                className={({ isActive }) =>
                                    `nav-link ${isActive ? 'nav-link-active' : ''}`
                                }
                                data-section={item.path === '/' ? 'dashboard' : item.path.slice(1)}
                            >
                                <span className="nav-icon">{item.icon}</span>
                                <span className="nav-label">{item.label}</span>
                                {item.badge !== undefined && (
                                    <span className="nav-badge">{item.badge}</span>
                                )}
                            </NavLink>
                        </li>
                    ))}
                </ul>
            </nav>

            <div className="sidebar-footer">
                <button
                    className="tutorial-trigger"
                    onClick={() => setShowTutorial(true)}
                    title="Start guided tutorial"
                >
                    <HelpCircle size={18} />
                    <span>Tutorial Mode</span>
                </button>

                <button className="theme-toggle" onClick={toggleTheme}>
                    {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
                    <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
                </button>

                <NavLink to="/settings" className="nav-link nav-link-compact">
                    <span className="nav-icon"><Settings size={18} /></span>
                    <span className="nav-label">Settings</span>
                </NavLink>

                <div className="sidebar-version">
                    <span>v1.0.0 Lite</span>
                    <span className="version-limit">20B Max</span>
                </div>
            </div>

            {/* Tutorial Modal */}
            <TutorialMode
                open={showTutorial}
                onClose={() => setShowTutorial(false)}
                onNavigate={handleTutorialNavigate}
            />
        </aside>
    );
}
