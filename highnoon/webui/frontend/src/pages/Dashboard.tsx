// HighNoon Dashboard - Dashboard Page
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, TrendingUp, Clock, Cpu, Rocket, Layers, MessageCircle, Code, Brain, Target } from 'lucide-react';
import { Card, CardHeader, CardContent, StatCard, ProgressBar, Button } from '../components/ui';
import { QuickStartWizard } from '../components/QuickStartWizard';
import { TemplateGallery, TEMPLATES } from '../components/TemplateGallery';
import type { TrainingTemplate } from '../components/TemplateGallery';
import { trainingApi, hpoApi } from '../api/client';
import type { TrainingJobInfo, HPOSweepInfo } from '../api/types';
import './Dashboard.css';

// Top recommended templates to show on dashboard
const RECOMMENDED_TEMPLATES = TEMPLATES.filter(t => t.recommended).slice(0, 3);

// Format parameter count for display
function formatParams(num: number): string {
    if (num >= 1_000_000_000) return `${(num / 1_000_000_000).toFixed(1)}B`;
    if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(0)}M`;
    if (num >= 1_000) return `${(num / 1_000).toFixed(0)}K`;
    return num.toString();
}

// Format uptime from milliseconds
function formatUptime(startTimeMs: number): string {
    const elapsed = Date.now() - startTimeMs;
    const hours = Math.floor(elapsed / (1000 * 60 * 60));
    const mins = Math.floor((elapsed % (1000 * 60 * 60)) / (1000 * 60));
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
}

export function Dashboard() {
    const navigate = useNavigate();
    const [showQuickStart, setShowQuickStart] = useState(false);
    const [showTemplates, setShowTemplates] = useState(false);

    // Stats state
    const [activeJobs, setActiveJobs] = useState<TrainingJobInfo[]>([]);
    const [sweeps, setSweeps] = useState<HPOSweepInfo[]>([]);
    const [serverStartTime] = useState(() => Date.now()); // Approximate server start
    const [loading, setLoading] = useState(true);

    // Fetch stats on mount and periodically
    useEffect(() => {
        async function loadStats() {
            try {
                const [jobs, allSweeps] = await Promise.all([
                    trainingApi.listJobs(),
                    hpoApi.listSweeps(),
                ]);
                setActiveJobs(jobs.filter(j => j.state === 'running' || j.state === 'paused'));
                setSweeps(allSweeps);
            } catch (err) {
                console.error('Failed to load dashboard stats:', err);
            } finally {
                setLoading(false);
            }
        }
        loadStats();
        // Refresh every 30 seconds
        const interval = setInterval(loadStats, 30000);
        return () => clearInterval(interval);
    }, []);

    // Computed stats
    const runningJobsCount = activeJobs.length;
    const totalSweeps = sweeps.length;
    const runningSweeps = sweeps.filter(s => s.state === 'running').length;
    const completedSweeps = sweeps.filter(s => s.state === 'completed');
    const latestSweep = completedSweeps[0];
    const bestModelSize = latestSweep?.model_config?.param_budget;

    // Recent activity: combine jobs and sweeps
    const recentActivity = [
        ...sweeps.slice(0, 3).map(s => ({
            type: 'sweep' as const,
            id: s.sweep_id,
            state: s.state,
            info: `${s.completed_trials} trials, best: ${s.best_loss?.toFixed(4) ?? '—'}`,
            time: s.started_at,
        })),
        ...activeJobs.slice(0, 2).map(j => ({
            type: 'job' as const,
            id: j.job_id,
            state: j.state,
            info: `Step ${j.global_step}, loss: ${j.loss?.toFixed(4) ?? '—'}`,
            time: j.started_at,
        })),
    ].slice(0, 5);

    const handleQuickStartComplete = (jobId: string) => {
        console.log('QuickStart completed, job:', jobId);
        navigate('/training');
    };

    const handleTemplateSelect = (template: TrainingTemplate) => {
        // Store selected template in sessionStorage for Curriculum page to pick up
        sessionStorage.setItem('highnoon-selected-template', JSON.stringify({
            id: template.id,
            name: template.name,
        }));
        setShowTemplates(false);
        navigate('/curriculum');
    };

    const handleQuickLaunchTemplate = (templateId: string) => {
        const template = TEMPLATES.find(t => t.id === templateId);
        if (template) {
            handleTemplateSelect(template);
        }
    };

    return (
        <div className="page">
            <div className="page-header">
                <div className="page-header-content">
                    <h1 className="page-title">Dashboard</h1>
                    <p className="page-subtitle">
                        HighNoon Language Framework - Overview &amp; Quick Actions
                    </p>
                </div>
                <div className="page-header-actions">
                    <Button
                        variant="primary"
                        leftIcon={<Rocket size={18} />}
                        onClick={() => setShowQuickStart(true)}
                    >
                        Start Training
                    </Button>
                </div>
            </div>

            {/* Stats Row */}
            <div className="dashboard-stats">
                <StatCard
                    label="Active Jobs"
                    value={loading ? '—' : String(runningJobsCount)}
                    description={runningJobsCount === 0 ? 'No training in progress' : `${runningJobsCount} job${runningJobsCount > 1 ? 's' : ''} running`}
                    icon={<Activity size={20} />}
                />
                <StatCard
                    label="HPO Sweeps"
                    value={loading ? '—' : String(totalSweeps)}
                    description={runningSweeps > 0 ? `${runningSweeps} running` : totalSweeps > 0 ? `${completedSweeps.length} completed` : 'Ready to optimize'}
                    icon={<TrendingUp size={20} />}
                />
                <StatCard
                    label="Uptime"
                    value={formatUptime(serverStartTime)}
                    description="Server running"
                    icon={<Clock size={20} />}
                />
                <StatCard
                    label="Best Model"
                    value={bestModelSize ? formatParams(bestModelSize) : '—'}
                    description="Lite Edition (20B max)"
                    icon={<Cpu size={20} />}
                />
            </div>

            {/* Main Content Grid */}
            <div className="dashboard-grid">
                {/* Quick Actions */}
                <Card variant="glass" padding="lg">
                    <CardHeader
                        title="Quick Actions"
                        subtitle="Start training with HPO-optimized parameters"
                    />
                    <CardContent>
                        <div className="quick-actions">
                            <a href="/hpo" className="quick-action-card">
                                <div className="quick-action-icon">🎯</div>
                                <div className="quick-action-text">
                                    <strong>HPO Orchestrator</strong>
                                    <span>Configure and run hyperparameter optimization with forest-style parameter grouping</span>
                                </div>
                            </a>
                            <a href="/curriculum" className="quick-action-card">
                                <div className="quick-action-icon">📚</div>
                                <div className="quick-action-text">
                                    <strong>Curriculum Builder</strong>
                                    <span>Design multi-stage training curricula with HuggingFace datasets</span>
                                </div>
                            </a>
                            <a href="/datasets" className="quick-action-card">
                                <div className="quick-action-icon">🗄️</div>
                                <div className="quick-action-text">
                                    <strong>Dataset Browser</strong>
                                    <span>Browse and add datasets from HuggingFace Hub</span>
                                </div>
                            </a>
                            <button className="quick-action-card" onClick={() => setShowTemplates(true)}>
                                <div className="quick-action-icon">📋</div>
                                <div className="quick-action-text">
                                    <strong>Template Gallery</strong>
                                    <span>Pre-certified training recipes for common use cases</span>
                                </div>
                            </button>
                        </div>
                    </CardContent>
                </Card>

                {/* Baseline Curriculums Quick Launch */}
                <Card variant="glass" padding="lg">
                    <CardHeader
                        title="Baseline Curriculums"
                        subtitle="Launch pre-configured training recipes"
                        action={
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setShowTemplates(true)}
                            >
                                View All
                            </Button>
                        }
                    />
                    <CardContent>
                        <div className="baseline-grid">
                            {RECOMMENDED_TEMPLATES.map((template) => (
                                <button
                                    key={template.id}
                                    className="baseline-card"
                                    onClick={() => handleQuickLaunchTemplate(template.id)}
                                >
                                    <div className="baseline-icon">
                                        {template.icon}
                                    </div>
                                    <div className="baseline-content">
                                        <span className="baseline-name">{template.name}</span>
                                        <span className="baseline-description">
                                            {template.description.substring(0, 60)}...
                                        </span>
                                        <div className="baseline-meta">
                                            <span className="baseline-time">⏱ {template.estimatedTime}</span>
                                            <span className={`baseline-difficulty baseline-${template.difficulty}`}>
                                                {template.difficulty}
                                            </span>
                                        </div>
                                    </div>
                                </button>
                            ))}
                        </div>
                    </CardContent>
                </Card>

                {/* System Status */}
                <Card padding="lg">
                    <CardHeader title="System Limits" subtitle="Lite Edition Constraints" />
                    <CardContent>
                        <div className="limits-list">
                            <div className="limit-item">
                                <span className="limit-label">Max Parameters</span>
                                <span className="limit-value">20B</span>
                                <ProgressBar value={0} max={100} size="sm" variant="gradient" />
                            </div>
                            <div className="limit-item">
                                <span className="limit-label">Max Context Length</span>
                                <span className="limit-value">5M tokens</span>
                                <ProgressBar value={0} max={100} size="sm" variant="gradient" />
                            </div>
                            <div className="limit-item">
                                <span className="limit-label">Reasoning Blocks</span>
                                <span className="limit-value">24 max</span>
                                <ProgressBar value={0} max={100} size="sm" variant="gradient" />
                            </div>
                            <div className="limit-item">
                                <span className="limit-label">MoE Experts</span>
                                <span className="limit-value">12 max</span>
                                <ProgressBar value={0} max={100} size="sm" variant="gradient" />
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Recent Activity */}
                <Card padding="lg" className="dashboard-activity">
                    <CardHeader title="Recent Activity" subtitle="Latest training runs and HPO sweeps" />
                    <CardContent>
                        {loading ? (
                            <div className="empty-state">
                                <p>Loading...</p>
                            </div>
                        ) : recentActivity.length === 0 ? (
                            <div className="empty-state">
                                <div className="empty-icon">📊</div>
                                <p>No recent activity</p>
                                <span>Start an HPO sweep to see activity here</span>
                            </div>
                        ) : (
                            <div className="activity-list">
                                {recentActivity.map((item) => (
                                    <div key={item.id} className="activity-item">
                                        <div className="activity-icon">
                                            {item.type === 'sweep' ? <TrendingUp size={16} /> : <Activity size={16} />}
                                        </div>
                                        <div className="activity-content">
                                            <div className="activity-header">
                                                <span className="activity-id">{item.id}</span>
                                                <span className={`activity-state activity-state-${item.state}`}>
                                                    {item.state}
                                                </span>
                                            </div>
                                            <span className="activity-info">{item.info}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>

            {/* QuickStart Wizard Modal */}
            <QuickStartWizard
                open={showQuickStart}
                onClose={() => setShowQuickStart(false)}
                onComplete={handleQuickStartComplete}
            />

            {/* Template Gallery Modal */}
            <TemplateGallery
                open={showTemplates}
                onClose={() => setShowTemplates(false)}
                onSelectTemplate={handleTemplateSelect}
            />
        </div>
    );
}
