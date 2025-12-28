// HPOImportanceChart.tsx - Hyperparameter Importance Visualization
// Displays fANOVA analysis results with bar charts and marginal curves

import { useState, useEffect, useCallback } from 'react';
import {
    BarChart3,
    TrendingUp,
    AlertCircle,
    RefreshCw,
    Info,
    ChevronDown,
    ChevronUp,
} from 'lucide-react';
import type { ImportanceAnalysisResponse, ParameterImportance, MarginalCurve } from '../api/types';
import './HPOImportanceChart.css';

interface HPOImportanceChartProps {
    sweepId: string;
    isRunning: boolean;
    completedTrials: number;
}

export default function HPOImportanceChart({
    sweepId,
    isRunning,
    completedTrials,
}: HPOImportanceChartProps) {
    const [data, setData] = useState<ImportanceAnalysisResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [expanded, setExpanded] = useState(true);
    const [selectedParam, setSelectedParam] = useState<string | null>(null);

    const fetchImportance = useCallback(async () => {
        if (completedTrials < 10) {
            setError(`Need at least 10 trials (${completedTrials}/10)`);
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`/api/hpo/sweep/${sweepId}/importance`);
            const result = await response.json();

            if (result.error) {
                setError(result.message || result.error);
            } else {
                setData(result);
                // Auto-select first param for marginal curve
                if (result.importance?.individual?.length > 0) {
                    setSelectedParam(result.importance.individual[0].name);
                }
            }
        } catch (err) {
            setError('Failed to fetch importance analysis');
        } finally {
            setLoading(false);
        }
    }, [sweepId, completedTrials]);

    // Auto-refresh when trials complete (but not too often)
    useEffect(() => {
        if (completedTrials >= 10 && completedTrials % 5 === 0) {
            fetchImportance();
        }
    }, [completedTrials, fetchImportance]);

    // Format percentage
    const formatPct = (value: number) => `${(value * 100).toFixed(1)}%`;

    // Get bar color based on importance
    const getBarColor = (importance: number) => {
        if (importance >= 0.3) return 'var(--importance-high)';
        if (importance >= 0.1) return 'var(--importance-medium)';
        return 'var(--importance-low)';
    };

    // Render importance bar chart
    const renderImportanceChart = () => {
        if (!data?.importance?.individual) return null;

        const maxImportance = Math.max(...data.importance.individual.map(p => p.importance));

        return (
            <div className="importance-chart">
                <h4>
                    <BarChart3 size={16} />
                    Parameter Importance
                </h4>
                <div className="importance-bars">
                    {data.importance.individual.map((param: ParameterImportance) => (
                        <div
                            key={param.name}
                            className={`importance-bar-row ${selectedParam === param.name ? 'selected' : ''}`}
                            onClick={() => setSelectedParam(param.name)}
                        >
                            <div className="param-name" title={param.name}>
                                {param.name}
                                {param.is_categorical && <span className="cat-badge">CAT</span>}
                            </div>
                            <div className="bar-container">
                                <div
                                    className="bar-fill"
                                    style={{
                                        width: `${(param.importance / maxImportance) * 100}%`,
                                        backgroundColor: getBarColor(param.importance),
                                    }}
                                />
                            </div>
                            <div className="importance-value">
                                {formatPct(param.importance)}
                            </div>
                        </div>
                    ))}
                </div>
                <div className="chart-legend">
                    <span className="legend-item">
                        <span className="legend-dot high" /> &gt;30% (Critical)
                    </span>
                    <span className="legend-item">
                        <span className="legend-dot medium" /> 10-30% (Important)
                    </span>
                    <span className="legend-item">
                        <span className="legend-dot low" /> &lt;10% (Minor)
                    </span>
                </div>
            </div>
        );
    };

    // Render marginal curve
    const renderMarginalCurve = () => {
        if (!selectedParam || !data?.marginal_curves?.[selectedParam]) {
            return (
                <div className="marginal-curve placeholder">
                    <p>Select a parameter to see its effect on loss</p>
                </div>
            );
        }

        const curve: MarginalCurve = data.marginal_curves[selectedParam];
        const yMin = Math.min(...curve.y_mean);
        const yMax = Math.max(...curve.y_mean);
        const yRange = yMax - yMin || 1;

        return (
            <div className="marginal-curve">
                <h4>
                    <TrendingUp size={16} />
                    Effect of "{selectedParam}" on Loss
                </h4>
                {curve.is_categorical ? (
                    // Categorical: bar chart
                    <div className="categorical-chart">
                        {curve.x_values.map((val, i) => (
                            <div key={String(val)} className="cat-bar-group">
                                <div className="cat-bar-wrapper">
                                    <div
                                        className="cat-bar"
                                        style={{
                                            height: `${((curve.y_mean[i] - yMin) / yRange) * 100}%`,
                                        }}
                                        title={`${val}: ${curve.y_mean[i].toFixed(4)}`}
                                    />
                                </div>
                                <span className="cat-label">{String(val)}</span>
                            </div>
                        ))}
                    </div>
                ) : (
                    // Continuous: line chart (SVG)
                    <svg className="line-chart" viewBox="0 0 300 100" preserveAspectRatio="none">
                        {/* Mean line */}
                        <polyline
                            fill="none"
                            stroke="var(--accent-blue)"
                            strokeWidth="2"
                            points={curve.x_values.map((_, i) => {
                                const x = (i / (curve.x_values.length - 1)) * 300;
                                const y = 100 - ((curve.y_mean[i] - yMin) / yRange) * 100;
                                return `${x},${y}`;
                            }).join(' ')}
                        />
                        {/* Std deviation band */}
                        <path
                            fill="var(--accent-blue)"
                            fillOpacity="0.2"
                            d={`
                                M ${0} ${100 - ((curve.y_mean[0] - curve.y_std[0] - yMin) / yRange) * 100}
                                ${curve.x_values.map((_, i) => {
                                const x = (i / (curve.x_values.length - 1)) * 300;
                                const y = 100 - ((curve.y_mean[i] + curve.y_std[i] - yMin) / yRange) * 100;
                                return `L ${x} ${y}`;
                            }).join(' ')}
                                ${curve.x_values.slice().reverse().map((_, i) => {
                                const idx = curve.x_values.length - 1 - i;
                                const x = (idx / (curve.x_values.length - 1)) * 300;
                                const y = 100 - ((curve.y_mean[idx] - curve.y_std[idx] - yMin) / yRange) * 100;
                                return `L ${x} ${y}`;
                            }).join(' ')}
                                Z
                            `}
                        />
                    </svg>
                )}
                <div className="curve-footer">
                    <span>Loss range: {yMin.toFixed(4)} - {yMax.toFixed(4)}</span>
                </div>
            </div>
        );
    };

    // Render interactions
    const renderInteractions = () => {
        const interactions = data?.importance?.interactions?.filter(i => i.is_significant);
        if (!interactions?.length) return null;

        return (
            <div className="interactions">
                <h4>
                    <Info size={16} />
                    Significant Interactions
                </h4>
                <ul>
                    {interactions.slice(0, 3).map((inter, i) => (
                        <li key={i}>
                            <strong>{inter.param1}</strong> × <strong>{inter.param2}</strong>
                            <span className="inter-pct">{formatPct(inter.importance)}</span>
                        </li>
                    ))}
                </ul>
            </div>
        );
    };

    return (
        <div className="hpo-importance-chart">
            <div className="importance-header" onClick={() => setExpanded(!expanded)}>
                <h3>
                    <BarChart3 size={18} />
                    Hyperparameter Importance (fANOVA)
                </h3>
                <div className="header-actions">
                    {data?.importance && (
                        <span className="explained-var">
                            R² = {formatPct(data.importance.explained_variance)}
                        </span>
                    )}
                    <button
                        className="refresh-btn"
                        onClick={(e) => {
                            e.stopPropagation();
                            fetchImportance();
                        }}
                        disabled={loading || completedTrials < 10}
                        title="Refresh analysis"
                    >
                        <RefreshCw size={14} className={loading ? 'spinning' : ''} />
                    </button>
                    {expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                </div>
            </div>

            {expanded && (
                <div className="importance-content">
                    {error && (
                        <div className="importance-error">
                            <AlertCircle size={16} />
                            <span>{error}</span>
                        </div>
                    )}

                    {loading && (
                        <div className="importance-loading">
                            <RefreshCw size={20} className="spinning" />
                            <span>Analyzing hyperparameters...</span>
                        </div>
                    )}

                    {!loading && !error && data && (
                        <div className="importance-grid">
                            {renderImportanceChart()}
                            {renderMarginalCurve()}
                            {renderInteractions()}
                        </div>
                    )}

                    {!loading && !error && !data && completedTrials >= 10 && (
                        <div className="importance-placeholder">
                            <button onClick={fetchImportance} className="analyze-btn">
                                <BarChart3 size={16} />
                                Analyze Hyperparameter Importance
                            </button>
                        </div>
                    )}

                    {completedTrials < 10 && (
                        <div className="importance-pending">
                            <Info size={16} />
                            <span>
                                Importance analysis available after 10 trials
                                ({completedTrials}/10)
                            </span>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
