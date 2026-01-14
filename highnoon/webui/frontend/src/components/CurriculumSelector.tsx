// HighNoon Dashboard - Enhanced Curriculum Selector Component
// Uses existing curriculum API and data structures

import { useState, useMemo, useCallback, useEffect } from 'react';
import {
    Search,
    ChevronDown,
    ChevronRight,
    Star,
    Clock,
    Database,
    Layers,
    X,
    Filter,
    SortAsc
} from 'lucide-react';
import type { Curriculum, CurriculumStage } from '../api/types';
import './CurriculumSelector.css';

// =============================================================================
// TYPES
// =============================================================================

interface CurriculumSelectorProps {
    /** All available curricula (user-created + presets) */
    curricula: Curriculum[];
    /** Whether curricula are still loading */
    loading: boolean;
    /** Currently selected curriculum ID */
    selectedId: string;
    /** Called when a curriculum is selected */
    onSelect: (curriculumId: string) => void;
}

type SortOption = 'recent' | 'name' | 'stages';
type FilterTag = 'all' | 'preset' | 'custom' | 'multi-stage';

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Fuzzy search - checks if query matches any part of the name
 */
function fuzzyMatch(text: string, query: string): boolean {
    const lowerText = text.toLowerCase();
    const lowerQuery = query.toLowerCase();

    // Exact substring match
    if (lowerText.includes(lowerQuery)) return true;

    // Simple character-by-character fuzzy match
    let queryIndex = 0;
    for (const char of lowerText) {
        if (char === lowerQuery[queryIndex]) {
            queryIndex++;
            if (queryIndex === lowerQuery.length) return true;
        }
    }

    return false;
}

/**
 * Estimate training time based on curriculum stages
 */
function estimateTrainingTime(curriculum: Curriculum): string {
    const totalEpochs = curriculum.stages.reduce((sum, s) => sum + (s.epochs || 1), 0);
    const totalDatasets = curriculum.stages.reduce((sum, s) => sum + s.datasets.length, 0);

    // Rough estimate: 1-2 hours per epoch per dataset (varies widely)
    const minHours = totalEpochs * totalDatasets * 0.5;
    const maxHours = totalEpochs * totalDatasets * 2;

    if (maxHours < 1) return '< 1h';
    if (minHours >= 24) return `${Math.round(minHours / 24)}-${Math.round(maxHours / 24)} days`;
    return `${Math.round(minHours)}-${Math.round(maxHours)}h`;
}

/**
 * Get total dataset count across all stages
 */
function getTotalDatasets(curriculum: Curriculum): number {
    return curriculum.stages.reduce((sum, s) => sum + s.datasets.length, 0);
}

/**
 * Check if curriculum is a preset (starts with [Preset])
 */
function isPreset(curriculum: Curriculum): boolean {
    return curriculum.name.startsWith('[Preset]');
}

/**
 * Get display name (strip [Preset] prefix if present)
 */
function getDisplayName(curriculum: Curriculum): string {
    return curriculum.name.replace(/^\[Preset\]\s*/, '');
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

/**
 * Stage preview row showing the pipeline of stages
 */
function StagePipeline({ stages }: { stages: CurriculumStage[] }) {
    if (stages.length === 0) {
        return <span className="stage-pipeline-empty">No stages defined</span>;
    }

    // Show first 3 stages with arrow connectors
    const displayedStages = stages.slice(0, 3);
    const remaining = stages.length - 3;

    return (
        <div className="stage-pipeline">
            {displayedStages.map((stage, index) => (
                <span key={stage.name} className="stage-pipeline-item">
                    {index > 0 && <span className="stage-pipeline-arrow">→</span>}
                    <span className="stage-pipeline-name">
                        {stage.display_name || stage.name}
                    </span>
                    <span className="stage-pipeline-datasets">
                        ({stage.datasets.length})
                    </span>
                </span>
            ))}
            {remaining > 0 && (
                <span className="stage-pipeline-more">
                    +{remaining} more
                </span>
            )}
        </div>
    );
}

/**
 * Expanded view of a curriculum with full stage details
 */
function CurriculumDetails({ curriculum }: { curriculum: Curriculum }) {
    return (
        <div className="curriculum-details">
            <div className="curriculum-details-header">
                <h5>Stages</h5>
            </div>
            <div className="curriculum-details-stages">
                {curriculum.stages.map((stage, index) => (
                    <div key={stage.name} className="curriculum-stage-row">
                        <span className="stage-number">{index + 1}</span>
                        <div className="stage-info">
                            <span className="stage-name">{stage.display_name || stage.name}</span>
                            <span className="stage-meta">
                                {stage.epochs} epoch{stage.epochs !== 1 ? 's' : ''} ·
                                {stage.datasets.length} dataset{stage.datasets.length !== 1 ? 's' : ''} ·
                                LR: {stage.learning_rate}
                            </span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function CurriculumSelector({
    curricula,
    loading,
    selectedId,
    onSelect,
}: CurriculumSelectorProps) {
    // Search and filter state
    const [searchQuery, setSearchQuery] = useState('');
    const [activeFilter, setActiveFilter] = useState<FilterTag>('all');
    const [sortBy, setSortBy] = useState<SortOption>('recent');
    const [expandedId, setExpandedId] = useState<string | null>(null);

    // Filter tags
    const filterTags: { value: FilterTag; label: string }[] = [
        { value: 'all', label: 'All' },
        { value: 'preset', label: '⭐ Recommended' },
        { value: 'custom', label: 'Custom' },
        { value: 'multi-stage', label: 'Multi-Stage' },
    ];

    // Sort options
    const sortOptions: { value: SortOption; label: string }[] = [
        { value: 'recent', label: 'Recent' },
        { value: 'name', label: 'Name' },
        { value: 'stages', label: 'Stages' },
    ];

    // Filtered and sorted curricula
    const filteredCurricula = useMemo(() => {
        let result = [...curricula];

        // Apply search filter
        if (searchQuery.trim()) {
            result = result.filter(c => fuzzyMatch(c.name, searchQuery));
        }

        // Apply tag filter
        switch (activeFilter) {
            case 'preset':
                result = result.filter(c => isPreset(c));
                break;
            case 'custom':
                result = result.filter(c => !isPreset(c));
                break;
            case 'multi-stage':
                result = result.filter(c => c.stages.length > 1);
                break;
        }

        // Apply sorting
        switch (sortBy) {
            case 'name':
                result.sort((a, b) => a.name.localeCompare(b.name));
                break;
            case 'stages':
                result.sort((a, b) => b.stages.length - a.stages.length);
                break;
            case 'recent':
            default:
                // Presets first, then by updated_at descending
                result.sort((a, b) => {
                    const aPreset = isPreset(a);
                    const bPreset = isPreset(b);
                    if (aPreset !== bPreset) return aPreset ? -1 : 1;
                    return (b.updated_at || '').localeCompare(a.updated_at || '');
                });
                break;
        }

        return result;
    }, [curricula, searchQuery, activeFilter, sortBy]);

    // Handle curriculum selection
    const handleSelect = useCallback((id: string) => {
        onSelect(id);
    }, [onSelect]);

    // Handle expand/collapse
    const handleToggleExpand = useCallback((id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        setExpandedId(prev => prev === id ? null : id);
    }, []);

    // Clear search
    const handleClearSearch = useCallback(() => {
        setSearchQuery('');
    }, []);

    // Auto-expand selected curriculum
    useEffect(() => {
        if (selectedId && !expandedId) {
            setExpandedId(selectedId);
        }
    }, [selectedId]);

    // Loading state
    if (loading) {
        return (
            <div className="curriculum-selector">
                <div className="curriculum-loading">
                    <div className="skeleton skeleton-search" />
                    <div className="skeleton skeleton-filter" />
                    <div className="skeleton skeleton-card" />
                    <div className="skeleton skeleton-card" />
                </div>
            </div>
        );
    }

    // Empty state
    if (curricula.length === 0) {
        return (
            <div className="curriculum-selector">
                <div className="empty-state">
                    <Database className="empty-state__icon" />
                    <h4 className="empty-state__title">No Curricula Found</h4>
                    <p className="empty-state__description">
                        Create a curriculum in the Curriculum tab to get started with hyperparameter optimization.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="curriculum-selector">
            {/* Search Bar */}
            <div className="curriculum-search">
                <Search size={16} className="curriculum-search-icon" />
                <input
                    type="text"
                    placeholder="Search curricula..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="curriculum-search-input"
                    aria-label="Search curricula"
                />
                {searchQuery && (
                    <button
                        className="curriculum-search-clear"
                        onClick={handleClearSearch}
                        aria-label="Clear search"
                    >
                        <X size={14} />
                    </button>
                )}

                {/* Sort Dropdown */}
                <div className="curriculum-sort">
                    <SortAsc size={14} />
                    <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value as SortOption)}
                        className="curriculum-sort-select"
                        aria-label="Sort curricula"
                    >
                        {sortOptions.map(opt => (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
                    </select>
                </div>
            </div>

            {/* Filter Tags */}
            <div className="curriculum-filters">
                {filterTags.map(tag => (
                    <button
                        key={tag.value}
                        className={`curriculum-filter-tag ${activeFilter === tag.value ? 'active' : ''}`}
                        onClick={() => setActiveFilter(tag.value)}
                    >
                        {tag.label}
                    </button>
                ))}
            </div>

            {/* Results Count */}
            {searchQuery && (
                <div className="curriculum-results-count">
                    {filteredCurricula.length} result{filteredCurricula.length !== 1 ? 's' : ''} for "{searchQuery}"
                </div>
            )}

            {/* Curriculum List */}
            <div className="curriculum-list">
                {filteredCurricula.length === 0 ? (
                    <div className="curriculum-no-results">
                        <p>No curricula match your search criteria.</p>
                        <button onClick={handleClearSearch}>Clear filters</button>
                    </div>
                ) : (
                    filteredCurricula.map(curriculum => {
                        const isSelected = selectedId === curriculum.id;
                        const isExpanded = expandedId === curriculum.id;
                        const preset = isPreset(curriculum);

                        return (
                            <div
                                key={curriculum.id}
                                className={`curriculum-card ${isSelected ? 'selected' : ''} ${preset ? 'preset' : ''}`}
                                onClick={() => handleSelect(curriculum.id)}
                                role="button"
                                tabIndex={0}
                                aria-selected={isSelected}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' || e.key === ' ') {
                                        e.preventDefault();
                                        handleSelect(curriculum.id);
                                    }
                                }}
                            >
                                {/* Main Content */}
                                <div className="curriculum-card-header">
                                    <div className="curriculum-card-radio">
                                        <input
                                            type="radio"
                                            name="curriculum"
                                            value={curriculum.id}
                                            checked={isSelected}
                                            onChange={() => handleSelect(curriculum.id)}
                                            aria-label={`Select ${getDisplayName(curriculum)}`}
                                        />
                                    </div>

                                    <div className="curriculum-card-content">
                                        <div className="curriculum-card-title-row">
                                            <span className="curriculum-card-name">
                                                {getDisplayName(curriculum)}
                                            </span>
                                            {preset && (
                                                <span className="curriculum-badge preset">
                                                    <Star size={10} />
                                                    Recommended
                                                </span>
                                            )}
                                        </div>

                                        {/* Stage Pipeline Preview */}
                                        <div className="curriculum-card-stages">
                                            <StagePipeline stages={curriculum.stages} />
                                        </div>

                                        {/* Metadata Row */}
                                        <div className="curriculum-card-meta">
                                            <span className="curriculum-meta-item">
                                                <Clock size={12} />
                                                Est. {estimateTrainingTime(curriculum)}
                                            </span>
                                            <span className="curriculum-meta-item">
                                                <Layers size={12} />
                                                {curriculum.stages.length} stage{curriculum.stages.length !== 1 ? 's' : ''}
                                            </span>
                                            <span className="curriculum-meta-item">
                                                <Database size={12} />
                                                {getTotalDatasets(curriculum)} dataset{getTotalDatasets(curriculum) !== 1 ? 's' : ''}
                                            </span>
                                        </div>
                                    </div>

                                    {/* Expand Button */}
                                    <button
                                        className="curriculum-card-expand"
                                        onClick={(e) => handleToggleExpand(curriculum.id, e)}
                                        aria-expanded={isExpanded}
                                        aria-label={isExpanded ? 'Collapse details' : 'Expand details'}
                                    >
                                        {isExpanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                                    </button>
                                </div>

                                {/* Expanded Details */}
                                {isExpanded && (
                                    <CurriculumDetails curriculum={curriculum} />
                                )}
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
}
