// HighNoon Dashboard - Settings Page
// Distributed Training Configuration and System Settings
import { useState, useEffect, useCallback } from 'react';
import {
    Server,
    Laptop,
    Network,
    Copy,
    Check,
    RefreshCw,
    Wifi,
    WifiOff,
    Play,
    Square,
    Trash2,
    Info,
    ChevronDown,
    ChevronUp,
    Cpu,
    HardDrive,
    Clock,
    Settings as SettingsIcon,
    Crown,
    ExternalLink,
    Save,
    Zap,
    Activity,
    RotateCcw,
    KeyRound,
    Eye,
    EyeOff,
    AlertTriangle,
} from 'lucide-react';
import { Card, CardHeader, CardContent, Button, Input } from '../components/ui';
import './Settings.css';

// Types
interface WorkerInfo {
    worker_id: string;
    hostname: string;
    address: string;
    status: string;
    cpu_count: number;
    memory_gb: number;
    connected_at: string;
    last_heartbeat: string;
    task_index: number;
}

interface ClusterStatus {
    role: string;
    cluster_secret: string | null;
    workers: WorkerInfo[];
    is_ready: boolean;
    tf_config: Record<string, unknown> | null;
    host_address: string | null;
    is_training: boolean;
    error: string | null;
}

interface SystemInfo {
    hostname: string;
    cpu_count: number;
    memory_gb: number;
    local_ip: string;
}

type ClusterRole = 'standalone' | 'host' | 'worker';

// Attribution types
interface AttributionConfig {
    framework_name: string;
    author: string;
    copyright_notice: string;
    version: string;
    support_url: string;
}

interface AttributionResponse {
    attribution: AttributionConfig;
    edition: string;
    edition_code: number;
    is_customizable: boolean;
    is_custom: boolean;
}

const API_BASE = 'http://localhost:8000';

export function Settings() {
    // selectedRole is for UI selection (before connecting)
    // actualRole comes from the backend and reflects true state
    const [selectedRole, setSelectedRole] = useState<ClusterRole>('standalone');
    const [actualRole, setActualRole] = useState<ClusterRole>('standalone');
    const [clusterStatus, setClusterStatus] = useState<ClusterStatus | null>(null);
    const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [copied, setCopied] = useState(false);
    const [showGuide, setShowGuide] = useState(true);

    // Host config
    const [hostPort, setHostPort] = useState(12345);
    const [checkpointDir, setCheckpointDir] = useState('/shared/checkpoints');
    const [commProtocol, setCommProtocol] = useState<'ring' | 'auto'>('ring');

    // Worker config
    const [hostAddress, setHostAddress] = useState('');
    const [clusterSecret, setClusterSecret] = useState('');
    const [connectionTesting, setConnectionTesting] = useState(false);

    // Attribution state
    const [attributionData, setAttributionData] = useState<AttributionResponse | null>(null);
    const [attributionForm, setAttributionForm] = useState<AttributionConfig>({
        framework_name: 'HighNoon Language Framework',
        author: '',
        copyright_notice: '',
        version: '1.0.0',
        support_url: '',
    });
    const [attributionLoading, setAttributionLoading] = useState(false);
    const [attributionError, setAttributionError] = useState<string | null>(null);
    const [attributionSuccess, setAttributionSuccess] = useState(false);

    // HuggingFace token state
    const [hfTokenConfigured, setHfTokenConfigured] = useState(false);
    const [hfTokenPrefix, setHfTokenPrefix] = useState<string | null>(null);
    const [hfToken, setHfToken] = useState('');
    const [hfTokenLoading, setHfTokenLoading] = useState(false);
    const [hfTokenError, setHfTokenError] = useState<string | null>(null);
    const [hfTokenSuccess, setHfTokenSuccess] = useState(false);
    const [hfTokenValidating, setHfTokenValidating] = useState(false);
    const [hfUsername, setHfUsername] = useState<string | null>(null);
    const [showHfToken, setShowHfToken] = useState(false);

    // Fetch cluster status
    const fetchStatus = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/distributed/status`);
            if (response.ok) {
                const data = await response.json();
                setClusterStatus(data);
                setActualRole(data.role as ClusterRole);
                setError(null);
            }
        } catch {
            // Silent fail for status polling
        }
    }, []);

    // Fetch system info
    const fetchSystemInfo = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/distributed/system-info`);
            if (response.ok) {
                const data = await response.json();
                setSystemInfo(data);
            }
        } catch {
            // Silent fail
        }
    }, []);

    // Fetch attribution settings
    const fetchAttribution = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/attribution`);
            if (response.ok) {
                const data = await response.json();
                setAttributionData(data);
                setAttributionForm(data.attribution);
            }
        } catch {
            // Silent fail
        }
    }, []);

    // Fetch HuggingFace token status
    const fetchHfTokenStatus = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/settings/hf-token/status`);
            if (response.ok) {
                const data = await response.json();
                setHfTokenConfigured(data.configured);
                setHfTokenPrefix(data.token_prefix || null);
            }
        } catch {
            // Silent fail
        }
    }, []);

    // Initial load and polling
    useEffect(() => {
        fetchStatus();
        fetchSystemInfo();
        fetchAttribution();
        fetchHfTokenStatus();
        const interval = setInterval(fetchStatus, 2000);
        return () => clearInterval(interval);
    }, [fetchStatus, fetchSystemInfo, fetchAttribution, fetchHfTokenStatus]);

    // Start hosting
    const handleStartHost = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE}/api/distributed/start-host`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    port: hostPort,
                    shared_checkpoint_dir: checkpointDir,
                    communication_protocol: commProtocol,
                }),
            });
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to start host');
            }
            await fetchStatus();
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to start host');
        } finally {
            setLoading(false);
        }
    };

    // Join cluster
    const handleJoinCluster = async () => {
        if (!hostAddress || !clusterSecret) {
            setError('Please enter host address and cluster secret');
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE}/api/distributed/join-cluster`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    host_address: hostAddress,
                    cluster_secret: clusterSecret,
                }),
            });
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to join cluster');
            }
            await fetchStatus();
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to join cluster');
        } finally {
            setLoading(false);
        }
    };

    // Disconnect
    const handleDisconnect = async () => {
        setLoading(true);
        setError(null);
        try {
            await fetch(`${API_BASE}/api/distributed/disconnect`, {
                method: 'POST',
            });
            await fetchStatus();
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to disconnect');
        } finally {
            setLoading(false);
        }
    };

    // Remove worker
    const handleRemoveWorker = async (workerId: string) => {
        try {
            await fetch(`${API_BASE}/api/distributed/workers/${workerId}`, {
                method: 'DELETE',
            });
            await fetchStatus();
        } catch {
            setError('Failed to remove worker');
        }
    };

    // Copy secret to clipboard
    const handleCopySecret = () => {
        if (clusterStatus?.cluster_secret) {
            navigator.clipboard.writeText(clusterStatus.cluster_secret);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    // Test connection
    const handleTestConnection = async () => {
        if (!hostAddress) return;
        setConnectionTesting(true);
        // Simulate connection test (actual implementation would ping the host)
        await new Promise((resolve) => setTimeout(resolve, 1500));
        setConnectionTesting(false);
    };

    // Update attribution
    const handleSaveAttribution = async () => {
        setAttributionLoading(true);
        setAttributionError(null);
        setAttributionSuccess(false);
        try {
            const response = await fetch(`${API_BASE}/api/attribution`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(attributionForm),
            });
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to save attribution');
            }
            const data = await response.json();
            setAttributionData(data);
            setAttributionSuccess(true);
            setTimeout(() => setAttributionSuccess(false), 3000);
        } catch (e) {
            setAttributionError(e instanceof Error ? e.message : 'Failed to save attribution');
        } finally {
            setAttributionLoading(false);
        }
    };

    // Reset attribution
    const handleResetAttribution = async () => {
        setAttributionLoading(true);
        setAttributionError(null);
        try {
            const response = await fetch(`${API_BASE}/api/attribution/reset`, {
                method: 'POST',
            });
            if (response.ok) {
                const data = await response.json();
                setAttributionData(data);
                setAttributionForm(data.attribution);
                setAttributionSuccess(true);
                setTimeout(() => setAttributionSuccess(false), 3000);
            }
        } catch (e) {
            setAttributionError(e instanceof Error ? e.message : 'Failed to reset attribution');
        } finally {
            setAttributionLoading(false);
        }
    };

    // Save HuggingFace token
    const handleSaveHfToken = async () => {
        if (!hfToken.trim()) {
            setHfTokenError('Please enter a token');
            return;
        }
        if (!hfToken.startsWith('hf_')) {
            setHfTokenError("Token must start with 'hf_'");
            return;
        }

        setHfTokenLoading(true);
        setHfTokenError(null);
        setHfTokenSuccess(false);
        try {
            const response = await fetch(`${API_BASE}/api/settings/hf-token`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ token: hfToken }),
            });
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to save token');
            }
            const data = await response.json();
            setHfTokenConfigured(true);
            setHfTokenPrefix(data.token_prefix);
            setHfToken(''); // Clear the input
            setShowHfToken(false);
            setHfTokenSuccess(true);
            setTimeout(() => setHfTokenSuccess(false), 3000);
        } catch (e) {
            setHfTokenError(e instanceof Error ? e.message : 'Failed to save token');
        } finally {
            setHfTokenLoading(false);
        }
    };

    // Validate HuggingFace token
    const handleValidateHfToken = async () => {
        const tokenToValidate = hfToken.trim() || undefined;
        if (!tokenToValidate && !hfTokenConfigured) {
            setHfTokenError('Please enter a token to validate');
            return;
        }

        setHfTokenValidating(true);
        setHfTokenError(null);
        setHfUsername(null);
        try {
            const response = await fetch(`${API_BASE}/api/settings/hf-token/validate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(tokenToValidate ? { token: tokenToValidate } : {}),
            });
            const data = await response.json();
            if (data.valid) {
                setHfUsername(data.username);
                setHfTokenSuccess(true);
                setTimeout(() => setHfTokenSuccess(false), 3000);
            } else {
                setHfTokenError(data.error || 'Token validation failed');
            }
        } catch (e) {
            setHfTokenError(e instanceof Error ? e.message : 'Failed to validate token');
        } finally {
            setHfTokenValidating(false);
        }
    };

    // Clear HuggingFace token
    const handleClearHfToken = async () => {
        setHfTokenLoading(true);
        setHfTokenError(null);
        try {
            await fetch(`${API_BASE}/api/settings/hf-token`, { method: 'DELETE' });
            setHfTokenConfigured(false);
            setHfTokenPrefix(null);
            setHfUsername(null);
            setHfTokenSuccess(true);
            setTimeout(() => setHfTokenSuccess(false), 3000);
        } catch (e) {
            setHfTokenError(e instanceof Error ? e.message : 'Failed to clear token');
        } finally {
            setHfTokenLoading(false);
        }
    };

    const isConnected = actualRole !== 'standalone';

    return (
        <div className="page settings-page">
            <div className="page-header">
                <div className="page-header-content">
                    <h1 className="page-title">Settings</h1>
                    <p className="page-subtitle">
                        Configure distributed training and system preferences
                    </p>
                </div>
            </div>

            <div className="settings-layout">
                {/* System Info Card */}
                {systemInfo && (
                    <Card variant="glass" className="system-info-card">
                        <CardContent>
                            <div className="system-info-grid">
                                <div className="system-info-item">
                                    <Laptop size={18} />
                                    <span className="info-label">Hostname</span>
                                    <span className="info-value">{systemInfo.hostname}</span>
                                </div>
                                <div className="system-info-item">
                                    <Cpu size={18} />
                                    <span className="info-label">CPU Cores</span>
                                    <span className="info-value">{systemInfo.cpu_count}</span>
                                </div>
                                <div className="system-info-item">
                                    <HardDrive size={18} />
                                    <span className="info-label">Memory</span>
                                    <span className="info-value">{systemInfo.memory_gb.toFixed(1)} GB</span>
                                </div>
                                <div className="system-info-item">
                                    <Network size={18} />
                                    <span className="info-label">Local IP</span>
                                    <span className="info-value">{systemInfo.local_ip}</span>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                )}

                {/* Distributed Training Section */}
                <div className="settings-section">
                    <h2 className="section-title">
                        <Network size={22} />
                        Distributed Training
                    </h2>

                    {error && (
                        <div className="error-banner">
                            <Info size={18} />
                            {error}
                            <button onClick={() => setError(null)}>&times;</button>
                        </div>
                    )}

                    {/* Role Selection Cards */}
                    {!isConnected && (
                        <div className="role-cards">
                            <button
                                className={`role-card ${selectedRole === 'standalone' ? 'active' : ''}`}
                                onClick={() => setSelectedRole('standalone')}
                            >
                                <Laptop size={32} />
                                <h3>Standalone</h3>
                                <p>Single machine training (default)</p>
                            </button>

                            <button
                                className={`role-card ${selectedRole === 'host' ? 'active' : ''}`}
                                onClick={() => setSelectedRole('host')}
                            >
                                <Server size={32} />
                                <h3>Host</h3>
                                <p>Coordinate cluster as chief node</p>
                            </button>

                            <button
                                className={`role-card ${selectedRole === 'worker' ? 'active' : ''}`}
                                onClick={() => setSelectedRole('worker')}
                            >
                                <Network size={32} />
                                <h3>Worker</h3>
                                <p>Join existing training cluster</p>
                            </button>
                        </div>
                    )}

                    {/* Host Configuration */}
                    {selectedRole === 'host' && !isConnected && (
                        <Card padding="lg" className="config-card">
                            <CardHeader
                                title="Host Configuration"
                                subtitle="Configure this machine as the cluster coordinator"
                            />
                            <CardContent>
                                <div className="config-form">
                                    <div className="form-row">
                                        <label htmlFor="host-port">Port</label>
                                        <Input
                                            id="host-port"
                                            type="number"
                                            value={hostPort}
                                            onChange={(e) => setHostPort(parseInt(e.target.value) || 12345)}
                                            min={1024}
                                            max={65535}
                                        />
                                    </div>

                                    <div className="form-row">
                                        <label htmlFor="checkpoint-dir">Shared Checkpoint Directory</label>
                                        <Input
                                            id="checkpoint-dir"
                                            value={checkpointDir}
                                            onChange={(e) => setCheckpointDir(e.target.value)}
                                            placeholder="/shared/checkpoints"
                                        />
                                    </div>

                                    <div className="form-row">
                                        <label htmlFor="comm-protocol">Communication Protocol</label>
                                        <select
                                            id="comm-protocol"
                                            value={commProtocol}
                                            onChange={(e) => setCommProtocol(e.target.value as 'ring' | 'auto')}
                                            className="select-input"
                                        >
                                            <option value="ring">Ring All-Reduce (CPU optimized)</option>
                                            <option value="auto">Auto (TensorFlow default)</option>
                                        </select>
                                    </div>

                                    <Button
                                        variant="primary"
                                        leftIcon={<Play size={18} />}
                                        onClick={handleStartHost}
                                        loading={loading}
                                        className="start-button"
                                    >
                                        Start Hosting
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    )}

                    {/* Worker Configuration */}
                    {selectedRole === 'worker' && !isConnected && (
                        <Card padding="lg" className="config-card">
                            <CardHeader
                                title="Worker Configuration"
                                subtitle="Connect to an existing cluster"
                            />
                            <CardContent>
                                <div className="config-form">
                                    <div className="form-row">
                                        <label htmlFor="host-address">Host Address</label>
                                        <div className="input-with-button">
                                            <Input
                                                id="host-address"
                                                value={hostAddress}
                                                onChange={(e) => setHostAddress(e.target.value)}
                                                placeholder="192.168.1.100:12345"
                                            />
                                            <Button
                                                variant="ghost"
                                                size="sm"
                                                onClick={handleTestConnection}
                                                loading={connectionTesting}
                                                title="Test connection"
                                            >
                                                <RefreshCw size={16} />
                                            </Button>
                                        </div>
                                    </div>

                                    <div className="form-row">
                                        <label htmlFor="cluster-secret">Cluster Secret</label>
                                        <Input
                                            id="cluster-secret"
                                            value={clusterSecret}
                                            onChange={(e) => setClusterSecret(e.target.value.toUpperCase())}
                                            placeholder="XXXX-XXXX-XXXX"
                                            maxLength={14}
                                        />
                                    </div>

                                    <Button
                                        variant="primary"
                                        leftIcon={<Wifi size={18} />}
                                        onClick={handleJoinCluster}
                                        loading={loading}
                                        disabled={!hostAddress || !clusterSecret}
                                        className="start-button"
                                    >
                                        Join Cluster
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    )}

                    {/* Connected Status - Host */}
                    {clusterStatus?.role === 'host' && (
                        <Card padding="lg" className="cluster-status-card">
                            <CardHeader
                                title="Cluster Status"
                                subtitle="You are hosting the training cluster"
                                action={
                                    <Button
                                        variant="danger"
                                        size="sm"
                                        leftIcon={<Square size={16} />}
                                        onClick={handleDisconnect}
                                        loading={loading}
                                    >
                                        Stop Hosting
                                    </Button>
                                }
                            />
                            <CardContent>
                                <div className="cluster-info">
                                    <div className="cluster-secret-display">
                                        <span className="secret-label">Cluster Secret:</span>
                                        <code className="secret-value">
                                            {clusterStatus.cluster_secret}
                                        </code>
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={handleCopySecret}
                                            title="Copy to clipboard"
                                        >
                                            {copied ? <Check size={16} /> : <Copy size={16} />}
                                        </Button>
                                    </div>

                                    <div className="cluster-stats">
                                        <div className="stat">
                                            <span className="stat-value">{clusterStatus.workers.length}</span>
                                            <span className="stat-label">Total Workers</span>
                                        </div>
                                        <div className="stat">
                                            <span className={`stat-value ${clusterStatus.is_ready ? 'ready' : ''}`}>
                                                {clusterStatus.is_ready ? 'Ready' : 'Waiting'}
                                            </span>
                                            <span className="stat-label">Status</span>
                                        </div>
                                        <div className="stat">
                                            <span className={`stat-value ${clusterStatus.is_training ? 'training' : ''}`}>
                                                {clusterStatus.is_training ? 'Yes' : 'No'}
                                            </span>
                                            <span className="stat-label">Training</span>
                                        </div>
                                    </div>

                                    {/* Worker List */}
                                    <div className="workers-section">
                                        <h4>Connected Workers</h4>
                                        <div className="workers-list">
                                            {clusterStatus.workers.map((worker) => (
                                                <div key={worker.worker_id} className="worker-item">
                                                    <div className="worker-info">
                                                        <span className={`status-dot ${worker.status}`}></span>
                                                        <span className="worker-hostname">{worker.hostname}</span>
                                                        <span className="worker-address">{worker.address}</span>
                                                        {worker.worker_id !== 'chief' && (
                                                            <span className="worker-task">Task {worker.task_index}</span>
                                                        )}
                                                        {worker.worker_id === 'chief' && (
                                                            <span className="worker-chief">Chief</span>
                                                        )}
                                                    </div>
                                                    <div className="worker-resources">
                                                        <span><Cpu size={14} /> {worker.cpu_count} cores</span>
                                                        <span><HardDrive size={14} /> {worker.memory_gb.toFixed(1)} GB</span>
                                                    </div>
                                                    {worker.worker_id !== 'chief' && (
                                                        <Button
                                                            variant="ghost"
                                                            size="sm"
                                                            onClick={() => handleRemoveWorker(worker.worker_id)}
                                                            title="Remove worker"
                                                        >
                                                            <Trash2 size={16} />
                                                        </Button>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    )}

                    {/* Connected Status - Worker */}
                    {clusterStatus?.role === 'worker' && (
                        <Card padding="lg" className="cluster-status-card worker-view">
                            <CardHeader
                                title="Connected to Cluster"
                                subtitle={`Host: ${clusterStatus.host_address}`}
                                action={
                                    <Button
                                        variant="danger"
                                        size="sm"
                                        leftIcon={<WifiOff size={16} />}
                                        onClick={handleDisconnect}
                                        loading={loading}
                                    >
                                        Disconnect
                                    </Button>
                                }
                            />
                            <CardContent>
                                <div className="worker-status">
                                    <div className="connection-indicator">
                                        <Wifi size={48} className="connected-icon" />
                                        <span>Connected to training cluster</span>
                                    </div>

                                    <div className="cluster-stats">
                                        <div className="stat">
                                            <span className={`stat-value ${clusterStatus.is_ready ? 'ready' : ''}`}>
                                                {clusterStatus.is_ready ? 'Ready' : 'Waiting'}
                                            </span>
                                            <span className="stat-label">Status</span>
                                        </div>
                                        <div className="stat">
                                            <span className={`stat-value ${clusterStatus.is_training ? 'training' : ''}`}>
                                                {clusterStatus.is_training ? 'Active' : 'Idle'}
                                            </span>
                                            <span className="stat-label">Training</span>
                                        </div>
                                    </div>

                                    {clusterStatus.tf_config && (
                                        <div className="tf-config-preview">
                                            <h5>TF_CONFIG</h5>
                                            <pre>{JSON.stringify(clusterStatus.tf_config, null, 2)}</pre>
                                        </div>
                                    )}
                                </div>
                            </CardContent>
                        </Card>
                    )}
                </div>

                {/* Attribution Settings Section */}
                <div className="settings-section">
                    <h2 className="section-title">
                        <SettingsIcon size={22} />
                        Attribution Settings
                        {attributionData?.edition && (
                            <span className={`edition-badge edition-${attributionData.edition.toLowerCase()}`}>
                                <Crown size={14} />
                                {attributionData.edition}
                            </span>
                        )}
                    </h2>

                    {attributionError && (
                        <div className="error-banner">
                            <Info size={18} />
                            {attributionError}
                            <button onClick={() => setAttributionError(null)}>&times;</button>
                        </div>
                    )}

                    {attributionSuccess && (
                        <div className="success-banner">
                            <Check size={18} />
                            Attribution updated successfully!
                        </div>
                    )}

                    <Card padding="lg" className="config-card">
                        <CardHeader
                            title="Custom Attribution"
                            subtitle={
                                attributionData?.is_customizable
                                    ? 'Customize the attribution displayed in your trained models'
                                    : 'Upgrade to Pro or Enterprise to customize attribution'
                            }
                        />
                        <CardContent>
                            <div className="config-form">
                                <div className="form-row">
                                    <label htmlFor="attr-framework-name">Framework Name</label>
                                    <Input
                                        id="attr-framework-name"
                                        value={attributionForm.framework_name}
                                        onChange={(e) => setAttributionForm(prev => ({
                                            ...prev,
                                            framework_name: e.target.value,
                                        }))}
                                        placeholder="My Framework powered by HSMN"
                                        disabled={!attributionData?.is_customizable}
                                    />
                                </div>

                                <div className="form-row">
                                    <label htmlFor="attr-author">Author / Company</label>
                                    <Input
                                        id="attr-author"
                                        value={attributionForm.author}
                                        onChange={(e) => setAttributionForm(prev => ({
                                            ...prev,
                                            author: e.target.value,
                                        }))}
                                        placeholder="Your Company Name"
                                        disabled={!attributionData?.is_customizable}
                                    />
                                </div>

                                <div className="form-row">
                                    <label htmlFor="attr-copyright">Copyright Notice</label>
                                    <Input
                                        id="attr-copyright"
                                        value={attributionForm.copyright_notice}
                                        onChange={(e) => setAttributionForm(prev => ({
                                            ...prev,
                                            copyright_notice: e.target.value,
                                        }))}
                                        placeholder="Copyright 2025 Your Company"
                                        disabled={!attributionData?.is_customizable}
                                    />
                                </div>

                                <div className="form-row two-column">
                                    <div>
                                        <label htmlFor="attr-version">Version</label>
                                        <Input
                                            id="attr-version"
                                            value={attributionForm.version}
                                            onChange={(e) => setAttributionForm(prev => ({
                                                ...prev,
                                                version: e.target.value,
                                            }))}
                                            placeholder="1.0.0"
                                            disabled={!attributionData?.is_customizable}
                                        />
                                    </div>
                                    <div>
                                        <label htmlFor="attr-support-url">Support URL</label>
                                        <Input
                                            id="attr-support-url"
                                            value={attributionForm.support_url}
                                            onChange={(e) => setAttributionForm(prev => ({
                                                ...prev,
                                                support_url: e.target.value,
                                            }))}
                                            placeholder="https://yourcompany.com/support"
                                            disabled={!attributionData?.is_customizable}
                                        />
                                    </div>
                                </div>

                                {attributionData?.is_customizable ? (
                                    <div className="button-row">
                                        <Button
                                            variant="primary"
                                            leftIcon={<Save size={18} />}
                                            onClick={handleSaveAttribution}
                                            loading={attributionLoading}
                                            disabled={!attributionForm.framework_name}
                                        >
                                            Save Changes
                                        </Button>
                                        {attributionData?.is_custom && (
                                            <Button
                                                variant="ghost"
                                                leftIcon={<RotateCcw size={18} />}
                                                onClick={handleResetAttribution}
                                                loading={attributionLoading}
                                            >
                                                Reset to Defaults
                                            </Button>
                                        )}
                                    </div>
                                ) : (
                                    <div className="upgrade-prompt">
                                        <div className="upgrade-message">
                                            <Crown size={24} />
                                            <div>
                                                <strong>Pro or Enterprise Edition Required</strong>
                                                <p>Upgrade to customize the attribution displayed in your trained models.</p>
                                            </div>
                                        </div>
                                        <a
                                            href="https://versoindustries.com/upgrade"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="upgrade-link"
                                        >
                                            <Button variant="primary" rightIcon={<ExternalLink size={16} />}>
                                                Upgrade Now
                                            </Button>
                                        </a>
                                    </div>
                                )}

                                <div className="attribution-info">
                                    <Info size={16} />
                                    <span>
                                        Changes apply to all models trained after saving.
                                        Existing models keep their original attribution.
                                    </span>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* HuggingFace Integration Section */}
                <div className="settings-section">
                    <h2 className="section-title">
                        <KeyRound size={22} />
                        HuggingFace Integration
                        {hfTokenConfigured && (
                            <span className="status-badge status-configured">
                                <Check size={14} />
                                Configured
                            </span>
                        )}
                    </h2>

                    {hfTokenError && (
                        <div className="error-banner">
                            <AlertTriangle size={18} />
                            {hfTokenError}
                            <button onClick={() => setHfTokenError(null)}>&times;</button>
                        </div>
                    )}

                    {hfTokenSuccess && (
                        <div className="success-banner">
                            <Check size={18} />
                            {hfUsername ? `Token validated! Logged in as ${hfUsername}` : 'Token saved successfully!'}
                        </div>
                    )}

                    <Card padding="lg" className="config-card">
                        <CardHeader
                            title="Access Token"
                            subtitle="Configure your HuggingFace token to access gated datasets like Llama, Mistral, and more"
                        />
                        <CardContent>
                            <div className="config-form">
                                {hfTokenConfigured ? (
                                    <div className="token-configured-display">
                                        <div className="token-status">
                                            <KeyRound size={20} className="token-icon" />
                                            <div className="token-info">
                                                <span className="token-label">Token configured</span>
                                                <code className="token-prefix">{hfTokenPrefix}</code>
                                                {hfUsername && (
                                                    <span className="token-user">Logged in as: {hfUsername}</span>
                                                )}
                                            </div>
                                        </div>
                                        <div className="button-row">
                                            <Button
                                                variant="ghost"
                                                leftIcon={<RefreshCw size={16} />}
                                                onClick={handleValidateHfToken}
                                                loading={hfTokenValidating}
                                            >
                                                Validate
                                            </Button>
                                            <Button
                                                variant="danger"
                                                leftIcon={<Trash2 size={16} />}
                                                onClick={handleClearHfToken}
                                                loading={hfTokenLoading}
                                            >
                                                Clear Token
                                            </Button>
                                        </div>
                                    </div>
                                ) : (
                                    <>
                                        <div className="form-row">
                                            <label htmlFor="hf-token">HuggingFace Token</label>
                                            <div className="input-with-button">
                                                <Input
                                                    id="hf-token"
                                                    type={showHfToken ? 'text' : 'password'}
                                                    value={hfToken}
                                                    onChange={(e) => setHfToken(e.target.value)}
                                                    placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
                                                />
                                                <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={() => setShowHfToken(!showHfToken)}
                                                    title={showHfToken ? 'Hide token' : 'Show token'}
                                                >
                                                    {showHfToken ? <EyeOff size={16} /> : <Eye size={16} />}
                                                </Button>
                                            </div>
                                        </div>

                                        <div className="button-row">
                                            <Button
                                                variant="primary"
                                                leftIcon={<Save size={18} />}
                                                onClick={handleSaveHfToken}
                                                loading={hfTokenLoading}
                                                disabled={!hfToken.trim()}
                                            >
                                                Save Token
                                            </Button>
                                            <Button
                                                variant="ghost"
                                                leftIcon={<RefreshCw size={18} />}
                                                onClick={handleValidateHfToken}
                                                loading={hfTokenValidating}
                                                disabled={!hfToken.trim()}
                                            >
                                                Validate
                                            </Button>
                                        </div>
                                    </>
                                )}

                                <div className="attribution-info">
                                    <Info size={16} />
                                    <span>
                                        Get your token from{' '}
                                        <a
                                            href="https://huggingface.co/settings/tokens"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                        >
                                            huggingface.co/settings/tokens
                                        </a>
                                        . Required for accessing gated datasets.
                                    </span>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Distributed HPO Section (Enterprise Phase 6) */}
                <div className="settings-section">
                    <h2 className="section-title">
                        <Cpu size={22} />
                        Distributed HPO (Ray Tune)
                        <span className="edition-badge edition-enterprise">
                            <Crown size={14} />
                            Enterprise
                        </span>
                    </h2>

                    <Card padding="lg" className="config-card">
                        <CardHeader
                            title="Ray Tune Cluster"
                            subtitle="Scale hyperparameter optimization across multiple workers"
                        />
                        <CardContent>
                            <div className="config-form">
                                <div className="distributed-hpo-status">
                                    <div className="status-indicator">
                                        <span className="status-dot offline"></span>
                                        <span>Cluster Status: Offline</span>
                                    </div>
                                    <div className="cluster-resources">
                                        <span><Cpu size={14} /> 0 CPUs</span>
                                        <span><HardDrive size={14} /> 0 GPUs</span>
                                    </div>
                                </div>

                                <div className="form-row">
                                    <label htmlFor="num-workers">Number of Workers</label>
                                    <Input
                                        id="num-workers"
                                        type="number"
                                        defaultValue={4}
                                        min={1}
                                        max={64}
                                    />
                                </div>

                                <div className="form-row">
                                    <label htmlFor="cpus-per-worker">CPUs per Worker</label>
                                    <Input
                                        id="cpus-per-worker"
                                        type="number"
                                        defaultValue={2}
                                        min={1}
                                        max={32}
                                    />
                                </div>

                                <div className="form-row">
                                    <label htmlFor="hpo-scheduler">HPO Scheduler</label>
                                    <select
                                        id="hpo-scheduler"
                                        className="select-input"
                                        defaultValue="asha"
                                    >
                                        <option value="asha">ASHA (Successive Halving)</option>
                                        <option value="pbt">Population-Based Training</option>
                                        <option value="qahpo">QAHPO (Quantum Adaptive)</option>
                                    </select>
                                </div>

                                <div className="form-row">
                                    <label htmlFor="max-concurrent">Max Concurrent Trials</label>
                                    <Input
                                        id="max-concurrent"
                                        type="number"
                                        defaultValue={8}
                                        min={1}
                                        max={100}
                                    />
                                </div>

                                <div className="form-row">
                                    <label htmlFor="ray-address">Ray Cluster Address</label>
                                    <Input
                                        id="ray-address"
                                        placeholder="auto (local) or ray://host:10001"
                                    />
                                </div>

                                <div className="button-row">
                                    <Button
                                        variant="primary"
                                        leftIcon={<Play size={18} />}
                                    >
                                        Start Cluster
                                    </Button>
                                    <Button
                                        variant="danger"
                                        leftIcon={<Square size={18} />}
                                        disabled
                                    >
                                        Stop Cluster
                                    </Button>
                                </div>

                                <div className="attribution-info">
                                    <Info size={16} />
                                    <span>
                                        Distributed HPO requires Ray Tune. Install with:{' '}
                                        <code>pip install ray[tune]</code>
                                    </span>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* AutoML Pipeline Section (Enterprise Phase 6.3) */}
                <div className="settings-section">
                    <h2 className="section-title">
                        <Zap size={22} />
                        AutoML Pipeline
                        <span className="edition-badge edition-enterprise">
                            <Crown size={14} />
                            Enterprise
                        </span>
                    </h2>

                    <Card padding="lg" className="config-card">
                        <CardHeader
                            title="End-to-End AutoML"
                            subtitle="Automated data profiling, model selection, HPO, and ensembling"
                        />
                        <CardContent>
                            <div className="config-form">
                                <div className="form-row">
                                    <label htmlFor="time-budget">Time Budget (seconds)</label>
                                    <Input
                                        id="time-budget"
                                        type="number"
                                        defaultValue={3600}
                                        min={60}
                                        max={86400}
                                    />
                                </div>

                                <div className="form-row">
                                    <label htmlFor="automl-max-trials">Max Trials</label>
                                    <Input
                                        id="automl-max-trials"
                                        type="number"
                                        defaultValue={100}
                                        min={10}
                                        max={1000}
                                    />
                                </div>

                                <div className="form-row">
                                    <label htmlFor="ensemble-size">Ensemble Size</label>
                                    <Input
                                        id="ensemble-size"
                                        type="number"
                                        defaultValue={5}
                                        min={1}
                                        max={20}
                                    />
                                </div>

                                <div className="button-row">
                                    <Button variant="primary" leftIcon={<Play size={18} />}>
                                        Start AutoML
                                    </Button>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Advanced Scheduler Section (Enterprise Phase 3) */}
                <div className="settings-section">
                    <h2 className="section-title">
                        <Activity size={22} />
                        Advanced HPO Scheduler
                        <span className="edition-badge edition-enterprise">
                            <Crown size={14} />
                            Enterprise
                        </span>
                    </h2>

                    <Card padding="lg" className="config-card">
                        <CardHeader
                            title="Multi-Fidelity & NAS"
                            subtitle="ASHA, MOBSTER, Freeze-Thaw, DARTS, and Zero-Cost Proxies"
                        />
                        <CardContent>
                            <div className="config-form">
                                <div className="form-row">
                                    <label htmlFor="scheduler-type">Scheduler Type</label>
                                    <select
                                        id="scheduler-type"
                                        className="select-input"
                                        defaultValue="asha"
                                    >
                                        <option value="asha">ASHA (Async Successive Halving)</option>
                                        <option value="mobster">MOBSTER (Model-Based ASHA)</option>
                                        <option value="freeze_thaw">Freeze-Thaw (Curve Prediction)</option>
                                        <option value="hyperband">Hyperband (Classic)</option>
                                        <option value="pbt_backtrack">PBT with Backtracking</option>
                                    </select>
                                </div>

                                <div className="form-row">
                                    <label htmlFor="fanova-type">Importance Analyzer</label>
                                    <select
                                        id="fanova-type"
                                        className="select-input"
                                        defaultValue="shapley"
                                    >
                                        <option value="shapley">Shapley Values</option>
                                        <option value="graph">Graph fANOVA</option>
                                        <option value="causal">Causal fANOVA</option>
                                        <option value="streaming">Streaming fANOVA</option>
                                    </select>
                                </div>

                                <div className="form-row">
                                    <label htmlFor="grace-period">Grace Period (epochs)</label>
                                    <Input
                                        id="grace-period"
                                        type="number"
                                        defaultValue={5}
                                        min={1}
                                        max={50}
                                    />
                                </div>

                                <div className="form-row">
                                    <label htmlFor="reduction-factor">Reduction Factor</label>
                                    <Input
                                        id="reduction-factor"
                                        type="number"
                                        defaultValue={3}
                                        min={2}
                                        max={5}
                                    />
                                </div>

                                <div className="attribution-info">
                                    <Info size={16} />
                                    <span>
                                        These settings apply to HPO sweeps started from the Training page.
                                    </span>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Quick Start Guide */}
                <Card padding="lg" className="guide-card">
                    <CardHeader
                        title="Quick Start Guide"
                        action={
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setShowGuide(!showGuide)}
                            >
                                {showGuide ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                            </Button>
                        }
                    />
                    {showGuide && (
                        <CardContent>
                            <div className="guide-content">
                                <div className="guide-section">
                                    <h4>1. Choose Your Role</h4>
                                    <p>
                                        <strong>Standalone:</strong> Train on a single machine (default).<br />
                                        <strong>Host:</strong> Start a cluster and coordinate multiple workers.<br />
                                        <strong>Worker:</strong> Join an existing cluster to contribute compute.
                                    </p>
                                </div>

                                <div className="guide-section">
                                    <h4>2. Setting Up a Cluster</h4>
                                    <ol>
                                        <li>On the main machine, select <strong>Host</strong> and click "Start Hosting"</li>
                                        <li>Copy the generated <strong>Cluster Secret</strong></li>
                                        <li>On other machines, select <strong>Worker</strong></li>
                                        <li>Enter the Host's IP address and the cluster secret</li>
                                        <li>Click "Join Cluster"</li>
                                    </ol>
                                </div>

                                <div className="guide-section">
                                    <h4>3. Firewall Configuration</h4>
                                    <p>
                                        Ensure port <code>12345</code> (or your chosen port) is open on the Host machine:
                                    </p>
                                    <code className="command">sudo ufw allow 12345/tcp</code>
                                </div>

                                <div className="guide-section">
                                    <h4>4. Shared Checkpoint Directory</h4>
                                    <p>
                                        All nodes need access to a shared filesystem (NFS, Lustre, etc.) for checkpoints.
                                        The path must be the same on all machines.
                                    </p>
                                </div>
                            </div>
                        </CardContent>
                    )}
                </Card>
            </div>
        </div>
    );
}
