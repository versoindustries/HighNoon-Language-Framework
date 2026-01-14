// useActivationStream.ts - WebSocket Hook for Live Activation Streaming
// Connects to /ws/activations/live/{job_id} for real-time activation data

import { useState, useEffect, useCallback, useRef } from 'react';
import type { SurfaceData } from './TensorSurfaceViz';

/** WebSocket message types from backend */
interface ActivationMessage {
    type: 'activation_update' | 'finished' | 'error';
    data?: {
        x: number[];
        y: number[];
        z: number[][];
        colorscale: string;
        layer_name: string;
        original_shape: [number, number];
        stats: {
            min: number;
            max: number;
            mean: number;
            std: number;
        };
        demo_mode: boolean;
        available_layers: string[];
        timestamp: number;
    };
    state?: string;
    message?: string;
}

/** State returned by the hook */
interface ActivationStreamState {
    /** Current activation surface data */
    data: SurfaceData | null;
    /** Available layers from backend */
    availableLayers: string[];
    /** Whether running in demo mode (no real activations) */
    isDemoMode: boolean;
    /** Whether WebSocket is connected */
    isConnected: boolean;
    /** Whether the job has finished */
    isFinished: boolean;
    /** Last error message */
    error: string | null;
    /** Timestamp of last update */
    lastUpdate: number | null;
}

/** Options for the activation stream hook */
interface UseActivationStreamOptions {
    /** Whether to enable the stream (default: true) */
    enabled?: boolean;
    /** Callback when data is received */
    onData?: (data: SurfaceData) => void;
    /** Callback when stream finishes */
    onFinished?: (state: string) => void;
    /** Callback on error */
    onError?: (error: string) => void;
}

/**
 * useActivationStream - React hook for live activation data streaming
 *
 * Connects to the backend WebSocket endpoint and receives real-time
 * activation surface data for visualization during training.
 *
 * @example
 * const { data, isConnected, availableLayers } = useActivationStream(jobId);
 * if (data) {
 *   return <TensorSurfaceViz data={data} />;
 * }
 */
export function useActivationStream(
    jobId: string | null,
    options: UseActivationStreamOptions = {}
): ActivationStreamState {
    const { enabled = true, onData, onFinished, onError } = options;

    const [state, setState] = useState<ActivationStreamState>({
        data: null,
        availableLayers: [],
        isDemoMode: true,
        isConnected: false,
        isFinished: false,
        error: null,
        lastUpdate: null,
    });

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<number | null>(null);
    const reconnectAttempts = useRef(0);
    const maxReconnectAttempts = 5;

    // Cleanup function
    const cleanup = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
    }, []);

    // Connect to WebSocket
    const connect = useCallback(() => {
        if (!jobId || !enabled) return;

        // Use the Vite proxy for WebSocket connections (handles both dev and prod)
        // This ensures consistent routing through the proxy which handles errors gracefully
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/activations/live/${jobId}`;

        try {
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                reconnectAttempts.current = 0;
                setState(prev => ({ ...prev, isConnected: true, error: null }));
            };

            ws.onmessage = (event) => {
                try {
                    const message: ActivationMessage = JSON.parse(event.data);

                    if (message.type === 'activation_update' && message.data) {
                        const surfaceData: SurfaceData = {
                            x: message.data.x,
                            y: message.data.y,
                            z: message.data.z,
                            colorscale: message.data.colorscale as 'viridis' | 'plasma' | 'inferno',
                            layer_name: message.data.layer_name,
                            original_shape: message.data.original_shape,
                            stats: message.data.stats,
                        };

                        setState(prev => ({
                            ...prev,
                            data: surfaceData,
                            availableLayers: message.data!.available_layers,
                            isDemoMode: message.data!.demo_mode,
                            lastUpdate: message.data!.timestamp,
                        }));

                        onData?.(surfaceData);
                    } else if (message.type === 'finished') {
                        setState(prev => ({ ...prev, isFinished: true }));
                        onFinished?.(message.state || 'unknown');
                        cleanup();
                    } else if (message.type === 'error') {
                        const errorMsg = message.message || 'Unknown error';
                        setState(prev => ({ ...prev, error: errorMsg }));
                        onError?.(errorMsg);
                    }
                } catch (parseError) {
                    console.error('[ActivationStream] Failed to parse message:', parseError);
                }
            };

            ws.onerror = (event) => {
                console.error('[ActivationStream] WebSocket error:', event);
                setState(prev => ({ ...prev, error: 'WebSocket connection error' }));
            };

            ws.onclose = (event) => {
                setState(prev => ({ ...prev, isConnected: false }));

                // Clean close or intentional disconnect - don't reconnect
                if (event.wasClean || event.code === 1000 || event.code === 1008) {
                    return;
                }

                // Attempt reconnect if not intentionally closed
                // Use longer delays for code 1006 (abnormal closure)
                if (reconnectAttempts.current < maxReconnectAttempts && enabled) {
                    reconnectAttempts.current += 1;
                    const isAbnormalClose = event.code === 1006;
                    const baseDelay = isAbnormalClose ? 2000 : 1000;  // Longer base for 1006
                    const delay = Math.min(baseDelay * Math.pow(2, reconnectAttempts.current), 15000);

                    console.debug(`[ActivationStream] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`);
                    reconnectTimeoutRef.current = window.setTimeout(() => {
                        connect();
                    }, delay);
                }
            };
        } catch (err) {
            console.error('[ActivationStream] Failed to create WebSocket:', err);
            setState(prev => ({
                ...prev,
                error: err instanceof Error ? err.message : 'Failed to connect'
            }));
        }
    }, [jobId, enabled, onData, onFinished, onError, cleanup]);

    // Effect to manage connection lifecycle
    useEffect(() => {
        if (jobId && enabled) {
            connect();
        }

        return cleanup;
    }, [jobId, enabled, connect, cleanup]);

    return state;
}

export default useActivationStream;
