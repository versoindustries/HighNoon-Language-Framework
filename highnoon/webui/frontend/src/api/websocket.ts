// HighNoon Dashboard - WebSocket Client
// Real-time updates for training and HPO metrics
// With visibility-aware reconnection to handle background tabs

import type { WebSocketMessage } from './types';

export type WebSocketHandler = (message: WebSocketMessage) => void;

/**
 * WebSocket connection options.
 */
interface WebSocketOptions {
    /** Maximum reconnection attempts when tab is hidden. Default: 3 */
    maxHiddenReconnects?: number;
    /** Base delay between reconnection attempts in ms. Default: 1000 */
    baseReconnectDelay?: number;
    /** Maximum delay between reconnection attempts in ms. Default: 15000 */
    maxReconnectDelay?: number;
    /** Interval for client-side ping messages in ms. Default: 25000 */
    pingInterval?: number;
}

/**
 * Creates a managed WebSocket connection with visibility-aware reconnection.
 *
 * Features:
 * - Detects when tab goes hidden/visible using visibilitychange API
 * - Pauses reconnection when tab is hidden (browser will throttle anyway)
 * - Immediately reconnects when tab becomes visible
 * - Unlimited reconnection attempts when tab is visible
 * - Client-side ping to keep connection alive
 *
 * @param endpoint - WebSocket endpoint path (e.g., '/ws/hpo/abc123')
 * @param onMessage - Callback for incoming messages
 * @param onStatusChange - Optional callback for connection status changes
 * @param options - Optional configuration
 */
export function createWebSocket(
    endpoint: string,
    onMessage: WebSocketHandler,
    onStatusChange?: (connected: boolean) => void,
    options?: WebSocketOptions
): { close: () => void } {
    const {
        maxHiddenReconnects = 3,
        baseReconnectDelay = 1000,
        maxReconnectDelay = 15000,
        pingInterval = 25000,
    } = options ?? {};

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}${endpoint}`;

    let ws: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
    let heartbeatInterval: ReturnType<typeof setInterval> | null = null;
    let isClosing = false;
    let reconnectAttempts = 0;
    let isTabHidden = document.hidden;

    /**
     * Start client-side heartbeat to keep connection alive.
     * Sends a ping message every pingInterval ms.
     */
    function startHeartbeat() {
        stopHeartbeat();
        heartbeatInterval = setInterval(() => {
            if (ws?.readyState === WebSocket.OPEN) {
                try {
                    // Send a simple ping message - server ignores but keeps connection alive
                    ws.send(JSON.stringify({ type: 'ping', ts: Date.now() }));
                } catch {
                    // Ignore send errors - onclose will handle reconnection
                }
            }
        }, pingInterval);
    }

    /**
     * Stop the client-side heartbeat.
     */
    function stopHeartbeat() {
        if (heartbeatInterval) {
            clearInterval(heartbeatInterval);
            heartbeatInterval = null;
        }
    }

    /**
     * Connect to the WebSocket server.
     */
    function connect() {
        if (isClosing) return;

        // Clear any pending reconnect
        if (reconnectTimeout) {
            clearTimeout(reconnectTimeout);
            reconnectTimeout = null;
        }

        console.debug(`[WS] Connecting to ${endpoint}...`);
        ws = new WebSocket(url);

        ws.onopen = () => {
            console.debug('[WS] Connected');
            reconnectAttempts = 0;
            startHeartbeat();
            onStatusChange?.(true);
        };

        ws.onmessage = (event) => {
            try {
                const message: WebSocketMessage = JSON.parse(event.data);
                // Handle pong response (if server echoes back)
                if (message.type === 'pong') return;
                onMessage(message);
            } catch (e) {
                console.error('[WS] Failed to parse message:', e);
            }
        };

        ws.onerror = (error) => {
            console.debug('[WS] Error:', error);
        };

        ws.onclose = (event) => {
            stopHeartbeat();
            ws = null;
            onStatusChange?.(false);

            // Don't reconnect if intentionally closing
            if (isClosing) return;

            // Don't reconnect for clean close or resource not found
            if (event.code === 1000 || event.code === 1008) {
                console.debug(`[WS] Clean close (code: ${event.code}), not reconnecting`);
                return;
            }

            // Handle reconnection based on tab visibility
            if (isTabHidden) {
                // Tab is hidden - limited reconnection attempts
                if (reconnectAttempts < maxHiddenReconnects) {
                    scheduleReconnect();
                } else {
                    console.debug(`[WS] Tab hidden and max attempts (${maxHiddenReconnects}) reached, waiting for tab focus`);
                }
            } else {
                // Tab is visible - always try to reconnect
                scheduleReconnect();
            }
        };
    }

    /**
     * Schedule a reconnection attempt with exponential backoff.
     */
    function scheduleReconnect() {
        if (isClosing || reconnectTimeout) return;

        // Exponential backoff with jitter
        const exponentialDelay = baseReconnectDelay * Math.pow(2, Math.min(reconnectAttempts, 5));
        const jitter = Math.random() * 0.3 * exponentialDelay;
        const delay = Math.min(exponentialDelay + jitter, maxReconnectDelay);

        reconnectAttempts++;
        console.debug(`[WS] Reconnecting in ${Math.round(delay)}ms (attempt ${reconnectAttempts})`);

        reconnectTimeout = setTimeout(() => {
            reconnectTimeout = null;
            connect();
        }, delay);
    }

    /**
     * Handle visibility change events.
     * Immediately reconnects when tab becomes visible.
     */
    function handleVisibilityChange() {
        const wasHidden = isTabHidden;
        isTabHidden = document.hidden;

        if (wasHidden && !isTabHidden) {
            // Tab just became visible
            console.debug('[WS] Tab visible, checking connection...');

            if (!ws || ws.readyState !== WebSocket.OPEN) {
                // Not connected - reconnect immediately
                console.debug('[WS] Reconnecting after tab focus');
                reconnectAttempts = 0; // Reset attempts on visibility
                connect();
            }
        } else if (!wasHidden && isTabHidden) {
            console.debug('[WS] Tab hidden, connection will be limited');
        }
    }

    // Register visibility change listener
    document.addEventListener('visibilitychange', handleVisibilityChange);

    // Initial connection
    connect();

    return {
        close: () => {
            isClosing = true;
            document.removeEventListener('visibilitychange', handleVisibilityChange);
            stopHeartbeat();
            if (reconnectTimeout) {
                clearTimeout(reconnectTimeout);
                reconnectTimeout = null;
            }
            if (ws) {
                ws.close(1000, 'Client closing');
                ws = null;
            }
        },
    };
}

/**
 * Creates a training metrics WebSocket connection.
 */
export function createTrainingWebSocket(
    jobId: string,
    onMessage: WebSocketHandler,
    onStatusChange?: (connected: boolean) => void
) {
    return createWebSocket(`/ws/training/${jobId}`, onMessage, onStatusChange);
}

/**
 * Creates an HPO sweep WebSocket connection.
 */
export function createHPOWebSocket(
    sweepId: string,
    onMessage: WebSocketHandler,
    onStatusChange?: (connected: boolean) => void
) {
    return createWebSocket(`/ws/hpo/${sweepId}`, onMessage, onStatusChange);
}
