// HighNoon Dashboard - WebSocket Client
// Real-time updates for training and HPO metrics

import type { WebSocketMessage } from './types';

export type WebSocketHandler = (message: WebSocketMessage) => void;

/**
 * Creates a managed WebSocket connection with automatic reconnection.
 */
export function createWebSocket(
    endpoint: string,
    onMessage: WebSocketHandler,
    onStatusChange?: (connected: boolean) => void
): { close: () => void } {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}${endpoint}`;

    let ws: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
    let isClosing = false;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const baseReconnectDelay = 1000;

    function connect() {
        if (isClosing) return;

        ws = new WebSocket(url);

        ws.onopen = () => {
            reconnectAttempts = 0;
            onStatusChange?.(true);
        };

        ws.onmessage = (event) => {
            try {
                const message: WebSocketMessage = JSON.parse(event.data);
                onMessage(message);
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        ws.onclose = () => {
            onStatusChange?.(false);

            if (!isClosing && reconnectAttempts < maxReconnectAttempts) {
                const delay = baseReconnectDelay * Math.pow(2, reconnectAttempts);
                reconnectAttempts++;
                reconnectTimeout = setTimeout(connect, delay);
            }
        };
    }

    connect();

    return {
        close: () => {
            isClosing = true;
            if (reconnectTimeout) {
                clearTimeout(reconnectTimeout);
            }
            if (ws) {
                ws.close();
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
