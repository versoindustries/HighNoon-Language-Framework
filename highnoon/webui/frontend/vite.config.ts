import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Suppress Node.js uncaught socket/websocket EPIPE errors
// These occur during normal operation when clients disconnect mid-stream
process.on('uncaughtException', (err: NodeJS.ErrnoException) => {
  const suppressedCodes = ['EPIPE', 'ECONNRESET', 'ECONNREFUSED', 'ETIMEDOUT'];
  if (err.code && suppressedCodes.includes(err.code)) {
    // Expected during HPO sweeps, page navigation, tab background - silently ignore
    return;
  }
  // Re-throw unexpected errors
  console.error('Uncaught exception:', err);
});

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Forward API requests to FastAPI backend
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      // Forward WebSocket connections
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
        // Handle proxy errors gracefully to prevent crashes
        // when backend closes WebSocket connections unexpectedly
        configure: (proxy) => {
          // Handle proxy-level errors - suppress expected connection issues
          proxy.on('error', (err) => {
            // Suppress common connection reset errors that occur during normal operation:
            // - EPIPE: Backend closes connection before proxy finishes writing
            // - ECONNRESET: Connection reset during handshake or client disconnect
            // - ECONNREFUSED: Backend not available (startup/shutdown)
            // - ETIMEDOUT: Connection timeout during high load
            // These are expected during sweep completion, trial switching, or page navigation
            const nodeErr = err as NodeJS.ErrnoException;
            const suppressedCodes = ['EPIPE', 'ECONNRESET', 'ECONNREFUSED', 'ETIMEDOUT'];
            if (nodeErr.code && suppressedCodes.includes(nodeErr.code)) {
              // Silently suppress - these are expected
              return;
            }
            if (err.message.includes('EPIPE') || err.message.includes('ECONNRESET')) {
              // Silently suppress - these are expected
              return;
            }
            console.warn('[Vite WS Proxy] Error:', err.message);
          });

          // Handle socket-level errors on proxy connections (client->proxy)
          proxy.on('proxyReqWs', (_proxyReq, _req, socket) => {
            socket.on('error', (err: NodeJS.ErrnoException) => {
              // Suppress socket errors from connection resets
              const suppressedCodes = ['ECONNRESET', 'EPIPE', 'ETIMEDOUT'];
              if (err.code && suppressedCodes.includes(err.code)) {
                // Silently suppress - these are expected
                return;
              }
              console.warn('[Vite WS Proxy] Socket error:', err.message);
            });
          });

          // Handle errors on the target connection (proxy->backend)
          proxy.on('proxyRes', (_proxyRes, _req, res) => {
            res.on('error', (err: NodeJS.ErrnoException) => {
              // Silently suppress EPIPE and ECONNRESET - expected during normal operation
              if (err.code === 'EPIPE' || err.code === 'ECONNRESET') {
                return;
              }
            });
          });
        },
      },
    },
  },
})
