// useVideoRecorder.ts - MediaRecorder API Hook for Canvas Recording
// Captures Three.js canvas animations as WebM video

import { useRef, useState, useCallback } from 'react';

/** Video recording state */
interface RecordingState {
    isRecording: boolean;
    isPaused: boolean;
    duration: number;
    error: string | null;
}

/** Options for video recording */
interface VideoRecorderOptions {
    /** Target frames per second */
    fps?: number;
    /** Video bitrate in bps */
    videoBitsPerSecond?: number;
    /** MIME type for recording */
    mimeType?: string;
}

/** Return type for useVideoRecorder hook */
interface UseVideoRecorderReturn {
    state: RecordingState;
    startRecording: () => void;
    stopRecording: () => Promise<Blob | null>;
    pauseRecording: () => void;
    resumeRecording: () => void;
    downloadRecording: (filename?: string) => Promise<void>;
}

/**
 * useVideoRecorder - React hook for recording canvas animations
 *
 * Uses MediaRecorder API to capture canvas stream as WebM video.
 * Compatible with react-three-fiber canvases that have
 * preserveDrawingBuffer: true enabled.
 *
 * @example
 * const canvasRef = useRef<HTMLCanvasElement>(null);
 * const { state, startRecording, stopRecording, downloadRecording } = useVideoRecorder(canvasRef);
 */
export function useVideoRecorder(
    canvasRef: React.RefObject<HTMLCanvasElement | null>,
    options: VideoRecorderOptions = {}
): UseVideoRecorderReturn {
    const {
        fps = 30,
        videoBitsPerSecond = 5000000, // 5 Mbps
        mimeType = 'video/webm;codecs=vp9',
    } = options;

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const startTimeRef = useRef<number>(0);
    const durationIntervalRef = useRef<number | null>(null);

    const [state, setState] = useState<RecordingState>({
        isRecording: false,
        isPaused: false,
        duration: 0,
        error: null,
    });

    // Start recording
    const startRecording = useCallback(() => {
        if (!canvasRef.current) {
            setState(prev => ({ ...prev, error: 'Canvas not available' }));
            return;
        }

        try {
            // Get canvas stream
            const stream = canvasRef.current.captureStream(fps);

            // Check supported MIME types
            let actualMimeType = mimeType;
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                if (MediaRecorder.isTypeSupported('video/webm')) {
                    actualMimeType = 'video/webm';
                } else if (MediaRecorder.isTypeSupported('video/mp4')) {
                    actualMimeType = 'video/mp4';
                } else {
                    throw new Error('No supported video MIME type');
                }
            }

            // Create MediaRecorder
            const recorder = new MediaRecorder(stream, {
                mimeType: actualMimeType,
                videoBitsPerSecond,
            });

            // Handle data available
            recorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };

            // Handle errors
            recorder.onerror = (e) => {
                setState(prev => ({ ...prev, error: `Recording error: ${e}` }));
            };

            // Clear previous chunks
            chunksRef.current = [];

            // Start recording
            recorder.start(100); // Collect data every 100ms
            mediaRecorderRef.current = recorder;
            startTimeRef.current = Date.now();

            // Update duration every 100ms
            durationIntervalRef.current = window.setInterval(() => {
                setState(prev => ({
                    ...prev,
                    duration: Math.floor((Date.now() - startTimeRef.current) / 1000),
                }));
            }, 100);

            setState({
                isRecording: true,
                isPaused: false,
                duration: 0,
                error: null,
            });
        } catch (err) {
            setState(prev => ({
                ...prev,
                error: err instanceof Error ? err.message : 'Failed to start recording'
            }));
        }
    }, [canvasRef, fps, mimeType, videoBitsPerSecond]);

    // Stop recording
    const stopRecording = useCallback((): Promise<Blob | null> => {
        return new Promise((resolve) => {
            if (!mediaRecorderRef.current) {
                resolve(null);
                return;
            }

            const recorder = mediaRecorderRef.current;

            recorder.onstop = () => {
                // Clear duration interval
                if (durationIntervalRef.current) {
                    clearInterval(durationIntervalRef.current);
                    durationIntervalRef.current = null;
                }

                // Create blob from chunks
                const blob = new Blob(chunksRef.current, { type: recorder.mimeType });

                setState(prev => ({
                    ...prev,
                    isRecording: false,
                    isPaused: false,
                }));

                resolve(blob);
            };

            recorder.stop();
        });
    }, []);

    // Pause recording
    const pauseRecording = useCallback(() => {
        if (mediaRecorderRef.current?.state === 'recording') {
            mediaRecorderRef.current.pause();
            setState(prev => ({ ...prev, isPaused: true }));
        }
    }, []);

    // Resume recording
    const resumeRecording = useCallback(() => {
        if (mediaRecorderRef.current?.state === 'paused') {
            mediaRecorderRef.current.resume();
            setState(prev => ({ ...prev, isPaused: false }));
        }
    }, []);

    // Download recorded video
    const downloadRecording = useCallback(async (filename?: string) => {
        const blob = await stopRecording();
        if (!blob) return;

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename || `recording_${new Date().toISOString().slice(0, 19).replace(/[:-]/g, '')}.webm`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, [stopRecording]);

    return {
        state,
        startRecording,
        stopRecording,
        pauseRecording,
        resumeRecording,
        downloadRecording,
    };
}

export default useVideoRecorder;
