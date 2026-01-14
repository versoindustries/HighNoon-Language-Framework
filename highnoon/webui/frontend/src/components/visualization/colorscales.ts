// Colorscale utilities for 3D tensor visualization
// Returns RGB tuples in [0, 1] range for Three.js

/**
 * Viridis colorscale implementation.
 * Returns [r, g, b] in range [0, 1] for Three.js
 */
export function viridisColor(t: number): [number, number, number] {
    // Clamp to [0, 1]
    t = Math.max(0, Math.min(1, t));

    // Viridis color stops (approximate)
    const colors = [
        [0.267, 0.005, 0.329], // 0.0 - dark purple
        [0.283, 0.141, 0.458], // 0.2
        [0.231, 0.322, 0.545], // 0.4 - blue
        [0.129, 0.566, 0.551], // 0.6 - teal
        [0.477, 0.821, 0.318], // 0.8 - green
        [0.993, 0.906, 0.144], // 1.0 - yellow
    ];

    const idx = t * (colors.length - 1);
    const i0 = Math.floor(idx);
    const i1 = Math.min(i0 + 1, colors.length - 1);
    const frac = idx - i0;

    return [
        colors[i0][0] + frac * (colors[i1][0] - colors[i0][0]),
        colors[i0][1] + frac * (colors[i1][1] - colors[i0][1]),
        colors[i0][2] + frac * (colors[i1][2] - colors[i0][2]),
    ];
}

/**
 * Plasma colorscale implementation.
 */
export function plasmaColor(t: number): [number, number, number] {
    t = Math.max(0, Math.min(1, t));

    const colors = [
        [0.050, 0.030, 0.527], // dark blue
        [0.416, 0.090, 0.624], // purple
        [0.694, 0.165, 0.564], // magenta
        [0.929, 0.361, 0.388], // salmon
        [0.992, 0.665, 0.251], // orange
        [0.940, 0.975, 0.131], // yellow
    ];

    const idx = t * (colors.length - 1);
    const i0 = Math.floor(idx);
    const i1 = Math.min(i0 + 1, colors.length - 1);
    const frac = idx - i0;

    return [
        colors[i0][0] + frac * (colors[i1][0] - colors[i0][0]),
        colors[i0][1] + frac * (colors[i1][1] - colors[i0][1]),
        colors[i0][2] + frac * (colors[i1][2] - colors[i0][2]),
    ];
}

/**
 * Inferno colorscale implementation.
 */
export function infernoColor(t: number): [number, number, number] {
    t = Math.max(0, Math.min(1, t));

    const colors = [
        [0.001, 0.000, 0.014], // black
        [0.232, 0.059, 0.437], // dark purple
        [0.552, 0.114, 0.423], // magenta
        [0.831, 0.283, 0.255], // red-orange
        [0.980, 0.598, 0.157], // orange
        [0.988, 0.998, 0.645], // light yellow
    ];

    const idx = t * (colors.length - 1);
    const i0 = Math.floor(idx);
    const i1 = Math.min(i0 + 1, colors.length - 1);
    const frac = idx - i0;

    return [
        colors[i0][0] + frac * (colors[i1][0] - colors[i0][0]),
        colors[i0][1] + frac * (colors[i1][1] - colors[i0][1]),
        colors[i0][2] + frac * (colors[i1][2] - colors[i0][2]),
    ];
}

export type ColorscaleFunction = (t: number) => [number, number, number];

export const colorscales: Record<string, ColorscaleFunction> = {
    viridis: viridisColor,
    plasma: plasmaColor,
    inferno: infernoColor,
};

export type ColorscaleName = keyof typeof colorscales;
