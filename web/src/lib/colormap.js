import * as THREE from "three";

// Viridis anchor stops (matplotlib), sampled at 0.0..1.0. Perceptually uniform.
const VIRIDIS = [
  [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
  [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
  [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
  [0.741, 0.873, 0.150], [0.993, 0.906, 0.144],
];

// Return [r,g,b] in 0..1 for t in 0..1. Robust to NaN / out-of-range input
// (a non-finite t used to make VIRIDIS[NaN] undefined and crash on a[0]).
export function viridis(t) {
  t = Number.isFinite(t) ? t : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const last = VIRIDIS.length - 1;
  const x = t * last;
  let i = Math.floor(x);
  if (i < 0) i = 0; else if (i > last) i = last;
  const f = x - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[i < last ? i + 1 : last];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export function viridisColor(t) {
  const [r, g, b] = viridis(t);
  return new THREE.Color(r, g, b);
}

// CSS gradient string for legends.
export function viridisCss(stops = 12) {
  const parts = [];
  for (let i = 0; i < stops; i++) {
    const t = i / (stops - 1);
    const [r, g, b] = viridis(t);
    parts.push(`rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0}) ${(t * 100).toFixed(0)}%`);
  }
  return `linear-gradient(to top, ${parts.join(",")})`;
}
