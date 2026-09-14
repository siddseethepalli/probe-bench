// Diverging scale for probe scores: blue below zero, paper at zero, orange above,
// symmetric and clipped at |z| = Z_CLIP. Backgrounds keep enough transparency that
// text on top stays readable.

export const Z_CLIP = 3;

const POSITIVE = "217, 101, 31";
const NEGATIVE = "42, 93, 176";
const MAX_ALPHA = 0.62;

export function clampZ(z: number): number {
  if (!Number.isFinite(z)) {
    return 0;
  }
  return Math.max(-Z_CLIP, Math.min(Z_CLIP, z));
}

export function zToBackground(z: number): string {
  const clipped = clampZ(z);
  const t = Math.abs(clipped) / Z_CLIP;
  if (t < 0.02) {
    return "transparent";
  }
  const rgb = clipped > 0 ? POSITIVE : NEGATIVE;
  return `rgba(${rgb}, ${(t * MAX_ALPHA).toFixed(3)})`;
}

export function zToSolid(z: number): string {
  return clampZ(z) >= 0 ? `rgb(${POSITIVE})` : `rgb(${NEGATIVE})`;
}

// Fixed-point text without a sign prefix; values that round to zero print as plain zero.
export function formatPlain(value: number, digits: number): string {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  const text = value.toFixed(digits);
  return Number(text) === 0 ? (0).toFixed(digits) : text;
}

export function formatZ(z: number, digits = 1): string {
  const text = formatPlain(z, digits);
  return z > 0 && Number(text) !== 0 ? `+${text}` : text;
}
