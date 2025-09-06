// Cosine similarity & normalization utilities for dense vectors.

export function l2norm(vec: Float32Array): number {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  return Math.sqrt(s);
}

export function normalizeInPlace(vec: Float32Array): void {
  const n = l2norm(vec);
  if (n > 0) {
    for (let i = 0; i < vec.length; i++) vec[i] = vec[i] / n as number;
  }
}

export function cosine(a: Float32Array, b: Float32Array): number {
  // Assumes (or benefits from) unit-normalized inputs.
  let s = 0;
  const len = a.length;
  for (let i = 0; i < len; i++) s += a[i] * b[i];
  return s;
}
