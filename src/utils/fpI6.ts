// Minimal fp16 <-> fp32 conversion for packing vectors efficiently.
// Source: standard IEEE754 half-precision conversion.

export function fp16ToFloat32Array(src: Uint16Array): Float32Array {
  const dst = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const h = src[i];
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7C00) >> 10;
    const f = h & 0x03FF;

    let val: number;
    if (e === 0) {
      // subnormal
      val = (f / 1024) * Math.pow(2, -14);
    } else if (e === 0x1F) {
      // Inf/NaN
      val = f ? NaN : Infinity;
    } else {
      val = (1 + f / 1024) * Math.pow(2, e - 15);
    }
    dst[i] = (s ? -1 : 1) * val;
  }
  return dst;
}

export function float32ToFp16Array(src: Float32Array): Uint16Array {
  // Simple round-to-nearest-even; not perfect but fine for storage
  const dst = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) {
    let x = src[i];
    const sign = x < 0 ? 1 : 0;
    x = Math.abs(x);
    if (isNaN(x)) { dst[i] = (sign << 15) | 0x7E00; continue; }
    if (x === Infinity) { dst[i] = (sign << 15) | 0x7C00; continue; }
    if (x === 0) { dst[i] = sign << 15; continue; }

    let e = Math.floor(Math.log2(x));
    let m = x / Math.pow(2, e) - 1;
    e += 15;
    if (e <= 0) {
      // subnormal
      m = x / Math.pow(2, -14);
      dst[i] = (sign << 15) | Math.max(0, Math.round(m));
    } else if (e >= 31) {
      dst[i] = (sign << 15) | 0x7C00;
    } else {
      const frac = Math.round(m * 1024);
      dst[i] = (sign << 15) | (e << 10) | (frac & 0x03FF);
    }
  }
  return dst;
}
