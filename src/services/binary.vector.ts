// src/services/binary.vector.ts
import { fp16ToFloat32Array, float32ToFp16Array } from "../utils/fpI6";

/**
 * Vector blob formats
 *
 * v0 (legacy, 13 bytes):
 *   0..3  : MAGIC "VECB" (big-endian)
 *   4..7  : dim (u32, little-endian)
 *   8..11 : count (u32, little-endian)
 *   12    : quant (u8) 0=fp32, 1=fp16
 *   13..  : payload
 *
 * v1 (current, 16 bytes):
 *   0..3  : MAGIC "VECB" (big-endian)
 *   4     : version u8 = 1
 *   5     : dtype  u8 = 1(fp32), 2(fp16)
 *   6..7  : pad u16 = 0
 *   8..11 : dim (u32, little-endian)
 *   12..15: count (u32, little-endian)
 *   16..  : payload
 */

const MAGIC = 0x56454342; // "VECB" big-endian

// normalized quant code used by this module's API
export type QuantCode = 0 | 1; // 0=fp32, 1=fp16

export interface VectorBlobHeader {
  dim: number;
  count: number;
  quant: QuantCode; // normalized (0=fp32,1=fp16)
  version?: number; // 1 for v1, undefined for v0
}

type EncodeFormat = "v0" | "v1";

/** Encode vectors into a blob. Default format is v1. */
export function encodeVectorsBlob(
  vectors: Float32Array[],
  quant: QuantCode,
  format: EncodeFormat = "v1"
): Uint8Array {
  if (vectors.length === 0) throw new Error("No vectors to encode");
  const dim = vectors[0].length;
  for (const v of vectors) if (v.length !== dim) throw new Error("Dimension mismatch");

  // build contiguous payload
  const count = vectors.length;
  let payload: Uint8Array;
  if (quant === 0) {
    const buf = new ArrayBuffer(dim * count * 4);
    const dst = new Float32Array(buf);
    let off = 0;
    for (const v of vectors) { dst.set(v, off); off += dim; }
    payload = new Uint8Array(buf);
  } else {
    const buf = new ArrayBuffer(dim * count * 2);
    const dst = new Uint16Array(buf);
    let off = 0;
    for (const v of vectors) {
      const half = float32ToFp16Array(v);
      dst.set(half, off);
      off += dim;
    }
    payload = new Uint8Array(buf);
  }

  if (format === "v1") {
    // v1 header (16 bytes): magic, version=1, dtype(1/2), pad, dim, count
    const out = new Uint8Array(16 + payload.byteLength);
    const dv = new DataView(out.buffer);
    dv.setUint32(0, MAGIC, false); // BE
    dv.setUint8(4, 1);             // version = 1
    dv.setUint8(5, quant === 0 ? 1 : 2); // dtype: 1=fp32, 2=fp16
    dv.setUint16(6, 0, true);      // pad
    dv.setUint32(8, dim, true);    // dim LE
    dv.setUint32(12, count, true); // count LE
    out.set(payload, 16);
    return out;
  } else {
    // v0 header (13 bytes): magic, dim, count, quant(0/1)
    const out = new Uint8Array(13 + payload.byteLength);
    const dv = new DataView(out.buffer);
    dv.setUint32(0, MAGIC, false); // BE
    dv.setUint32(4, dim, true);
    dv.setUint32(8, count, true);
    dv.setUint8(12, quant);        // 0=fp32, 1=fp16
    out.set(payload, 13);
    return out;
  }
}

/** Decode vectors blob; supports both v0 and v1 headers. */
export function decodeVectorsBlob(
  buf: ArrayBufferLike | Uint8Array
): { header: VectorBlobHeader; vectors: Float32Array[] } {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  const magic = dv.getUint32(0, false);
  if (magic !== MAGIC) throw new Error("Bad vectors blob magic");

  // ---- Robust v1 detection ----
  // v1 layout:
  //  [4]=version (1..10 ok), [5]=dtype (1=fp32,2=fp16), [6..7]=pad=0, len>=16
  const v1Version = dv.getUint8(4);
  const v1Dtype = dv.getUint8(5);
  const v1Pad = dv.getUint16(6, true);
  const looksV1 =
    bytes.byteLength >= 16 &&
    v1Version >= 1 &&
    v1Version <= 10 &&
    (v1Dtype === 1 || v1Dtype === 2) &&
    v1Pad === 0;

  let dim: number, count: number, quant: QuantCode, offset: number;

  if (looksV1) {
    // v1 header
    dim = dv.getUint32(8, true);
    count = dv.getUint32(12, true);
    offset = 16;
    quant = v1Dtype === 1 ? 0 : 1; // 1=fp32->0, 2=fp16->1
  } else {
    // v0 header
    dim = dv.getUint32(4, true);
    count = dv.getUint32(8, true);
    const q = dv.getUint8(12); // expected 0(fp32) or 1(fp16)
    offset = 13;

    // Be lenient: some writers incorrectly used 2 for fp16 in v0.
    if (q === 0) quant = 0;
    else if (q === 1 || q === 2) quant = 1;
    else throw new Error(`Unknown quant code in v0 header: ${q}`);
  }

  const vectors: Float32Array[] = [];
  const payload = new Uint8Array(bytes.buffer, bytes.byteOffset + offset, bytes.byteLength - offset);

  if (quant === 0) {
    // fp32 payload
    const aligned = payload.byteOffset % 4 === 0 ? payload : payload.slice();
    const view = new Float32Array(aligned.buffer, aligned.byteOffset, Math.floor(aligned.byteLength / 4));
    for (let i = 0; i < count; i++) {
      const slice = view.subarray(i * dim, (i + 1) * dim);
      vectors.push(new Float32Array(slice));
    }
  } else {
    // fp16 payload
    const aligned = payload.byteOffset % 2 === 0 ? payload : payload.slice();
    const view16 = new Uint16Array(aligned.buffer, aligned.byteOffset, Math.floor(aligned.byteLength / 2));
    for (let i = 0; i < count; i++) {
      const slice = view16.subarray(i * dim, (i + 1) * dim);
      vectors.push(fp16ToFloat32Array(slice));
    }
  }

  return { header: { dim, count, quant, version: looksV1 ? v1Version : undefined }, vectors };
}
