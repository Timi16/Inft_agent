// Simple binary format for large vector sets.
// Layout:
//  - 4 bytes: magic "VECB"
//  - 4 bytes: uint32 dim
//  - 4 bytes: uint32 count
//  - 1 byte : quant code (0=fp32,1=fp16)
//  - payload: vectors row-major
//
// No per-row metadata here; the text/meta live in manifest.entries
// in the same order as the vectors in the blob.

import { fp16ToFloat32Array, float32ToFp16Array } from "../utils/fpI6";

const MAGIC = 0x56454342; // "VECB" big-endian

export type QuantCode = 0 | 1; // 0=fp32, 1=fp16

export interface VectorBlobHeader {
  dim: number;
  count: number;
  quant: QuantCode;
}

export function encodeVectorsBlob(vectors: Float32Array[], quant: QuantCode): Uint8Array {
  if (vectors.length === 0) throw new Error("No vectors to encode");
  const dim = vectors[0].length;
  for (const v of vectors) if (v.length !== dim) throw new Error("Dimension mismatch");

  const headerSize = 4 + 4 + 4 + 1;
  let payload: Uint8Array;

  if (quant === 0) {
    // fp32
    const buf = new ArrayBuffer(dim * vectors.length * 4);
    const dst = new Float32Array(buf);
    let off = 0;
    for (const v of vectors) { dst.set(v, off); off += dim; }
    payload = new Uint8Array(buf);
  } else {
    // fp16
    const buf = new ArrayBuffer(dim * vectors.length * 2);
    const dst = new Uint16Array(buf);
    let off = 0;
    for (const v of vectors) {
      const half = float32ToFp16Array(v);
      dst.set(half, off);
      off += dim;
    }
    payload = new Uint8Array(buf);
  }

  const out = new Uint8Array(headerSize + payload.byteLength);
  const dv = new DataView(out.buffer);

  dv.setUint32(0, MAGIC, false); // big-endian
  dv.setUint32(4, dim, true);    // little-endian for numbers
  dv.setUint32(8, vectors.length, true);
  dv.setUint8(12, quant);

  out.set(payload, headerSize);
  return out;
}

export function decodeVectorsBlob(buf: ArrayBufferLike): { header: VectorBlobHeader; vectors: Float32Array[] } {
  const dv = new DataView(buf);

  const magic = dv.getUint32(0, false);
  if (magic !== MAGIC) throw new Error("Bad vectors blob magic");

  const dim = dv.getUint32(4, true);
  const count = dv.getUint32(8, true);
  const quant = dv.getUint8(12) as QuantCode;

  const payload = new Uint8Array(buf, 13);
  const vectors: Float32Array[] = [];

  if (quant === 0) {
    // fp32
    const view = new Float32Array(payload.buffer, payload.byteOffset, Math.floor(payload.byteLength / 4));
    for (let i = 0; i < count; i++) {
      const slice = view.subarray(i * dim, (i + 1) * dim);
      // copy to standalone array (to be safe if buffer is freed)
      vectors.push(new Float32Array(slice));
    }
  } else if (quant === 1) {
    // fp16 -> fp32
    const view = new Uint16Array(payload.buffer, payload.byteOffset, Math.floor(payload.byteLength / 2));
    for (let i = 0; i < count; i++) {
      const slice = view.subarray(i * dim, (i + 1) * dim);
      vectors.push(fp16ToFloat32Array(slice));
    }
  } else {
    throw new Error(`Unknown quant code: ${quant}`);
  }

  return { header: { dim, count, quant }, vectors };
}
