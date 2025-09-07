import { fp16ToFloat32Array, float32ToFp16Array } from "../utils/fpI6";

const MAGIC = 0x56454342; // "VECB" big-endian
export type QuantCode = 0 | 1;

export interface VectorBlobHeader {
  dim: number;
  count: number;
  quant: QuantCode;
}

export function encodeVectorsBlob(vectors: Float32Array[], quant: QuantCode): Uint8Array {
  if (vectors.length === 0) throw new Error("No vectors to encode");
  const dim = vectors[0].length;
  for (const v of vectors) if (v.length !== dim) throw new Error("Dimension mismatch");

  const headerSize = 4 + 4 + 4 + 1; // 13 bytes
  let payload: Uint8Array;

  if (quant === 0) {
    const buf = new ArrayBuffer(dim * vectors.length * 4);
    const dst = new Float32Array(buf);
    let off = 0;
    for (const v of vectors) { dst.set(v, off); off += dim; }
    payload = new Uint8Array(buf);
  } else {
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
  dv.setUint32(0, MAGIC, false); // BE
  dv.setUint32(4, dim, true);    // LE
  dv.setUint32(8, vectors.length, true);
  dv.setUint8(12, quant);

  out.set(payload, headerSize);
  return out;
}

export function decodeVectorsBlob(buf: ArrayBufferLike | Uint8Array): {
  header: VectorBlobHeader; vectors: Float32Array[];
} {
  // Work with a Uint8Array view so we preserve byteOffset/byteLength.
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  const magic = dv.getUint32(0, false);
  if (magic !== MAGIC) throw new Error("Bad vectors blob magic");

  const dim = dv.getUint32(4, true);
  const count = dv.getUint32(8, true);
  const quant = dv.getUint8(12) as QuantCode;

  // Payload starts right after the 13-byte header
  const payload = new Uint8Array(bytes.buffer, bytes.byteOffset + 13, bytes.byteLength - 13);
  const vectors: Float32Array[] = [];

  if (quant === 0) {
    // Align to 4 bytes if needed by copying (slice() creates offset=0)
    const p = (payload.byteOffset % 4 === 0) ? payload : payload.slice();
    const view = new Float32Array(p.buffer, p.byteOffset, Math.floor(p.byteLength / 4));
    for (let i = 0; i < count; i++) {
      const slice = view.subarray(i * dim, (i + 1) * dim);
      vectors.push(new Float32Array(slice));
    }
  } else if (quant === 1) {
    // Align to 2 bytes for Uint16
    const p = (payload.byteOffset % 2 === 0) ? payload : payload.slice();
    const view16 = new Uint16Array(p.buffer, p.byteOffset, Math.floor(p.byteLength / 2));
    for (let i = 0; i < count; i++) {
      const slice = view16.subarray(i * dim, (i + 1) * dim);
      vectors.push(fp16ToFloat32Array(slice));
    }
  } else {
    throw new Error(`Unknown quant code: ${quant}`);
  }

  return { header: { dim, count, quant }, vectors };
}