export function float32ToBase64(vec: Float32Array): string {
  return Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength).toString("base64");
}

export function base64ToFloat32(b64: string): Float32Array {
  const buf = Buffer.from(b64, "base64");
  // Create a Float32Array view of the underlying bytes
  return new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
}
