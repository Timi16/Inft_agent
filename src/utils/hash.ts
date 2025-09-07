import { createHash } from "node:crypto";

export function sha256Hex(bytes: Uint8Array | ArrayBufferLike): string {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes as ArrayBufferLike);
  const h = createHash("sha256");
  h.update(buf);
  return "sha256:" + h.digest("hex");
}
