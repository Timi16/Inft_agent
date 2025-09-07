// src/utils/uri.ts
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { isAbsolute, resolve as resolvePath } from "node:path";

const dequote = (s?: string | null) => (s ?? "").trim().replace(/^[\'\"]+|[\'\"]+$/g, "");

export const isHttpLike = (u: string) => /^https?:\/\//i.test(u);
export const isIpfsLike = (u: string) => /^ipfs:\/\//i.test(u);
export const isFileLike = (u: string) => /^file:/i.test(u);
export const isWinAbs = (p: string) => /^[a-zA-Z]:[\\/]/.test(p);

/** ipfs://… → http(s) gateway url */
export function ipfsToHttp(ipfsUri: string, gateway?: string) {
  const gw = (gateway || process.env.PINATA_GATEWAY || process.env.IPFS_GATEWAY || "https://ipfs.io/ipfs/").replace(/\/+$/, "");
  const path = ipfsUri.replace(/^ipfs:\/\//i, "");
  return `${gw}/${path}`;
}

/** Normalize anything "file-ish" into a real local path */
export function toLocalPath(input: string): string {
  let s = dequote(input);

  // fix invalid variants like "file:\C:\..." or "file:/C:/..."
  if (/^file:\\/i.test(s) || /^file:\/(?!\/)/i.test(s)) {
    s = s.replace(/^file:\\/i, "file:///").replace(/^file:\//i, "file:///");
  }

  if (isFileLike(s)) {
    // Proper file URL → convert via WHATWG URL
    try {
      return fileURLToPath(new URL(s));
    } catch {
      // fallback: strip file: and leading slashes/backslashes
      let p = s.replace(/^file:/i, "").replace(/^[\/\\]+/, "");
      return p;
    }
  }

  // plain path
  if (isWinAbs(s) || isAbsolute(s)) return s;
  // relative path → resolve against CWD
  return resolvePath(process.cwd(), s);
}

/** Fetch bytes from ipfs/http/file/local path safely */
export async function fetchBytesSmart(uriOrPath: string, gateway?: string): Promise<Uint8Array> {
  const src = dequote(uriOrPath);

  if (isIpfsLike(src)) {
    const url = ipfsToHttp(src, gateway);
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetchBytesSmart: ${r.status} ${r.statusText} for ${url}`);
    const ab = await r.arrayBuffer();
    return new Uint8Array(ab);
  }

  if (isHttpLike(src)) {
    const r = await fetch(src);
    if (!r.ok) throw new Error(`fetchBytesSmart: ${r.status} ${r.statusText} for ${src}`);
    const ab = await r.arrayBuffer();
    return new Uint8Array(ab);
  }

  // file:// or plain local path
  const path = toLocalPath(src);
  const buf = await readFile(path);
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}
