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

function normalizeGatewayHttpUrl(u: string): string {
  // If someone passed a Pinata subdomain URL without /ipfs/, insert it.
  // Examples fixed:
  //   https://SUB.mypinata.cloud/Qm...        -> https://SUB.mypinata.cloud/ipfs/Qm...
  //   https://gateway.pinata.cloud/Qm...      -> https://gateway.pinata.cloud/ipfs/Qm...
  try {
    const url = new URL(u);
    const host = url.hostname;
    const path = url.pathname;

    // Matches leading /<cid> or /<cid>/something
    const cidMatch = path.match(/^\/(Qm[1-9A-HJ-NP-Za-km-z]{44}|bafy[2-7a-z]{10,})(\/.*)?$/i);

    const isPinataHost =
      /\.mypinata\.cloud$/i.test(host) ||
      /(^|\.)pinata\.cloud$/i.test(host); // gateway.pinata.cloud or subdomain

    const hasIpfsPrefix = /^\/ip(fs|ns)\//i.test(path);

    if (isPinataHost && cidMatch && !hasIpfsPrefix) {
      url.pathname = `/ipfs/${cidMatch[1]}${cidMatch[2] ?? ""}`;
      return url.toString();
    }

    // leave other gateways/URLs as-is
    return url.toString();
  } catch {
    return u; // not a valid URL string; leave untouched
  }
}

/** Fetch bytes from ipfs/http/file/local path safely */

export async function fetchBytesSmart(uriOrPath: string, gateway?: string): Promise<Uint8Array> {
  const src = dequote(uriOrPath);

  if (isIpfsLike(src)) {
    const url = ipfsToHttp(src, gateway);        // ensures .../ipfs/<cid>
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetchBytesSmart: ${r.status} ${r.statusText} for ${url}`);
    const ab = await r.arrayBuffer();
    return new Uint8Array(ab);
  }

  if (isHttpLike(src)) {
    const canonical = normalizeGatewayHttpUrl(src);  // <-- new
    const r = await fetch(canonical);
    if (!r.ok) throw new Error(`fetchBytesSmart: ${r.status} ${r.statusText} for ${canonical}`);
    const ab = await r.arrayBuffer();
    return new Uint8Array(ab);
  }

  // file:// or plain local path
  const path = toLocalPath(src);
  const buf = await readFile(path);
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}