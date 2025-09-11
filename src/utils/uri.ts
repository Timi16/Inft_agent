// src/utils/uri.ts
// URI helpers + robust fetchers for IPFS/http/file, with Pinata gateway normalization.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

/** Strip surrounding single/double quotes */
export function dequote(s?: string | null): string {
  return (s ?? "").trim().replace(/^[\'\"]+|[\'\"]+$/g, "");
}

/** http(s)://... */
export function isHttpLike(u: string): boolean {
  return /^https?:\/\//i.test(u);
}

/** file://... */
export function isFileLike(u: string): boolean {
  return /^file:\/\//i.test(u);
}

/** CIDv0/v1 quick tests */
export function looksLikeCid(s: string): boolean {
  return /^Qm[1-9A-HJ-NP-Za-km-z]{44}$/.test(s) || /^bafy[2-7a-z]{10,}$/i.test(s);
}

/** ipfs://, /ipfs/<cid>, or bare CID */
export function isIpfsLike(s: string): boolean {
  return /^ipfs:\/\//i.test(s) || /^\/?ipfs\//i.test(s) || looksLikeCid(s);
}

/** Convert ipfs://<cid>/path OR bare CID to HTTP gateway URL (.../ipfs/<cid>/path) */
export function ipfsToHttp(ipfsLike: string, gateway?: string): string {
  const base = (gateway || process.env.PINATA_GATEWAY || "https://ipfs.io").replace(/\/+$/, "");
  if (/^ipfs:\/\//i.test(ipfsLike)) {
    const rest = ipfsLike.replace(/^ipfs:\/\//i, ""); // "<cid>/path?"
    return `${base}/ipfs/${rest}`;
  }
  if (/^\/?ipfs\//i.test(ipfsLike)) {
    const rest = ipfsLike.replace(/^\/?ipfs\//i, "");
    return `${base}/ipfs/${rest}`;
  }
  if (looksLikeCid(ipfsLike)) {
    return `${base}/ipfs/${ipfsLike}`;
  }
  return ipfsLike;
}

/** For http(s) URLs hitting Pinata gateways without /ipfs/, insert it. */
export function normalizeGatewayHttpUrl(u: string): string {
  try {
    const url = new URL(u);
    const host = url.hostname;
    const path = url.pathname;

    // Dedicated gateway hosts
    const isPinata =
      /\.mypinata\.cloud$/i.test(host) ||
      /(^|\.)pinata\.cloud$/i.test(host);

    const hasIpfsPrefix = /^\/ip(fs|ns)\//i.test(path);

    // path like "/<cid>" or "/<cid>/...":
    const m = path.match(/^\/(Qm[1-9A-HJ-NP-Za-km-z]{44}|bafy[2-7a-z]{10,})(\/.*)?$/i);

    if (isPinata && !hasIpfsPrefix && m) {
      const tail = m[2] ?? "";
      url.pathname = `/ipfs/${m[1]}${tail}`;
      return url.toString();
    }
    return url.toString();
  } catch {
    return u;
  }
}

/** file:// URL -> local fs path; otherwise return as-is (relative -> CWD absolute) */
export function toLocalPath(u: string): string {
  if (isFileLike(u)) return fileURLToPath(u);
  // Normalize relative paths to absolute
  if (!path.isAbsolute(u)) return path.resolve(process.cwd(), u);
  return u;
}

/** Fetch raw bytes from ipfs/http/file with all normalizations applied. */
export async function fetchBytesSmart(uriOrPath: string, gateway?: string): Promise<Uint8Array> {
  const src = dequote(uriOrPath);

  if (isIpfsLike(src)) {
    // Always force .../ipfs/<cid> on chosen gateway
    const url = ipfsToHttp(src, gateway);
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetchBytesSmart: ${r.status} ${r.statusText} for ${url}`);
    return new Uint8Array(await r.arrayBuffer());
  }

  if (isHttpLike(src)) {
    // If it's a Pinata URL missing /ipfs/, fix it.
    const url = normalizeGatewayHttpUrl(src);
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetchBytesSmart: ${r.status} ${r.statusText} for ${url}`);
    return new Uint8Array(await r.arrayBuffer());
  }

  // file:// or plain local path
  const p = toLocalPath(src);
  const buf = await readFile(p);
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}

/** Fetch text/JSON helpers (optional but handy) */
export async function fetchJsonSmart<T = unknown>(uriOrPath: string, gateway?: string): Promise<T> {
  const src = dequote(uriOrPath);

  if (isIpfsLike(src)) {
    const url = ipfsToHttp(src, gateway);
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetchJsonSmart: ${r.status} ${r.statusText} for ${url}`);
    return (await r.json()) as T;
  }

  if (isHttpLike(src)) {
    const url = normalizeGatewayHttpUrl(src);
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetchJsonSmart: ${r.status} ${r.statusText} for ${url}`);
    return (await r.json()) as T;
  }

  const p = toLocalPath(src);
  const text = await readFile(p, "utf8");
  return JSON.parse(text) as T;
}
