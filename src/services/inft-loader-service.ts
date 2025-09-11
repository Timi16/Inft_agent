// Loads an iNFT manifest from a file path, file:// URL, HTTP(S) URL, or IPFS (ipfs://, /ipfs/, or bare CID).

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { InftManifestSchema } from "../schema/inft-manifest-schema";
import type { InftManifest } from "../types";

function isHttpUrl(s: string) {
  return /^https?:\/\//i.test(s);
}
function isFileUrl(s: string) {
  return /^file:\/\//i.test(s);
}
function looksLikeCid(s: string) {
  // CIDv0 (base58btc) e.g. Qm... (46 chars), CIDv1 (base32) e.g. bafy...
  return /^Qm[1-9A-HJ-NP-Za-km-z]{44}$/.test(s) || /^bafy[2-7a-z]{10,}$/i.test(s);
}
function isIpfsLike(s: string) {
  return /^ipfs:\/\//i.test(s) || /^\/?ipfs\//i.test(s) || looksLikeCid(s);
}
function toGatewayUrl(ipfsLike: string, gateway?: string) {
  const base = (gateway || process.env.PINATA_GATEWAY || "https://ipfs.io").replace(/\/+$/, "");
  if (/^ipfs:\/\//i.test(ipfsLike)) {
    const rest = ipfsLike.replace(/^ipfs:\/\//i, "");
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

export async function loadManifest(
  source: string,
  opts?: { gateway?: string }
): Promise<InftManifest> {
  let raw: unknown;

  // Normalize IPFS → HTTP gateway if needed
  const resolved = isIpfsLike(source) ? toGatewayUrl(source, opts?.gateway) : source;

  try {
    if (isHttpUrl(resolved)) {
      const res = await fetch(resolved, { redirect: "follow" as RequestRedirect });
      if (!res.ok) {
        const body = await res.text().catch(() => "");
        throw new Error(`Fetch failed: ${res.status} ${res.statusText}${body ? ` — ${body.slice(0, 200)}...` : ""}`);
      }
      // Some gateways can return text/plain; parse manually
      const text = await res.text();
      raw = JSON.parse(text);
    } else if (isFileUrl(resolved)) {
      // Convert file:// to a real filesystem path (Windows-safe).
      const p = fileURLToPath(resolved);
      const buf = await readFile(p, "utf8");
      raw = JSON.parse(buf);
    } else {
      // Treat as plain filesystem path.
      const buf = await readFile(resolved, "utf8");
      raw = JSON.parse(buf);
    }
  } catch (e: any) {
    const hint = isIpfsLike(source)
      ? `Tip: set PINATA_GATEWAY or pass opts.gateway (e.g. https://YOURSUB.mypinata.cloud) to avoid public gateway rate limits.`
      : isFileUrl(resolved)
      ? "Tip: file:// URLs on Windows must look like file:///C:/path/to/file.json."
      : isHttpUrl(resolved)
      ? "Tip: ensure the URL is reachable and returns JSON."
      : "Tip: ensure the path exists and is readable.";
    throw new Error(`Failed to load manifest from "${source}": ${e?.message}\n${hint}`);
  }

  // Validate with Zod; surface where it failed.
  try {
    return InftManifestSchema.parse(raw);
  } catch (e: any) {
    throw new Error(`Manifest schema validation failed for "${source}":\n${e?.message ?? e}`);
  }
}
