// src/services/inft-loader-service.ts
// Loads an iNFT manifest from ipfs://, bare CID, /ipfs/<cid>, http(s), file://, or local paths.
// Normalizes dedicated Pinata gateway URLs that forget "/ipfs/".

import { InftManifestSchema } from "../schema/inft-manifest-schema";
import type { InftManifest } from "../types";
import {
  fetchJsonSmart,
  isIpfsLike,
  isHttpLike,
  isFileLike,
  ipfsToHttp,
  normalizeGatewayHttpUrl,
} from "../utils/uri";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

function toGatewayUrl(ipfsLike: string, gateway?: string): string {
  return ipfsToHttp(ipfsLike, gateway);
}

export async function loadManifest(
  source: string,
  opts?: { gateway?: string }
): Promise<InftManifest> {
  const gateway = opts?.gateway || process.env.PINATA_GATEWAY;

  // Route IPFS-like sources through a gateway
  if (isIpfsLike(source)) {
    const url = toGatewayUrl(source, gateway);
    const raw = await fetchJsonSmart(url);
    return InftManifestSchema.parse(raw);
  }

  // HTTP(S) sources — normalize Pinata paths if needed
  if (isHttpLike(source)) {
    const url = normalizeGatewayHttpUrl(source);
    const raw = await fetchJsonSmart(url);
    return InftManifestSchema.parse(raw);
  }

  // file:// URL
  if (isFileLike(source)) {
    const p = fileURLToPath(source);
    const text = await readFile(p, "utf8");
    const raw = JSON.parse(text);
    return InftManifestSchema.parse(raw);
  }

  // Plain filesystem path
  const p = path.isAbsolute(source) ? source : path.resolve(process.cwd(), source);
  const text = await readFile(p, "utf8");
  const raw = JSON.parse(text);
  return InftManifestSchema.parse(raw);
}
