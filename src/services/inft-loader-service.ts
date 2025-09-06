// src/services/inft-loader.service.ts
// Loads an iNFT manifest from a file path or HTTP(S) URL.
// For on-chain NFTs, resolve tokenURI in your app then pass the URL here.

import { readFile } from "node:fs/promises";
import { InftManifestSchema } from "../schema/inft-manifest-schema";
import type { InftManifest } from "../types";

export async function loadManifest(source: string): Promise<InftManifest> {
  if (/^https?:\/\//i.test(source)) {
    const res = await fetch(source);
    if (!res.ok) throw new Error(`Failed to fetch manifest: ${res.status} ${res.statusText}`);
    const json = await res.json();
    return InftManifestSchema.parse(json);
  }
  // Treat as file path
  const buf = await readFile(source, "utf8");
  const json = JSON.parse(buf);
  return InftManifestSchema.parse(json);
}
