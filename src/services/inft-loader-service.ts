// Loads an iNFT manifest from a file path, file:// URL, or HTTP(S) URL.

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

export async function loadManifest(source: string): Promise<InftManifest> {
  let raw: unknown;

  try {
    if (isHttpUrl(source)) {
      const res = await fetch(source);
      if (!res.ok) {
        throw new Error(`Fetch failed: ${res.status} ${res.statusText}`);
      }
      raw = await res.json();
    } else if (isFileUrl(source)) {
      // Convert file:// to a real filesystem path (Windows-safe).
      const p = fileURLToPath(source);
      const buf = await readFile(p, "utf8");
      raw = JSON.parse(buf);
    } else {
      // Treat as plain filesystem path.
      const buf = await readFile(source, "utf8");
      raw = JSON.parse(buf);
    }
  } catch (e: any) {
    // Give a helpful message for common pitfalls.
    const hint =
      isFileUrl(source)
        ? "Tip: file:// URLs must use file:///C:/... on Windows."
        : isHttpUrl(source)
        ? "Tip: ensure the URL is reachable and returns JSON."
        : "Tip: ensure the path exists and is readable.";
    throw new Error(`Failed to load manifest from "${source}": ${e?.message}\n${hint}`);
  }

  // Validate with Zod but surface where it failed.
  try {

    const res =InftManifestSchema.parse(raw);
    return res
  } catch (e: any) {
    throw new Error(
      `Manifest schema validation failed for "${source}":\n${e?.message ?? e}`,
    );
  }
}
