// src/services/external-store.ts
// ExternalVectorStore: load vectors from external URI, verify checksum, decode, and provide search.
// Supports IPFS (via gateway), HTTP(S), local file URLs, and bare file paths.

import { fetchBytesSmart, isHttpLike, isIpfsLike, isFileLike, toLocalPath } from "../utils/uri";
import { cosine, normalizeInPlace } from "../utils/cosine";
import { sha256Hex } from "../utils/hash";
import { decodeVectorsBlob } from "./binary.vector"

// Prefer your shared types if present
import type { InftManifest } from "../types";
// If you don't have VectorHit defined centrally, uncomment this:
// export type VectorHit = { id: string; score: number; text: string; meta?: Record<string, unknown> };

export class ExternalVectorStore {
  readonly model = {
    id: "",
    dim: 0,
    normalize: true,
    instruction: "e5" as "e5" | "none",
  };

  private vectors: Float32Array[] = [];
  private texts: string[] = [];
  private metas: (Record<string, unknown> | undefined)[] = [];
  private ids: string[] = [];

  /**
   * Load an ExternalVectorStore from a manifest that references an external vectors blob.
   * - Fetches bytes via ipfs/http/file/local path
   * - Verifies checksum (if provided)
   * - Decodes blob and checks (dim,count) against manifest
   * - Populates in-memory vectors + metadata
   *
   * NOTE on relative vectors_uri:
   * If your manifest uses a relative vectors_uri (e.g. "vectors.bin"), you can
   * attach `__source_uri__` (the manifest's own URI) in loadManifest so we can
   * resolve relative paths against the manifest directory here.
   */
  static async fromManifestWithExternalVectors(
    manifest: InftManifest & { __source_uri__?: string },
    gateway?: string
  ): Promise<ExternalVectorStore> {
    const raw = (manifest.vectors_uri || "").trim();
    if (!raw) throw new Error("Manifest has no vectors_uri");

    // Do NOT resolve/join if already absolute (ipfs/http/file/absolute path)
    const isWinAbs = /^[a-zA-Z]:[\\/]/.test(raw);
    const isUnixAbs = raw.startsWith("/");
    const needsBase = !isHttpLike(raw) && !isIpfsLike(raw) && !isFileLike(raw) && !isWinAbs && !isUnixAbs;

    let resolved = raw;
    if (needsBase && manifest.__source_uri__) {
      // If your loader sets __source_uri__ (the manifest path/URL),
      // you can resolve a relative vectors_uri against its directory here.
      // Example (if you want this behavior):
      //   const baseDir = path.dirname(toLocalPath(manifest.__source_uri__));
      //   resolved = path.resolve(baseDir, raw);
      // For now, we keep `resolved = raw` to avoid surprises.
      resolved = raw;
    }

    // 1) Fetch bytes (ipfs/http/file/path)
    const bytes = await fetchBytesSmart(resolved, gateway);

    // 2) Checksum verify (supports manifest.vectors_checksum or model.checksum)
    const expected = manifest.vectors_checksum || (manifest.model as any)?.checksum;
    if (expected) {
      const got = sha256Hex(bytes);
      if (got !== expected) {
        throw new Error(`Vectors checksum mismatch: got=${got} expected=${expected}`);
      }
    }

    // 3) Decode vectors blob
    const { header, vectors } = decodeVectorsBlob(bytes);

    // 4) Contract checks
    if (header.dim !== manifest.model.dim) {
      throw new Error(`Blob dim ${header.dim} != manifest dim ${manifest.model.dim}`);
    }
    if (vectors.length !== manifest.entries.length) {
      throw new Error(`Blob count ${vectors.length} != manifest entries ${manifest.entries.length}`);
    }

    // 5) Assemble store
    const store = new ExternalVectorStore();
    store.model.id = manifest.model.id;
    store.model.dim = manifest.model.dim;
    store.model.normalize = manifest.model.normalize;
    store.model.instruction = (manifest.model as any).instruction ?? "e5";

    for (let i = 0; i < vectors.length; i++) {
      const v = vectors[i];
      // If vectors were not normalized at pack-time, normalize now for cosine correctness.
      if (!store.model.normalize) normalizeInPlace(v);

      store.vectors.push(v);

      const entry = manifest.entries[i] as any;
      store.texts.push(entry.text);
      store.metas.push(entry.meta);
      store.ids.push(entry.id);
    }

    return store;
  }

  size(): number {
    return this.vectors.length;
  }

  /**
   * Brute-force cosine search (works fine for small KBs; swap for HNSW later if needed).
   * Ensure query vector is normalized before calling for optimal cosine scores.
   */
  search(
    queryVec: Float32Array,
    k = 8,
    filter?: (meta?: Record<string, unknown>) => boolean
  ) {
    const scores: { i: number; s: number }[] = [];
    for (let i = 0; i < this.vectors.length; i++) {
      if (filter && !filter(this.metas[i])) continue;
      scores.push({ i, s: cosine(queryVec, this.vectors[i]) });
    }
    scores.sort((a, b) => b.s - a.s);
    return scores.slice(0, Math.min(k, scores.length)).map(({ i, s }) => ({
      id: this.ids[i],
      score: s,
      text: this.texts[i],
      meta: this.metas[i],
    }));
  }
}
