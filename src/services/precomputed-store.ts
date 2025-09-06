// Loads an iNFT manifest into memory and provides fast cosine search.
// This uses a simple brute-force scan (great up to ~50k vectors).
// You can later upgrade to an HNSW index behind the same interface.

import { base64ToFloat32 } from "../utils/base64";
import { cosine, normalizeInPlace } from "../utils/cosine";
import type { InftManifest, VectorHit } from "../types";
import { InftManifestSchema } from "../schema/inft-manifest-schema";

export interface StoreLoadOptions {
  reNormalize?: boolean; // if manifest.normalize=false or unsure, enforce unit length
}

export class PrecomputedVectorStore {
  readonly model = { id: "", dim: 0, normalize: true, instruction: "e5" as "e5" | "none" };
  private vectors: Float32Array[] = [];
  private texts: string[] = [];
  private metas: (Record<string, unknown> | undefined)[] = [];
  private ids: string[] = [];

  static fromManifest(manifest: unknown, opts: StoreLoadOptions = {}): PrecomputedVectorStore {
    const parsed = InftManifestSchema.parse(manifest);
    const store = new PrecomputedVectorStore();
    store.model.id = parsed.model.id;
    store.model.dim = parsed.model.dim;
    store.model.normalize = parsed.model.normalize;
    store.model.instruction = parsed.model.instruction;

    for (const e of parsed.entries) {
      let vec: Float32Array | null = null;

      if (e.embedding_b64) {
        vec = base64ToFloat32(e.embedding_b64);
      } else if (Array.isArray(e.embedding)) {
        vec = new Float32Array(e.embedding);
      }

      if (!vec) continue;
      if (vec.length !== parsed.model.dim) {
        throw new Error(`Entry ${e.id} has dim ${vec.length}, expected ${parsed.model.dim}`);
      }

      // Ensure unit-length if requested or the manifest says normalize=false
      if (opts.reNormalize || !parsed.model.normalize) {
        normalizeInPlace(vec);
      }

      store.vectors.push(vec);
      store.texts.push(e.text);
      store.metas.push(e.meta);
      store.ids.push(e.id);
    }

    if (store.vectors.length === 0) {
      throw new Error("No vectors loaded from manifest entries.");
    }

    return store;
  }

  size(): number {
    return this.vectors.length;
  }

  /** Cosine search over the in-memory set. Assumes queryVec is already unit-normalized. */
  search(queryVec: Float32Array, k = 8, filter?: (meta?: Record<string, unknown>) => boolean): VectorHit[] {
    const scores: { i: number; s: number }[] = [];

    for (let i = 0; i < this.vectors.length; i++) {
      if (filter && !filter(this.metas[i])) continue;
      const s = cosine(queryVec, this.vectors[i]);
      scores.push({ i, s });
    }

    // Partial sort would be faster; for simplicity, full sort here
    scores.sort((a, b) => b.s - a.s);
    const top = scores.slice(0, Math.min(k, scores.length));

    return top.map(({ i, s }) => ({
      id: this.ids[i],
      score: s,
      text: this.texts[i],
      meta: this.metas[i],
    }));
  }
}
