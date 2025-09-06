// In-memory store for manifests with inline vectors.

import { base64ToFloat32 } from "../utils/base64";
import { cosine, normalizeInPlace } from "../utils/cosine";
import type { InftManifest, VectorHit } from "../types";
import { InftManifestSchema } from "../schema/inft-manifest-schema";

export interface StoreLoadOptions {
  reNormalize?: boolean;
}

export class PrecomputedVectorStore {
  readonly model = { id: "", dim: 0, normalize: true, instruction: "e5" as "e5" | "none" };
  protected vectors: Float32Array[] = [];
  protected texts: string[] = [];
  protected metas: (Record<string, unknown> | undefined)[] = [];
  protected ids: string[] = [];

  static fromManifest(manifest: unknown, opts: StoreLoadOptions = {}): PrecomputedVectorStore {
    const parsed = InftManifestSchema.parse(manifest);
    const store = new PrecomputedVectorStore();
    store.model.id = parsed.model.id;
    store.model.dim = parsed.model.dim;
    store.model.normalize = parsed.model.normalize;
    store.model.instruction = parsed.model.instruction;

    for (const e of parsed.entries) {
      let vec: Float32Array | undefined;
      if (e.embedding_b64) vec = base64ToFloat32(e.embedding_b64);
      else if (Array.isArray(e.embedding)) vec = new Float32Array(e.embedding);

      if (!vec) continue;
      if (vec.length !== store.model.dim) throw new Error(`Dim mismatch ${e.id}: ${vec.length} vs ${store.model.dim}`);

      if (opts.reNormalize || !parsed.model.normalize) normalizeInPlace(vec);

      store.vectors.push(vec);
      store.texts.push(e.text);
      store.metas.push(e.meta);
      store.ids.push(e.id);
    }

    if (store.vectors.length === 0) throw new Error("No vectors loaded from inline entries.");
    return store;
  }

  size(): number { return this.vectors.length; }

  search(queryVec: Float32Array, k = 8, filter?: (meta?: Record<string, unknown>) => boolean): VectorHit[] {
    const scores: { i: number; s: number }[] = [];
    for (let i = 0; i < this.vectors.length; i++) {
      if (filter && !filter(this.metas[i])) continue;
      scores.push({ i, s: cosine(queryVec, this.vectors[i]) });
    }
    scores.sort((a, b) => b.s - a.s);
    return scores.slice(0, Math.min(k, scores.length)).map(({ i, s }) => ({
      id: this.ids[i], score: s, text: this.texts[i], meta: this.metas[i],
    }));
  }
}
