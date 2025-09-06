//External store implementation

// src/services/external-store.ts
// Store that can load large vector sets from vectors_uri (ipfs:// or https://).
// Uses the same ordering as manifest.entries (text/meta come from the JSON).

import { decodeVectorsBlob } from "./binary.vector";
import { resolveToHttp } from "../utils/uri";
import { cosine, normalizeInPlace } from "../utils/cosine";
import type { InftManifest, VectorHit } from "../types";

export class ExternalVectorStore {
  readonly model = { id: "", dim: 0, normalize: true, instruction: "e5" as "e5" | "none" };
  private vectors: Float32Array[] = [];
  private texts: string[] = [];
  private metas: (Record<string, unknown> | undefined)[] = [];
  private ids: string[] = [];

  static async fromManifestWithExternalVectors(manifest: InftManifest, gateway?: string): Promise<ExternalVectorStore> {
    if (!manifest.vectors_uri) throw new Error("Manifest has no vectors_uri");
    const httpUrl = resolveToHttp(manifest.vectors_uri, gateway);

    const res = await fetch(httpUrl);
    if (!res.ok) throw new Error(`Failed to fetch vectors blob: ${res.status} ${res.statusText}`);
    const buf = await res.arrayBuffer();

    const { header, vectors } = decodeVectorsBlob(buf);

    // Basic contract checks
    if (header.dim !== manifest.model.dim) {
      throw new Error(`Blob dim ${header.dim} != manifest dim ${manifest.model.dim}`);
    }
    if (vectors.length !== manifest.entries.length) {
      throw new Error(`Blob count ${vectors.length} != manifest entries ${manifest.entries.length}`);
    }

    const store = new ExternalVectorStore();
    store.model.id = manifest.model.id;
    store.model.dim = manifest.model.dim;
    store.model.normalize = manifest.model.normalize;
    store.model.instruction = manifest.model.instruction;

    // Normalize if needed
    for (let i = 0; i < vectors.length; i++) {
      const v = vectors[i];
      if (!store.model.normalize) normalizeInPlace(v);
      store.vectors.push(v);
      const entry = manifest.entries[i];
      store.texts.push(entry.text);
      store.metas.push(entry.meta);
      store.ids.push(entry.id);
    }

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
