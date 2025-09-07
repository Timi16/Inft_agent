// ExternalVectorStore: load vectors from external URI, verify checksum, decode, and provide search.
// Supports IPFS (via gateway), HTTP(S), local file URLs, and bare file paths.

import { decodeVectorsBlob } from "./binary.vector"; // ← fix filename
import { fetchBytesSmart } from "../utils/uri";
import { cosine, normalizeInPlace } from "../utils/cosine";
import { sha256Hex } from "../utils/hash";
import type { InftManifest, VectorHit } from "../types";

export class ExternalVectorStore {
  readonly model = { id: "", dim: 0, normalize: true, instruction: "e5" as "e5" | "none" };
  private vectors: Float32Array[] = [];
  private texts: string[] = [];
  private metas: (Record<string, unknown> | undefined)[] = [];
  private ids: string[] = [];

  static async fromManifestWithExternalVectors(
    manifest: InftManifest,
    gateway?: string
  ): Promise<ExternalVectorStore> {
    if (!manifest.vectors_uri) {
      throw new Error("Manifest has no vectors_uri");
    }

    // 1) Fetch bytes (ipfs/http/file/path)
    const bytes = await fetchBytesSmart(manifest.vectors_uri, gateway);

    // 2) Checksum verify (supports manifest.vectors_checksum or model.checksum)
    const expected = manifest.vectors_checksum || manifest.model.checksum;
    if (expected) {
      const got = sha256Hex(bytes);
      if (got !== expected) {
        throw new Error(`Vectors checksum mismatch: got=${got} expected=${expected}`);
      }
    }

    // 3) Decode vectors blob
    const { header, vectors } = decodeVectorsBlob(bytes.buffer);

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
    store.model.instruction = manifest.model.instruction;

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

  size(): number {
    return this.vectors.length;
  }

  search(
    queryVec: Float32Array,
    k = 8,
    filter?: (meta?: Record<string, unknown>) => boolean
  ): VectorHit[] {
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
