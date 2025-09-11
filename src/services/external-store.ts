// src/services/external-store.ts
// Vector store backed by external "vectors.bin" referenced in manifest.vectors_uri.
// Loads bytes from ipfs/http/file via fetchBytesSmart, decodes them, and supports cosine search.

import type { InftManifest } from "../types";
import { fetchBytesSmart, isIpfsLike, ipfsToHttp, normalizeGatewayHttpUrl } from "../utils/uri";
import { normalizeInPlace } from "../utils/cosine";

// If you already have a decoder, import it. The function should accept Uint8Array and return Float32Array[].
// Adjust the import/name below to match your project.
import { decodeVectorsBlob } from "./binary.vector"; // <-- ensure this exists

export type SearchHit = {
  index: number;
  score: number; // cosine similarity if vectors are normalized
  text: string;
  id?: string | number;
};

export class ExternalVectorStore {
  private vecs: Float32Array[];
  private texts: string[];
  private ids: (string | number | undefined)[];
  private dim: number;
  private normalized: boolean;

  private constructor(opts: {
    vecs: Float32Array[];
    texts: string[];
    ids: (string | number | undefined)[];
    dim: number;
    normalized: boolean;
  }) {
    this.vecs = opts.vecs;
    this.texts = opts.texts;
    this.ids = opts.ids;
    this.dim = opts.dim;
    this.normalized = opts.normalized;
  }

  static async fromManifestWithExternalVectors(
    manifest: InftManifest,
    gateway?: string
  ): Promise<ExternalVectorStore> {
    const vectorsUri: string | undefined =
      (manifest as any).vectors_uri || (manifest as any).vectorsUrl || (manifest as any).vectors;
    if (!vectorsUri) {
      throw new Error("Manifest has no vectors_uri/vectorsUrl/vectors.");
    }

    // Build the final URL/path to load bytes
    let src = vectorsUri;
    if (isIpfsLike(src)) {
      src = ipfsToHttp(src, gateway);
    } else if (/^https?:\/\//i.test(src)) {
      src = normalizeGatewayHttpUrl(src);
    }

    const bytes = await fetchBytesSmart(src, gateway);

    // Decode vectors.bin → Float32Array[] (adjust to your decoder's API if needed)
    const decoded: any = decodeVectorsBlob(bytes);
    const vecs: Float32Array[] = Array.isArray(decoded)
      ? decoded
      : (decoded?.vectors || decoded?.vecs);

    if (!Array.isArray(vecs) || vecs.length === 0) {
      throw new Error("Failed to decode vectors: empty or invalid vectors.bin.");
    }

    // Text + IDs from manifest entries
    const entries: any[] = Array.isArray((manifest as any).entries) ? (manifest as any).entries : [];
    const texts = entries.map((e) => String(e.text ?? e.content ?? ""));
    const ids = entries.map((e, i) => e.id ?? e._id ?? i);

    // If mismatch, trim to the smaller length (common when filtering entries elsewhere)
    const n = Math.min(vecs.length, texts.length || vecs.length);
    const vecsN = vecs.slice(0, n);
    const textsN = (texts.length ? texts : new Array(vecs.length).fill("")).slice(0, n);
    const idsN = ids.slice(0, n);

    // Normalize vectors if required by the manifest contract
    const shouldNormalize = Boolean((manifest as any).e_model?.normalize);
    if (shouldNormalize) {
      for (const v of vecsN) normalizeInPlace(v);
    }

    const dim = (manifest as any).e_model?.dim ?? (vecsN[0]?.length ?? 0);
    if (!dim) throw new Error("Cannot determine vector dimension.");

    return new ExternalVectorStore({
      vecs: vecsN,
      texts: textsN,
      ids: idsN,
      dim,
      normalized: shouldNormalize,
    });
  }

  size(): number {
    return this.vecs.length;
  }

  /** Search by cosine similarity (if vectors are normalized, dot == cosine). */
  search(query: Float32Array, topK = 5): SearchHit[] {
    if (query.length !== this.dim) {
      throw new Error(`Query dim ${query.length} != store dim ${this.dim}`);
    }

    const q = new Float32Array(query); // copy (may already be normalized)
    if (!this.normalized) {
      // If stored vectors aren't normalized, compute cosine by normalizing q and each vector.
      normalizeInPlace(q);
    }

    // Compute scores
    const scores: number[] = new Array(this.vecs.length);
    for (let i = 0; i < this.vecs.length; i++) {
      const v = this.vecs[i];
      let s = 0;
      const L = this.dim;
      for (let j = 0; j < L; j++) s += q[j] * v[j];
      scores[i] = s;
    }

    // Arg-sort topK
    const idxs = scores.map((_, i) => i);
    idxs.sort((a, b) => scores[b] - scores[a]);
    const K = Math.min(topK, idxs.length);

    const hits: SearchHit[] = [];
    for (let r = 0; r < K; r++) {
      const i = idxs[r];
      hits.push({
        index: i,
        score: scores[i],
        text: this.texts[i] ?? "",
        id: this.ids[i],
      });
    }
    return hits;
  }
}
