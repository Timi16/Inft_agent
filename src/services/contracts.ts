import type { InftModelCard } from "../types";
import type { EmbedderInfo } from "./local-embedder";

/** Throws if manifest model contract and embedder info don't match. */
export function ensureModelContract(manifest: InftModelCard, embedder: EmbedderInfo, opts?: { allowDimNull?: boolean }) {
  if (embedder.id && manifest.id && embedder.id !== manifest.id) {
    throw new Error(`Embedder model id mismatch: runtime=${embedder.id} vs manifest=${manifest.id}`);
  }
  if (!(opts?.allowDimNull && embedder.dim == null) && embedder.dim !== manifest.dim) {
    throw new Error(`Embedder dim mismatch: runtime=${embedder.dim} vs manifest=${manifest.dim}`);
  }
}