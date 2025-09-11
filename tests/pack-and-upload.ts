#!/usr/bin/env tsx
/**
 * Pack → IPFS (Pinata) only. No file:// outputs, no local writes.
 *
 * - Loads a character manifest (text-only, inline vectors, or already external).
 * - If text-only: embeds with e_model (remote /v1/embeddings) and builds vectors.bin in-memory.
 * - If inline vectors: externalizes to vectors.bin in-memory.
 * - Uploads vectors.bin to Pinata (pinFileToIPFS) → vectors CID.
 * - Produces final manifest with vectors_uri = ipfs://<vectorsCID> and vectors_checksum.
 * - Uploads final manifest (pinJSONToIPFS) → manifest CID.
 *
 * ENV (required):
 *   INPUT_MANIFEST=path\to\manifest.json   // source character file
 *   PINATA_JWT=eyJhbGciOi...               // Pinata JWT
 *
 * ENV (recommended):
 *   EMBEDDINGS_BASE_URL=http://127.0.0.1:8080  // your embeddings service base
 *
 * ENV (optional):
 *   MODEL_ID=Xenova/bge-small-en-v1.5      // override e_model.id for embedding
 *   QUANT=fp16|fp32|int8                   // default fp16 (int8 falls back to fp16)
 *   CLEAR_INLINE=1|0                       // default 1 (strip inline embeddings after externalizing)
 *   PINATA_GATEWAY=https://...             // just for printing friendly URLs
 */

import "dotenv/config";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { createHash } from "node:crypto";

import { InftManifestSchema } from "../src/schema/inft-manifest-schema";
import type { InftManifestParsed } from "../src/schema/inft-manifest-schema";
import { encodeVectorsBlob } from "../src/services/binary.vector";

// ---- helpers (no local files) ----
const env = (k: string, d?: string) => (process.env[k] ?? d)?.toString().trim();
const bool = (k: string, def = false) => {
  const v = (process.env[k] ?? "").toLowerCase().trim();
  return v === "1" || v === "true" || (v === "" ? def : false);
};

function sha256Hex(u8: Uint8Array) {
  const h = createHash("sha256");
  h.update(u8);
  return "sha256:" + h.digest("hex");
}
function b64ToF32(b64: string): Float32Array {
  const buf = Buffer.from(b64, "base64");
  return new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
}
type Quant = 0 | 1; // 0=fp32, 1=fp16
function pickQuant(q?: string): Quant {
  const s = (q ?? "fp16").toLowerCase();
  if (s === "fp32") return 0;
  if (s === "fp16") return 1;
  if (s === "int8") {
    console.warn("[warn] QUANT=int8 not supported; using fp16.");
    return 1;
  }
  console.warn(`[warn] Unknown QUANT=${q}; defaulting to fp16.`);
  return 1;
}

// ---- Pinata (JWT) ----
// We keep it self-contained to avoid adding deps. Use global FormData/Blob from Node 20+/23.
async function pinFileToIPFS_JWT(fileBytes: Uint8Array, fileName: string, jwt: string): Promise<string> {
  const fd = new (globalThis as any).FormData();
  const blob = new (globalThis as any).Blob([fileBytes], { type: "application/octet-stream" });
  fd.append("file", blob, fileName);

  const res = await fetch("https://api.pinata.cloud/pinning/pinFileToIPFS", {
    method: "POST",
    headers: { Authorization: `Bearer ${jwt}` },
    body: fd as any,
  });
  if (!res.ok) throw new Error(`pinFileToIPFS failed: ${res.status} ${await res.text()}`);
  const json = await res.json();
  const cid = json?.IpfsHash;
  if (!cid) throw new Error(`pinFileToIPFS: no IpfsHash in response: ${JSON.stringify(json)}`);
  return cid;
}

async function pinJSONToIPFS_JWT(obj: unknown, jwt: string): Promise<string> {
  const res = await fetch("https://api.pinata.cloud/pinning/pinJSONToIPFS", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${jwt}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ pinataContent: obj }),
  });
  if (!res.ok) throw new Error(`pinJSONToIPFS failed: ${res.status} ${await res.text()}`);
  const json = await res.json();
  const cid = json?.IpfsHash;
  if (!cid) throw new Error(`pinJSONToIPFS: no IpfsHash in response: ${JSON.stringify(json)}`);
  return cid;
}

// ---- Remote embeddings  ----
async function remoteEmbedBatch(
  baseUrl: string,
  texts: string[],
  opts: { modelId?: string; instruction: "e5" | "none"; mode: "query" | "passage"; normalize: boolean }
): Promise<Float32Array[]> {
  const body = {
    input: texts,
    mode: opts.mode,
    instruction: opts.instruction,
    normalize: opts.normalize,
  };
  const res = await fetch(`${baseUrl.replace(/\/+$/, "")}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Remote embed failed: ${res.status} ${await res.text()}`);
  const json = await res.json();
  const arrs: number[][] = json?.data?.map((d: any) => d.embedding) ?? [];
  return arrs.map((a) => new Float32Array(a));
}

// ---- Main ----
async function main() {
  const inputPath = resolve("C:/Users/A/Documents/inft-agent/tests/demon-slayer-manifest.json")
  const PINATA_JWT = env("PINATA_JWT");
  if (!inputPath) throw new Error("INPUT_MANIFEST is required");
  if (!PINATA_JWT) throw new Error("PINATA_JWT is required");

  const raw = JSON.parse(await readFile(inputPath, "utf-8"));

  // Pre-validate only the shape we need for decisions (no Zod yet).
  if (!raw.e_model) throw new Error("Manifest is missing e_model (embedding model spec).");
  const eModel = {
    id: String(env("MODEL_ID", raw.e_model.id)),
    dim: Number(raw.e_model.dim),
    normalize: Boolean(raw.e_model.normalize),
    instruction: (raw.e_model.instruction ?? "e5") as "e5" | "none",
    mode: (raw.e_model.mode ?? "passage") as "query" | "passage",
  };

  const entries: any[] = Array.isArray(raw.entries) ? raw.entries : [];
  if (!entries.length) throw new Error("Manifest has no entries.");

  const hasExternal = !!(raw.vectors_uri || raw.vectors_index);
  const hasAllInline = entries.every((e) => e.embedding || e.embedding_b64);
  const hasAnyInline = entries.some((e) => e.embedding || e.embedding_b64);

  console.log("Input:", inputPath);
  console.log(`Entries=${entries.length} | external=${hasExternal} | inlineAny=${hasAnyInline} | inlineAll=${hasAllInline}`);
  console.log(`Embedder=${eModel.id} dim=${eModel.dim} norm=${eModel.normalize} instr=${eModel.instruction}`);

  const quant = pickQuant(env("QUANT", "fp16"));
  const clearInline = bool("CLEAR_INLINE", true);

  let vectorsBin: Uint8Array | null = null;
  let vectorsCID: string | null = null;
  let vectorsChecksum: string | null = null;

  if (hasExternal) {
    // Already externalized — we assume it's correct and just (re)pin manifest below.
    console.log("[pack] External vectors already referenced; not re-embedding.");
  } else if (hasAllInline) {
    // Externalize inline
    console.log("[pack] Externalizing inline embeddings → vectors.bin (in-memory)...");
    const vecs: Float32Array[] = entries.map((e, i) => {
      if (e.embedding_b64) return b64ToF32(e.embedding_b64);
      if (e.embedding) return new Float32Array(e.embedding);
      throw new Error(`Entry ${i} missing embedding despite inlineAll=true`);
    });

    // Sanity check dims
    if (!vecs.every((v) => v.length === eModel.dim)) {
      throw new Error(`Inline vectors dim mismatch vs e_model.dim=${eModel.dim}`);
    }

    vectorsBin = encodeVectorsBlob(vecs, quant);
    vectorsChecksum = sha256Hex(vectorsBin);

    // Upload vectors.bin
    vectorsCID = await pinFileToIPFS_JWT(vectorsBin, "vectors.bin", PINATA_JWT);
    console.log(`[ipfs] vectors CID: ${vectorsCID}`);

    if (clearInline) {
      for (const e of entries) {
        delete e.embedding;
        delete e.embedding_b64;
      }
    }
  } else {
    // Text-only → embed using remote server
    console.log("[pack] Embedding entries via remote /v1/embeddings ...");
    const base = env("EMBEDDINGS_BASE_URL");
    if (!base) throw new Error("EMBEDDINGS_BASE_URL is required for text-only embedding.");
    const texts: string[] = entries.map((e) => String(e.text ?? ""));
    if (texts.some((t) => t.length === 0)) throw new Error("All entries must have non-empty text.");

    const vecs = await remoteEmbedBatch(base, texts, {
      modelId: eModel.id,
      instruction: eModel.instruction,
      mode: eModel.mode,
      normalize: eModel.normalize,
    });
    if (!vecs.every((v) => v.length === eModel.dim)) {
      throw new Error(`Embedded dimension mismatch vs e_model.dim=${eModel.dim}`);
    }

    vectorsBin = encodeVectorsBlob(vecs, quant);
    vectorsChecksum = sha256Hex(vectorsBin);

    // Upload vectors.bin
    vectorsCID = await pinFileToIPFS_JWT(vectorsBin, "vectors.bin", PINATA_JWT);
    console.log(`[ipfs] vectors CID: ${vectorsCID}`);

    console.log("vectorsCheckSum",vectorsChecksum)
  }

  // Build final manifest with ipfs:// vectors (if we produced/require them)
  const finalManifest: any = {
    ...raw,
    entries, // possibly stripped inline vectors
  };

  if (!hasExternal) {
    if (!vectorsCID || !vectorsChecksum) {
      throw new Error("Internal error: vectors not produced for non-external manifest.");
    }
    finalManifest.vectors_uri = `ipfs://${vectorsCID}`;
    finalManifest.vectors_checksum = vectorsChecksum;
  }

  // Validate final manifest (NOW it contains vectors_uri, so it passes your superRefine)
  const parsed: InftManifestParsed = InftManifestSchema.parse(finalManifest);

  // Upload the final manifest JSON
  const manifestCID = await pinJSONToIPFS_JWT(parsed, PINATA_JWT);
  console.log(`[ipfs] manifest CID: ${manifestCID}`);

  const gw = env("PINATA_GATEWAY");
  if (gw) {
    const vecUrl = vectorsCID ? `${gw}/ipfs/${vectorsCID}` : "(unchanged)";
    const manUrl = `${gw}/ipfs/${manifestCID}`;
    console.log("\nGateway URLs:");
    if (vectorsCID) console.log("  vectors.bin:", vecUrl);
    console.log("  manifest   :", manUrl);
  }

  console.log("\nRESULT:");
  if (vectorsCID) console.log("  vectors_cid  :", vectorsCID);
  console.log("  manifest_cid :", manifestCID);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
