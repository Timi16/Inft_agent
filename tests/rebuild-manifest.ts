/**
 * tests/rebuild-manifest.ts
 *
 * Rebuild an iNFT manifest so its model contract matches your query embedder.
 * - Re-embeds all entries (passage mode, e5, normalize:true)
 * - Writes external vectors blob (fp16/fp32) to OUTPUT_DIR/vectors.bin
 * - Computes sha256 checksum and writes manifest.external.json with vectors_uri
 * - (Optional) uploads vectors + manifest to Pinata and prints ipfs:// URIs
 *
 * USAGE (Windows CMD examples):
 *   set INPUT_MANIFEST=file:///C:/Users/A/Documents/inft-agent/sample-manifest.json
 *   set OUTPUT_DIR=C:\Users\A\Documents\inft-agent\dist
 *   set USE_REMOTE_EMBEDDER=1
 *   set EMBEDDINGS_BASE_URL=http://127.0.0.1:8080
 *   set MODEL_ID=Xenova/bge-small-en-v1.5
 *   set QUANT=fp16
 *   set UPLOAD=1
 *   set PINATA_GATEWAY=https://violet-deliberate-fly-257.mypinata.cloud
 *   set PINATA_JWT=eyJhbGciOi...
 *
 *   npx tsx tests/rebuild-manifest.ts
 */

import "dotenv/config";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { createHash } from "node:crypto";

// reuse your SDK
import { loadManifest } from "../src/services/inft-loader-service";
import { normalizeInPlace } from "../src/utils/cosine";
import { LocalEmbedder } from "../src/services/local-embedder";
import { RemoteEmbedder } from "../src/services/embedding.service";

// Try to reuse your encoder; otherwise use fallback implemented below
type QuantStr = "fp16" | "fp32";
type EncodeCompat = (vectors: Float32Array[], dim: number, quant: QuantStr) => Uint8Array;

// default to our local fallback; we'll override if the project exports one
let encodeVectorsBlob: EncodeCompat = encodeVectorsBlobFallback;

try {
  const mod: any = await import("../src/services/binary.vector");
  if (typeof mod.encodeVectorsBlob === "function") {
    const inner = mod.encodeVectorsBlob;

    // Wrap to a common signature (vectors, dim, "fp16"|"fp32")
    encodeVectorsBlob = (vectors: Float32Array[], dim: number, quant: QuantStr): Uint8Array => {
      // If the exported function already supports (vectors, dim, quantString), use it
      if (inner.length >= 3) {
        // e.g. inner(vectors, dim, "fp16"|"fp32")
        return inner(vectors, dim, quant) as Uint8Array;
      }
      // Otherwise map to its (vectors, QuantCode) form
      const QuantCode = mod.QuantCode || { FP32: 1, FP16: 2 }; // sensible defaults
      const qc = quant === "fp32" ? QuantCode.FP32 : QuantCode.FP16;
      return inner(vectors, qc) as Uint8Array;
    };
  }
} catch {
  // keep fallback
}


// ----------------- helpers -----------------
const dequote = (s?: string | null) =>
  (s ?? "").trim().replace(/^[\'\"]+|[\'\"]+$/g, "");

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`Missing required env: ${name}`);
  return v;
}

function isUrl(u: string) {
  return /^https?:\/\//i.test(u) || /^ipfs:\/\//i.test(u) || /^file:\/\//i.test(u);
}

function resolveHttpFromIPFS(ipfsUri: string, gateway: string) {
  const gw = (gateway || "https://ipfs.io/ipfs/").replace(/\/+$/, "");
  const path = ipfsUri.replace(/^ipfs:\/\//i, "");
  return `${gw}/${path}`;
}

async function fetchJsonSmart(uriOrPath: string, gateway?: string) {
  if (isUrl(uriOrPath)) {
    const url = uriOrPath.startsWith("ipfs://")
      ? resolveHttpFromIPFS(
          uriOrPath,
          gateway || process.env.PINATA_GATEWAY || process.env.IPFS_GATEWAY || ""
        )
      : uriOrPath;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Fetch failed: ${r.status} ${r.statusText} for ${url}`);
    return r.json();
  }
  const buf = await readFile(uriOrPath.replace(/^file:\/\//i, ""));
  return JSON.parse(buf.toString("utf-8"));
}

function sha256Hex(bytes: Uint8Array | ArrayBufferLike) {
  const b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes as ArrayBufferLike);
  const h = createHash("sha256");
  h.update(b);
  return "sha256:" + h.digest("hex");
}

// ---- float16 conversion (fallback) ----
function float32ToFloat16Buffer(f32: Float32Array): Uint8Array {
  const u16 = new Uint16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const x = f32[i];
    const sign = x < 0 ? 1 : 0;
    const ax = Math.abs(x);
    if (ax === 0) {
      u16[i] = sign << 15;
      continue;
    }
    if (ax >= 65504) {
      u16[i] = (sign << 15) | 0x7bff;
      continue;
    }
    if (ax < 6.103515625e-5) {
      u16[i] = (sign << 15) | Math.round(ax / 5.960464477539063e-8);
      continue;
    }
    let exp = Math.floor(Math.log2(ax));
    let mant = ax / Math.pow(2, exp) - 1;
    exp += 15;
    mant = Math.round(mant * 1024);
    u16[i] = (sign << 15) | (exp << 10) | (mant & 0x3ff);
  }
  return new Uint8Array(u16.buffer);
}

// ---- vectors blob encoder (fallback if your project didn't export one) ----
function encodeVectorsBlobFallback(
  vectors: Float32Array[],
  dim: number,
  quant: "fp16" | "fp32"
): Uint8Array {
  // Layout:
  // magic "VECB" (4 bytes)
  // version u8 = 1
  // dtype  u8 = 1(fp32) | 2(fp16)
  // pad    u16 = 0
  // dim    u32
  // count  u32
  // data   contiguous row-major
  const count = vectors.length;
  const headerSize = 4 + 1 + 1 + 2 + 4 + 4;

  let dataBytes: Uint8Array;
  if (quant === "fp16") {
    const total = dim * count;
    const tmp = new Float32Array(total);
    let o = 0;
    for (let i = 0; i < count; i++) {
      tmp.set(vectors[i], o);
      o += dim;
    }
    dataBytes = float32ToFloat16Buffer(tmp);
  } else {
    const total = dim * count;
    const tmp = new Float32Array(total);
    let o = 0;
    for (let i = 0; i < count; i++) {
      tmp.set(vectors[i], o);
      o += dim;
    }
    dataBytes = new Uint8Array(tmp.buffer);
  }

  const buf = new Uint8Array(headerSize + dataBytes.byteLength);
  const dv = new DataView(buf.buffer);

  // "VECB"
  buf.set([0x56, 0x45, 0x43, 0x42], 0);
  dv.setUint8(4, 1); // version
  dv.setUint8(5, quant === "fp32" ? 1 : 2);
  dv.setUint16(6, 0, true);
  dv.setUint32(8, dim, true);
  dv.setUint32(12, count, true);
  buf.set(dataBytes, headerSize);

  return buf;
}
if (!encodeVectorsBlob) encodeVectorsBlob = encodeVectorsBlobFallback;

// ---- pinata upload helpers (optional) ----
/** Get a guaranteed ArrayBuffer (not ArrayBufferLike) for Blob/File. */
function toArrayBuffer(bytes: Uint8Array | ArrayBufferLike): ArrayBuffer {
  if (bytes instanceof Uint8Array) {
    // Copy into a fresh ArrayBuffer (ensures ArrayBuffer, not SharedArrayBuffer)
    const out = new Uint8Array(bytes.byteLength);
    out.set(bytes);
    return out.buffer; // ArrayBuffer
  }
  // bytes is ArrayBufferLike; copy into ArrayBuffer via a Uint8Array view
  const u8 = new Uint8Array(bytes as ArrayBufferLike);
  const out = new Uint8Array(u8.byteLength);
  out.set(u8);
  return out.buffer; // ArrayBuffer
}

async function pinataUploadBytes(name: string, bytes: Uint8Array): Promise<string> {
  const jwt = process.env.PINATA_JWT;
  if (!jwt) throw new Error("PINATA_JWT not set");

  const form = new FormData();
  const ab = toArrayBuffer(bytes);
  const file = new File([ab], name, { type: "application/octet-stream" });
  form.append("file", file, name);
  form.append(
    "pinataMetadata",
    new Blob([JSON.stringify({ name })], { type: "application/json" })
  );

  const r = await fetch("https://api.pinata.cloud/pinning/pinFileToIPFS", {
    method: "POST",
    headers: { Authorization: `Bearer ${jwt}` },
    body: form,
  });
  if (!r.ok) throw new Error(`pinFileToIPFS failed: ${r.status} ${r.statusText}`);
  const j: any = await r.json();
  return j.IpfsHash; // CID
}

async function pinataUploadJSON(name: string, json: any): Promise<string> {
  const jwt = process.env.PINATA_JWT;
  if (!jwt) throw new Error("PINATA_JWT not set");
  const r = await fetch("https://api.pinata.cloud/pinning/pinJSONToIPFS", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${jwt}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ pinataMetadata: { name }, pinataContent: json }),
  });
  if (!r.ok) throw new Error(`pinJSONToIPFS failed: ${r.status} ${r.statusText}`);
  const j: any = await r.json();
  return j.IpfsHash; // CID
}

// ----------------- main -----------------
async function main() {
  const inputUri = process.env.INPUT_MANIFEST || process.env.INFT_MANIFEST_URI;
  if (!inputUri) {
    throw new Error(
      "Set INPUT_MANIFEST (or INFT_MANIFEST_URI) to the source manifest (ipfs://, http(s):// or file:///path.json)"
    );
  }

  const outDir = process.env.OUTPUT_DIR || resolve(process.cwd(), "dist");
  const quant = (dequote(process.env.QUANT) as "fp16" | "fp32") || "fp16";
  const useRemote = /^1|true$/i.test(String(process.env.USE_REMOTE_EMBEDDER || "0"));
  const modelId = dequote(process.env.MODEL_ID) || "Xenova/bge-small-en-v1.5";

  await mkdir(outDir, { recursive: true });

  // 1) Load source manifest
  const src: any = await fetchJsonSmart(
    inputUri,
    process.env.PINATA_GATEWAY || process.env.IPFS_GATEWAY
  );
  if (!Array.isArray(src.entries) || src.entries.length === 0) {
    throw new Error("Manifest has no entries to embed.");
  }

  // 2) Build embedder
  let embedBatch: (texts: string[]) => Promise<Float32Array[]>;
  if (useRemote) {
    const base = requireEnv("EMBEDDINGS_BASE_URL"); // e.g., http://127.0.0.1:8080
    const remote = new RemoteEmbedder(base);
    embedBatch = async (texts: string[]) => {
      // remote returns number[][]
      const vecs = (await remote.embed(texts, {
        mode: "passage",
        instruction: "e5",
        normalize: true,
      })) as unknown as number[][];
      const out: Float32Array[] = [];
      for (const row of vecs) out.push(new Float32Array(row));
      return out;
    };
  } else {
    const local = new LocalEmbedder(modelId);
    await local.info(); // warm
    embedBatch = async (texts: string[]) => {
      const arr = await local.embed(texts, {
        mode: "passage",
        instruction: "e5",
        normalize: true,
      });
      return arr; // Float32Array[]
    };
  }

  // 3) Re-embed passages
  const texts = src.entries.map((e: any) => e.text);
  const vectors = await embedBatch(texts);
  const dim = vectors[0].length;

  // 4) Update manifest model contract to match the embedder used
  const rebuilt = {
    ...src,
    model: {
      id: modelId,
      dim,
      normalize: true,
      instruction: "e5",
      mode: "passage",
      quantization: quant,
    },
    // remove any inline embeddings in entries
    entries: src.entries.map((e: any) => {
      const { embedding, embedding_b64, ...rest } = e;
      return rest;
    }),
  };

  // 5) Encode external vectors blob
  const blob = (encodeVectorsBlob as any)(vectors, dim, quant);
  const checksum = sha256Hex(blob);

  // 6) Write local files
  const vectorsPath = resolve(outDir, "vectors.bin");
  const manifestPath = resolve(outDir, "manifest.external.json");

  await writeFile(vectorsPath, blob);
  const fileUri = `file:///${vectorsPath.replace(/\\/g, "/")}`;

  const externalManifest = {
    ...rebuilt,
    vectors_uri: fileUri,
    vectors_checksum: checksum,
  };
  await writeFile(
    manifestPath,
    Buffer.from(JSON.stringify(externalManifest, null, 2), "utf-8")
  );

  console.log("✔ Wrote:", manifestPath);
  console.log("✔ Wrote:", vectorsPath);
  console.log("checksum:", checksum);

  // 7) Optional upload to Pinata
  const doUpload = /^1|true$/i.test(String(process.env.UPLOAD || "0"));
  if (doUpload) {
    console.log("\nUploading to Pinata…");
    const vecCID = await pinataUploadBytes("vectors.bin", blob);
    const vectorsIpfsUri = `ipfs://${vecCID}`;

    const manifestForIpfs = {
      ...rebuilt,
      vectors_uri: vectorsIpfsUri,
      vectors_checksum: checksum,
    };
    const manCID = await pinataUploadJSON("manifest.external.json", manifestForIpfs);
    const manifestIpfsUri = `ipfs://${manCID}`;

    const gateway = process.env.PINATA_GATEWAY || "https://ipfs.io/ipfs/";
    const manifestHttp = resolveHttpFromIPFS(manifestIpfsUri, gateway);
    const vectorsHttp = resolveHttpFromIPFS(vectorsIpfsUri, gateway);

    console.log("✔ Uploaded vectors:", vectorsIpfsUri);
    console.log("✔ Uploaded manifest:", manifestIpfsUri);
    console.log("HTTP (gateway):");
    console.log("  manifest:", manifestHttp);
    console.log("  vectors :", vectorsHttp);

    console.log("\nNext steps:");
    console.log("- Set INFT_MANIFEST_URI to:", manifestIpfsUri);
    console.log("- Ensure MODEL_ID is:", modelId);
    console.log("- Re-run: npx tsx tests/run-plugin-e2e.ts");
  } else {
    console.log("\nUpload disabled. To upload, set UPLOAD=1 and PINATA_JWT.");
    console.log("Next steps:");
    console.log("- Set INFT_MANIFEST_URI to:", `file:///${manifestPath.replace(/\\/g, "/")}`);
    console.log("- Ensure MODEL_ID is:", modelId);
    console.log("- Re-run: npx tsx tests/run-plugin-e2e.ts");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
