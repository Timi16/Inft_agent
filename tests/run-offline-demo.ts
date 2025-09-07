// Fully offline flow:
//  - vectors & manifest are local files (no http/ipfs)
//  - query embedding uses LocalEmbedder (no network)

import { loadManifest } from "../src/services/inft-loader-service";
import { ExternalVectorStore } from "../src/services/external-store";
import { LocalEmbedder } from "../src/services/local-embedder";
import { normalizeInPlace } from "../src/utils/cosine";
import { ensureModelContract } from "../src/services/contracts";
import { encodeVectorsBlob } from "../src/services/binary.vector";
import { writeFile, access, readFile } from "node:fs/promises";
import path from "path/win32";
import { fileURLToPath } from "url";

async function pickManifestPath(__dirname: string) {
  const external = path.join(__dirname, "sample-manifest.external.json");
  const inline = path.join(__dirname, "sample-manifest.json");
  try {
    await access(external);
    return external; // prefer external (already has vectors_uri)
  } catch {}
  return inline; // fallback to inline
}

async function ensureVectorsUri(manifest: any, __dirname: string) {
  if (manifest.vectors_uri) return manifest;

  // Inline manifest: build vectors.bin locally from entries and point vectors_uri to it
  const vectors: Float32Array[] = manifest.entries.map((e: any) => {
    if (Array.isArray(e.embedding)) return new Float32Array(e.embedding);
    if (e.embedding_b64) {
      const buf = Buffer.from(String(e.embedding_b64), "base64");
      return new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
    }
    throw new Error(`Entry ${e.id} has no embedding (embedding[] or embedding_b64)`);
  });

  const blob = encodeVectorsBlob(vectors, 0); // 0 = fp32
  const vecPath = path.join(__dirname, "vectors.bin");
  await writeFile(vecPath, Buffer.from(blob));
  // Optionally: add a checksum if your loader enforces it
  // const { sha256Hex } = await import("../src/utils/hash");
  // manifest.vectors_checksum = sha256Hex(new Uint8Array(blob));
  manifest.vectors_uri = vecPath;
  return manifest;
}

async function main() {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const manifestPath = await pickManifestPath(__dirname);
  const manifest = await loadManifest(manifestPath);
  await ensureVectorsUri(manifest, __dirname);

  // Build store from local vectors.bin
  const store = await ExternalVectorStore.fromManifestWithExternalVectors(manifest);
  console.log(`Loaded external store: model=${store.model.id} dim=${store.model.dim} size=${store.size()}`);

  // Local embedder (no network). Ensure MODEL_ID matches manifest.model.id if required.
  const embedder = new LocalEmbedder(process.env.MODEL_ID || manifest.model.id);

  const info = await embedder.info();
  ensureModelContract(manifest.model, info);

  const query = "Explain the feedback policy and incentives.";
  const [q] = await embedder.embed([query], { mode: "query", instruction: "e5", normalize: true });
  normalizeInPlace(q);

  const hits = store.search(q, Number(process.env.K || 2));
  console.log("\nTop hits:");
  for (const h of hits) console.log(`- ${h.id}  ${h.score.toFixed(3)}  :: ${h.text}`);
}

main().catch((e) => { console.error(e); process.exit(1); });