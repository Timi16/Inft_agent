import "dotenv/config";
import { loadManifest } from "../src/services/inft-loader-service";
import { ExternalVectorStore } from "../src/services/external-store";
import { LocalEmbedder } from "../src/services/local-embedder";
import { normalizeInPlace } from "../src/utils/cosine";
import { ensureModelContract } from "../src/services/contracts";
import { encodeVectorsBlob } from "../src/services/binary.vector";
import { writeFile, access } from "node:fs/promises";
import path from "path/win32";
import { fileURLToPath } from "url";

async function pickManifestPath(__dirname: string) {
  const external = path.join(__dirname, "sample-manifest.external.json");
  const inline = path.join(__dirname, "sample-manifest.json");
  try { await access(external); return external; } catch {}
  return inline;
}

async function upgradeManifestToRealModel(manifest: any, embedder: LocalEmbedder, outDir: string) {
  const info = await embedder.info(); // { id, dim, normalize? }
  const idsMatch = !!(info.id && manifest.model.id && info.id === manifest.model.id);
  const dimsMatch = info.dim === manifest.model.dim;

  if (idsMatch && dimsMatch && manifest.vectors_uri) return { manifest, changed: false };

  // Re-embed entries with the real model in PASSAGE mode
  const texts = manifest.entries.map((e: any) => e.text);
  const vecs = await embedder.embed(texts, { mode: "passage", instruction: "e5", normalize: true });
  vecs.forEach(normalizeInPlace);

  const blob = encodeVectorsBlob(vecs, 0); // 0 = fp32
  const vecPath = path.join(outDir, "vectors.bin");
  await writeFile(vecPath, Buffer.from(blob));

  // Update manifest: model id/dim and vectors_uri; drop inline embeddings
  manifest.model.id = info.id;
  manifest.model.dim = info.dim;
  manifest.model.normalize = true;
  manifest.model.instruction = "e5"; // keep consistent with how you embed
  manifest.vectors_uri = vecPath;
  manifest.entries = manifest.entries.map(({ embedding, embedding_b64, ...rest }: any) => rest);

  return { manifest, changed: true };
}

async function main() {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const manifestPath = await pickManifestPath(__dirname);
  let manifest: any = await loadManifest(manifestPath);

  // Choose real model (or set via $env:MODEL_ID in PowerShell)
  const modelId = process.env.MODEL_ID || "Xenova/bge-small-en-v1.5";
  const embedder = new LocalEmbedder(modelId);

  // If manifest doesn't match embedder, upgrade it on the fly
  const { manifest: upgraded, changed } = await upgradeManifestToRealModel(manifest, embedder, __dirname);
  manifest = upgraded;

  // After upgrade, contract must pass
  const info = await embedder.info();
  ensureModelContract(manifest.model, info);

  // Build store from external vectors.bin (local path)
  const store = await ExternalVectorStore.fromManifestWithExternalVectors(manifest);
  console.log(`Loaded external store: model=${store.model.id} dim=${store.model.dim} size=${store.size()}`);

  // Query with the real model (QUERY mode)
  const query = "Explain the feedback policy and incentives.";
  const [q] = await embedder.embed([query], { mode: "query", instruction: "e5", normalize: true });
  normalizeInPlace(q);

  const hits = store.search(q, Number(process.env.K || 2));
  console.log("\nTop hits:");
  for (const h of hits) console.log(`- ${h.id}  ${h.score.toFixed(3)}  :: ${h.text}`);

  if (changed) console.log("\n(Manifest upgraded to real model; vectors.bin written locally.)");
}

main().catch((e) => { console.error(e); process.exit(1); });