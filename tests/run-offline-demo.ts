// Fully offline flow:
//  - vectors & manifest are local files (no http/ipfs)
//  - query embedding uses LocalEmbedder (no network after first download)

import { loadManifest } from "../src/services/inft-loader-service";
import { ExternalVectorStore } from "../src/services/external-store";
import { LocalEmbedder } from "../src/services/local-embedder";
import { normalizeInPlace } from "../src/utils/cosine";
import { ensureModelContract } from "../src/services/contracts";
import path from "path/win32";
import { fileURLToPath } from "url";

async function main() {
  // Use the externalized manifest produced by pack-and-upload (edit path to match your file)
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const manifestPath = path.join(__dirname, "sample-manifest.json");
  
  const manifest = await loadManifest(manifestPath);
  if (manifest.vectors_uri?.startsWith("ipfs://") === false && !manifest.vectors_uri?.startsWith("http")) {
    // assume it's already a local path
  }

  // Load external vectors (supports file://, path, ipfs:// via gateway, or http)
  const store = await ExternalVectorStore.fromManifestWithExternalVectors(manifest);
  console.log(`Loaded external store: model=${store.model.id} dim=${store.model.dim} size=${store.size()}`);

  // Offline embedder (local); set MODEL_ID env to match manifest.model.id
  const embedder = new LocalEmbedder(process.env.MODEL_ID || manifest.model.id);

  // Contract check (id + dim)
  const info = await embedder.info();
  ensureModelContract(manifest.model, info);

  // Embed query and search
  const query = "Explain the feedback policy and incentives.";
  const [q] = await embedder.embed([query], { mode: "query", instruction: "e5", normalize: true });
  normalizeInPlace(q);

  const hits = store.search(q, Number(process.env.K || 2));
  console.log("\nTop hits:");
  for (const h of hits) console.log(`- ${h.id}  ${h.score.toFixed(3)}  :: ${h.text}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
