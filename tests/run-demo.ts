// Minimal demo that loads the sample manifest, creates an in-memory store,
// embeds a test query with MockEmbedder3D, and prints top hits.

import { loadManifest } from "../src/services/inft-loader-service";
import { PrecomputedVectorStore } from "../src/services/precomputed-store";
import { MockEmbedder3D, RemoteEmbedder } from "../src/services/embedding.service";
import { normalizeInPlace } from "../src/utils/cosine";
import path from "path/win32";
import { fileURLToPath } from "url";

async function main() {
  // 1) Load sample manifest (local file)
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
const manifestPath = path.join(__dirname, "sample-manifest.json");

const manifest = await loadManifest(manifestPath);


  // 2) Build in-memory store (re-normalize = false because sample is already unit vectors)
  const store = PrecomputedVectorStore.fromManifest(manifest, { reNormalize: false });
  console.log(`Loaded store: model=${store.model.id} dim=${store.model.dim} size=${store.size()}`);

  // 3) Choose an embedder for the QUERY
  //    - For this test: MockEmbedder3D (works with dim=3 sample data)
  //    - For real use: RemoteEmbedder("http://localhost:8080") etc.
  const embedder = new RemoteEmbedder("http://localhost:8080");
  // const embedder = new RemoteEmbedder(process.env.EMBED_URL ?? "http://localhost:8080");

  // 4) Embed the user query (unit-normalized)
  const query = "How do incentives work? and what feedback policy applies?";
  const [q] = await embedder.embed([query], { mode: "query", instruction: "e5", normalize: true });
  normalizeInPlace(q); // safety

  // 5) Search top-k
  const k = Number(process.env.K ?? 2);
  const hits = store.search(q, k);

  // 6) Print results
  console.log("\nQuery:", query);
  console.log("\nTop hits:");
  for (const h of hits) {
    console.log(`- ${h.id}  score=${h.score.toFixed(3)}  ${String(h.meta?.source)} :: ${h.text}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
