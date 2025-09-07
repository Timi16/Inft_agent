/**
 * End-to-end test for the iNFT + Eliza + 0G plugin services.
 *
 * What it does (no mocks):
 * 1) Loads a real iNFT manifest (ipfs://, https://, or file://).
 * 2) Verifies manifest has vectors_uri and checksum (if present).
 * 3) Loads the external vectors, builds an in-memory store.
 * 4) Uses LocalEmbedder (offline) to embed a real query, then searches top-k.
 * 5) Sends a composed prompt (with retrieved snippets) to 0G via broker (wallet-signed).
 *
 * Run:
 *   npx tsx test/run-plugin-e2e.ts
 *
 * Required ENV:
 *   INFT_MANIFEST_URI=ipfs://CID/manifest.json | file:///.../manifest.external.json
 *   OG_RPC_URL=...  OG_PRIVATE_KEY=0x...  (to actually hit 0G inference)
 * Optional ENV:
 *   MODEL_ID=...  IPFS_GATEWAY=...  INFT_TOP_K=4  QUERY="..."
 *   OG_MODEL_HINT="llama"  TRANSFORMERS_CACHE=/path/to/cache
 */

import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

// Eliza core types (for basic runtime wiring)
import { logger } from "@elizaos/core";

// Our services from the plugin you added
import { InftKnowledgeService } from "../src/eliza/services/inft-knowledge.service";
import { OgBrokerService } from "../src/eliza/services/og-broker.service";
import "dotenv/config";
// Utilities we already have
import { loadManifest } from "../src/services/inft-loader-service";
import { ExternalVectorStore } from "../src/services/external-store";
import { LocalEmbedder } from "../src/services/local-embedder";
import { ensureModelContract } from "../src/services/contracts";
import { normalizeInPlace } from "../src/utils/cosine";

// ---- Minimal runtime shim (real network/services, no mocks) ----
type RuntimeLike = {
  getSetting: (k: string) => string | undefined;
  getService: <T = unknown>(_name: any) => T | null;
};
const runtime: RuntimeLike = {
  getSetting: (k) => process.env[k],
  getService: () => null,
};

// ---- Helpers ----
function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`Missing required env: ${name}`);
  return v;
}

async function main() {
  logger.info("=== iNFT + 0G plugin E2E test ===");

  // 1) Load manifest
  const manifestUri = process.env.INFT_MANIFEST_URI;
  if (!manifestUri) {
    throw new Error(
      "INFT_MANIFEST_URI is required. Example: ipfs://<CID>/manifest.json or file:///abs/path/manifest.external.json"
    );
  }
  const manifest = await loadManifest(manifestUri);

  if (!manifest.vectors_uri) {
    throw new Error(
      "Manifest has no vectors_uri. Run your packer to externalize vectors, or provide a manifest with vectors_uri."
    );
  }

  logger.info(`Manifest loaded. model=${manifest.model.id} dim=${manifest.model.dim}`);

  // 2) Build ExternalVectorStore (fetches ipfs/http/file and verifies checksum if present)
  const store = await ExternalVectorStore.fromManifestWithExternalVectors(
    manifest,
    process.env.PINATA_GATEWAY
  );
  logger.info(`Vectors loaded: ${store.size()} entries.`);

  // 3) Local query embedding (offline). Ensure model contract matches.
  const embedder = new LocalEmbedder(process.env.MODEL_ID);
  const info = await embedder.info();
  ensureModelContract(manifest.model, info);

  const query ="Give me a concise explanation of the feedback and incentive policies with key points.";
  const [q] = await embedder.embed([query], {
    mode: "query",
    instruction: "e5",
    normalize: true,
  });
  normalizeInPlace(q);

  const k = Number(process.env.INFT_TOP_K);
  const hits = store.search(q, k);

  if (!hits.length) {
    throw new Error("No retrieval hits. Check your manifest entries/text and query.");
  }

  logger.info("Top retrieval hits:");
  hits.forEach((h, i) =>
    console.log(`  (${i + 1}) score=${h.score.toFixed(3)} :: ${h.text}`)
  );

  const retrievedBlock =
    "### Retrieved Knowledge\n" +
    hits.map((h, i) => `(${i + 1}) [${h.score.toFixed(3)}] ${h.text}`).join("\n");

  const prompt = [
    `You are ${manifest.character.name}.`,
    manifest.character.system ? `System: ${manifest.character.system}` : "",
    retrievedBlock,
    "### Task",
    query,
    "",
    "Answer concisely and cite snippets (1..k) inline like [1],[2] where relevant.",
  ]
    .filter(Boolean)
    .join("\n");

  const missingOg = !process.env.OG_RPC_URL || !process.env.OG_PRIVATE_KEY;
  if (missingOg) {
    console.warn(
      "\n[SKIP] OG inference: set OG_RPC_URL and OG_PRIVATE_KEY to run a real 0G call."
    );
    console.log("\nComposed prompt preview:\n", prompt);
    process.exit(0);
  }

  // OgBrokerService requires a runtime that exposes env; we pass our shim
  const og = await (OgBrokerService as any).start(runtime);
  const res = await og.infer({
    prompt,
    modelHint: process.env.OG_MODEL_HINT,
  });

  console.log("\n=== 0G Model Reply ===\n");
  console.log(res.text);
  console.log("\n--- meta ---");
  console.log(`provider: ${res.provider}`);
  console.log(`model:    ${res.model}`);
  if (res.chatID) console.log(`chatID:   ${res.chatID}`);

  logger.info("E2E test completed.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
