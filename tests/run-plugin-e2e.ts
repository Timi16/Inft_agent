// tests/run-plugin-e2e.ts
/**
 * End-to-end test for the iNFT + Eliza + 0G plugin services (no mocks).
 *
 * Steps:
 * 1) Load real iNFT manifest (ipfs://, https://, or file://).
 * 2) Load external vectors (verifies checksum if present).
 * 3) Choose embedder (REMOTE or LOCAL) based on env, and verify model contract.
 * 4) Embed a real query -> cosine search top-k.
 * 5) (Optional) Call 0G via broker and print reply.
 *
 * IMPORTANT: The manifest.model.id/dim must match the query embedder model.
 * Using RemoteEmbedder does NOT bypass that requirement.
 */

import "dotenv/config";
import { logger } from "@elizaos/core";

import { loadManifest } from "../src/services/inft-loader-service";
import { ExternalVectorStore } from "../src/services/external-store";
import { normalizeInPlace } from "../src/utils/cosine";
import { ensureModelContract } from "../src/services/contracts";
import { LocalEmbedder } from "../src/services/local-embedder";
import { RemoteEmbedder } from "../src/services/embedding.service";

import { OgBrokerService } from "../src/eliza/services/og-broker.service";

// ---- Small helpers ----
const dequote = (s?: string | null) =>
  (s ?? "").trim().replace(/^[\'\"]+|[\'\"]+$/g, "");

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`Missing required env: ${name}`);
  return v;
}

type RuntimeLike = {
  getSetting: (k: string) => string | undefined;
  getService: <T = unknown>(_name: any) => T | null;
};
const runtime: RuntimeLike = {
  getSetting: (k) => process.env[k],
  getService: () => null,
};

async function main() {
  logger.info("=== iNFT + 0G plugin E2E test ===");

  // 1) Load manifest
  const manifestUri = requireEnv("INFT_MANIFEST_URI");
  const manifest = await loadManifest(manifestUri);
  if (!manifest.vectors_uri) {
    throw new Error("Manifest has no vectors_uri. Use your packer to externalize vectors first.");
  }
  logger.info(`Manifest loaded. model=${manifest.model.id} dim=${manifest.model.dim}`);

  // 2) Load vectors
  const gateway = process.env.PINATA_GATEWAY;
  const store = await ExternalVectorStore.fromManifestWithExternalVectors(manifest, gateway);
  logger.info(`Vectors loaded: ${store.size()} entries.`);

  // 3) Pick embedder: REMOTE (your server) or LOCAL (xenova)cl
  const useRemote = /^1|true$/i.test(String(process.env.USE_REMOTE_EMBEDDER || "0"));
  const modelIdEnv = dequote(process.env.MODEL_ID) || manifest.model.id;

  // Hard contract check: manifest.model.id must equal the embedder model id you intend to use.
  // If not, we fail here with a clear message so you regenerate the manifest correctly.
  if (modelIdEnv !== manifest.model.id) {
    throw new Error(
      `Model contract mismatch:\n` +
      `  manifest.model.id = ${manifest.model.id}\n` +
      `  runtime MODEL_ID   = ${modelIdEnv}\n\n` +
      `Fix: Rebuild the manifest with the same model you use for queries, OR set MODEL_ID to ${manifest.model.id}.\n` +
      `Note: Using RemoteEmbedder does NOT bypass this — query and passage embeddings must be in the same vector space.`
    );
  }

  // Build the embedder
  let embedOne: (q: string) => Promise<Float32Array>;
  if (useRemote) {
    const base = requireEnv("EMBEDDINGS_BASE_URL"); // e.g., http://localhost:8080
    const remote = new RemoteEmbedder(base);
    embedOne = async (q: string) => {
      const vecs = await remote.embed([q], {
        mode: "query",
        instruction: "e5",
        normalize: true,
      });
      const v = vecs[0];
      if (v.length !== manifest.model.dim) {
        throw new Error(
          `Remote embedder dim ${v.length} != manifest dim ${manifest.model.dim}. ` +
          `Ensure your embedding server runs the same model (${manifest.model.id}).`
        );
      }
      return v;
    };
  } else {
    const local = new LocalEmbedder(modelIdEnv);
    const info = await local.info(); // warms model & sets dim
    ensureModelContract(manifest.model, info); // id + dim check
    embedOne = async (q: string) => {
      const [v] = await local.embed([q], {
        mode: "query",
        instruction: "e5",
        normalize: true,
      });
      return v;
    };
  }

  // 4) Query -> search
  const query =
    process.env.QUERY ||
    "Give me a concise explanation of the feedback and incentive policies with key points.";
  const q = await embedOne(query);
  normalizeInPlace(q);

  const k = Number(process.env.INFT_TOP_K || 4);
  const hits = store.search(q, k);
  if (!hits.length) throw new Error("No retrieval hits. Check entries/text and your query.");

  logger.info("Top retrieval hits:");
  hits.forEach((h, i) => console.log(`  (${i + 1}) score=${h.score.toFixed(3)} :: ${h.text}`));

  // Build prompt with retrieved context
  const ctx =
    "### Retrieved Knowledge\n" +
    hits.map((h, i) => `(${i + 1}) [${h.score.toFixed(3)}] ${h.text}`).join("\n");
  const prompt = [
    `You are ${manifest.character.name}.`,
    manifest.character.system ? `System: ${manifest.character.system}` : "",
    ctx,
    "### Task",
    query,
    "",
    "Answer concisely and cite snippets (1..k) inline like [1],[2] where relevant.",
  ]
    .filter(Boolean)
    .join("\n");

  // 5) 0G call (optional)
  if (!process.env.OG_RPC_URL && !process.env.EVM_RPC) {
    console.warn("\n[SKIP] 0G inference: set OG_RPC_URL/EVM_RPC and OG_PRIVATE_KEY to run the call.");
    console.log("\nComposed prompt preview:\n", prompt);
    return;
  }
  // Allow both naming styles
  process.env.OG_RPC_URL = process.env.OG_RPC_URL || process.env.EVM_RPC;
  process.env.OG_PRIVATE_KEY = process.env.OG_PRIVATE_KEY || process.env.PRIVATE_KEY;

  const og = await (OgBrokerService as any).start(runtime);
  const res = await og.infer({ prompt, modelHint: process.env.OG_MODEL_HINT });

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
