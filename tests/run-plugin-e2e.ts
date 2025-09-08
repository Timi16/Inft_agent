// tests/run-plugin-e2e.ts
import "dotenv/config";
import { logger } from "@elizaos/core";

import { loadManifest } from "../src/services/inft-loader-service";
import { ExternalVectorStore } from "../src/services/external-store";
import { normalizeInPlace } from "../src/utils/cosine";
import { ensureModelContract } from "../src/services/contracts";
import { LocalEmbedder } from "../src/services/local-embedder";
import { RemoteEmbedder } from "../src/services/embedding.service";
import { OgBrokerService } from "../src/eliza/services/og-broker.service";

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

  // 1) Manifest & vectors
  const manifestUri = requireEnv("INFT_MANIFEST_URI");
  const manifest = await loadManifest(manifestUri);
  if (!manifest.vectors_uri) throw new Error("Manifest has no vectors_uri.");
  logger.info(`Manifest loaded. model=${manifest.model.id} dim=${manifest.model.dim}`);

  const gateway = process.env.PINATA_GATEWAY;
  const store = await ExternalVectorStore.fromManifestWithExternalVectors(manifest, gateway);
  logger.info(`Vectors loaded: ${store.size()} entries.`);

  // 2) Embedder (REMOTE or LOCAL)
  const useRemote = /^1|true$/i.test(String(process.env.USE_REMOTE_EMBEDDER || "0"));
  const modelIdEnv = dequote(process.env.MODEL_ID) || manifest.model.id;

  if (modelIdEnv !== manifest.model.id) {
    throw new Error(
      `Model contract mismatch:\n` +
      `  manifest.model.id = ${manifest.model.id}\n` +
      `  runtime MODEL_ID   = ${modelIdEnv}\n`
    );
  }

  let embedOne: (q: string) => Promise<Float32Array>;
  if (useRemote) {
    const base = requireEnv("EMBEDDINGS_BASE_URL");
    const remote = new RemoteEmbedder(base);
    embedOne = async (q: string) => {
      const vecs = await remote.embed([q], { mode: "query", instruction: "e5", normalize: true });
      const v = vecs[0];
      if (v.length !== manifest.model.dim) {
        throw new Error(`Remote dim ${v.length} != manifest dim ${manifest.model.dim}`);
      }
      return v;
    };
  } else {
    const local = new LocalEmbedder(modelIdEnv);
    const info = await local.info();
    ensureModelContract(manifest.model, info);
    embedOne = async (q: string) => {
      const [v] = await local.embed([q], { mode: "query", instruction: "e5", normalize: true });
      return v;
    };
  }

  // 3) Query → search
  const query =
    process.env.QUERY ||
    "Give me a concise explanation of the feedback and incentive policies with key points.";
  const q = await embedOne(query);
  normalizeInPlace(q);

  const k = Number(process.env.INFT_TOP_K || 4);
  const hits = store.search(q, k);
  if (!hits.length) throw new Error("No retrieval hits.");

  logger.info("Top retrieval hits:");
  hits.forEach((h, i) => console.log(`  (${i + 1}) score=${h.score.toFixed(3)} :: ${h.text}`));

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

  // 4) 0G call (requires EVM_RPC & PRIVATE_KEY)
  if (!process.env.EVM_RPC || !process.env.PRIVATE_KEY) {
    console.warn("\n[SKIP] 0G inference: set EVM_RPC and PRIVATE_KEY.");
    console.log("\nComposed prompt preview:\n", prompt);
    return;
  }

  const og = await (OgBrokerService).start(runtime as any);
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
