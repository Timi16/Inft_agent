import { logger } from "@elizaos/core";
import { loadManifest } from "../src/services/inft-loader-service";
import { ExternalVectorStore } from "../src/services/external-store";
import { normalizeInPlace } from "../src/utils/cosine";
import { RemoteEmbedder } from "../src/services/embedding.service";
import { Wallet, JsonRpcProvider } from "ethers";
import { OgBrokerService } from "../src/eliza/services/og-broker.service";

const GATEWAY = "https://violet-deliberate-fly-257.mypinata.cloud";
const MANIFEST_CID = "QmTckaLnmAyDLyhsSQqGwkVRmYqyVcxW5Cjptz9fFgjPqn";
const EMBEDDER_BASE_URL = "http://170.75.163.164:4916";

const EVM_RPC = "https://evmrpc-testnet.0g.ai";
const PRIVATE_KEY = "0x5c3a638856b1b708f5e75d4831d700208eef4c2484eb59dd817ce25cdd918ad4";

const runtime: any = { getSetting: () => undefined, getService: () => null };

const requireNonEmpty = (name: string, v: string) => {
  if (!v || !v.trim()) throw new Error(`Missing required ${name}`);
  return v.trim();
};

const gatewayJoin = (base: string, cidOrPath: string) =>
  `${base.replace(/\/+$/, "")}/ipfs/${cidOrPath.replace(/^\/+/, "")}`;


async function main() {
  logger.info("=== iNFT + Remote Embedder + 0G (IPFS-only) ===");

  const gateway = requireNonEmpty("GATEWAY", GATEWAY);
  const manifestCid = requireNonEmpty("MANIFEST_CID", MANIFEST_CID);
  const embedderBase = requireNonEmpty("EMBEDDER_BASE_URL", EMBEDDER_BASE_URL);

  // 1) Manifest
  const manifestUrl = gatewayJoin(gateway, manifestCid);
  const manifest = await loadManifest(manifestUrl, { gateway });
  logger.info(`Manifest loaded. model=${manifest.e_model.id} dim=${manifest.e_model.dim}`);

  const m: any = manifest;
  if (!m.vectors_uri) throw new Error("Manifest missing vectors_uri.");
  // 2) Checksum
  // const vecUrl = gatewayJoin(gateway, extractCid(String(m.vectors_uri)));
  // const vecBytes = await fetchBytes(vecUrl);
  // const got = sha256Hex(vecBytes);
  // if (got !== m.vectors_checksum) {
  //   throw new Error(`vectors_checksum mismatch:\n  manifest: ${m.vectors_checksum}\n  computed: ${got}`);
  // }

  // 3) Store + embed
  const store = await ExternalVectorStore.fromManifestWithExternalVectors(manifest, gateway);
  logger.info(`Vectors loaded: ${store.size()} entries.`);

  const embedder = new RemoteEmbedder(embedderBase);
  const query = "What is Demon Slayer about?";
  const [q] = await embedder.embed([query], { mode: "query", instruction: "e5", normalize: true });
  if (!q || q.length !== manifest.e_model.dim) throw new Error("Query vector dim mismatch.");
  normalizeInPlace(q);

  const k = Number(m.character?.settings?.k ?? 5);
  const hits = store.search(q, k);
  if (!hits.length) throw new Error("No retrieval hits.");
  logger.info("Top retrieval hits:");
  hits.forEach((h: any, i: number) => console.log(`  (${i + 1}) score=${h.score.toFixed(3)} :: ${h.text}`));

  const ctx = "### Retrieved Knowledge\n" + hits.map((h: any, i: number) => `(${i + 1}) [${h.score.toFixed(3)}] ${h.text}`).join("\n");
  const prompt = [
    manifest.character?.name ? `You are ${manifest.character.name}.` : "",
    manifest.character?.system ? `System: ${manifest.character.system}` : "",
    ctx,
    "### Task",
    query,
    "",
    "Answer concisely and cite snippets (1..k) inline like [1],[2]."
  ].filter(Boolean).join("\n");

  console.log("\n--- Prompt Preview ---\n" + prompt + "\n");

  // 4) 0G call (strict: no env fallback)
  const evmRpc = requireNonEmpty("EVM_RPC", EVM_RPC);
  const privKey = requireNonEmpty("PRIVATE_KEY", PRIVATE_KEY);

  const wallet = new Wallet(privKey, new JsonRpcProvider(evmRpc));
  const og = await OgBrokerService.start(runtime);
  await og.initialize(wallet);

  const providerId = m.model?.providerId;
  if (!providerId) throw new Error("model.providerId missing.");
  const res = await og.infer({ prompt, providerAddress: providerId });

  console.log("\n=== 0G Model Reply ===\n" + res.text + "\n");
  console.log("--- meta ---");
  console.log(`provider: ${res.provider}`);
  console.log(`model:    ${res.model}`);
  if ((res as any).chatID) console.log(`chatID:   ${(res as any).chatID}`);

  logger.info("Done.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
