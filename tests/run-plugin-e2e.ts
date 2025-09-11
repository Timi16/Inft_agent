#!/usr/bin/env tsx
import { logger } from "@elizaos/core";
import { loadManifest } from "../src/services/inft-loader-service";
import { ExternalVectorStore } from "../src/services/external-store";
import { normalizeInPlace } from "../src/utils/cosine";
import { RemoteEmbedder } from "../src/services/embedding.service";
import { fetchBytesSmart, isIpfsLike, ipfsToHttp } from "../src/utils/uri";
import { createHash } from "node:crypto";

// =====================
// REQUIRED CONSTANTS
// =====================
// Pinata subdomain gateway (NO fallback). Edit to your exact subdomain.
const GATEWAY = "https://violet-deliberate-fly-257.mypinata.cloud";

// Manifest CID (NO ipfs:// prefix, NO args, NO env). Edit to your manifest CID.
const MANIFEST_CID = "QmY7wfF53UzHQjkLRQF6fi4s4afqMrb33wayeeySc4ZY4P";

// Remote embedder base URL (NO fallback).
const EMBEDDER_BASE_URL = "http://170.75.163.164:4916";

// =====================
// UTILS
// =====================
const sha256Hex = (u8: Uint8Array) => {
  const h = createHash("sha256");
  h.update(u8);
  return "sha256:" + h.digest("hex");
};
function requireNonEmpty(name: string, v: string) {
  if (!v || !v.trim()) throw new Error(`Missing required ${name}`);
  return v.trim();
}
function gatewayJoin(base: string, cid: string): string {
  const b = base.replace(/\/+$/, "");
  const c = cid.replace(/^\/+/, "");
  return `${b}/ipfs/${c}`;
}

// =====================
// MAIN
// =====================
async function main() {
  logger.info("=== iNFT + Remote Embedder (IPFS-only) E2E ===");

  const gateway = requireNonEmpty("GATEWAY", GATEWAY);
  const manifestCid = requireNonEmpty("MANIFEST_CID", MANIFEST_CID);
  const embedderBase = requireNonEmpty("EMBEDDER_BASE_URL", EMBEDDER_BASE_URL);

  // 1) Load manifest from Pinata subdomain gateway (no args, no ipfs://)
  const manifestUrl = gatewayJoin(gateway, manifestCid);
  const manifest = await loadManifest(manifestUrl, { gateway });
  logger.info(`Manifest loaded. model=${manifest.e_model.id} dim=${manifest.e_model.dim}`);

  // 2) Verify vectors checksum strictly via IPFS/gateway
  const mAny = manifest as any;
  if (!mAny.vectors_uri) throw new Error("Manifest missing vectors_uri (expected IPFS CID/URI).");
  if (!mAny.vectors_checksum) throw new Error("Manifest missing vectors_checksum.");

  const vectorsUrl = isIpfsLike(mAny.vectors_uri)
    ? ipfsToHttp(mAny.vectors_uri, gateway) // requires gateway; no fallback
    : mAny.vectors_uri;

  const vecBytes = await fetchBytesSmart(vectorsUrl, gateway);
  const got = sha256Hex(vecBytes);
  if (got !== mAny.vectors_checksum) {
    throw new Error(`vectors_checksum mismatch:\n  manifest: ${mAny.vectors_checksum}\n  computed: ${got}`);
  }

  // 3) Build store (downloads vectors.bin from the same gateway)
  const store = await ExternalVectorStore.fromManifestWithExternalVectors(manifest, gateway);
  logger.info(`Vectors loaded: ${store.size()} entries.`);

  // 4) Remote embedder ONLY (no LocalEmbedder, no fallback)
  const remote = new RemoteEmbedder(embedderBase);

  // 5) Query -> embed -> search
  const k: number = Number((mAny.character?.settings?.k ?? 5) as number);
  const query = "Give me a concise explanation of the feedback and incentive policies with key points.";

  const [qVec] = await remote.embed([query], { mode: "query", instruction: "e5", normalize: true });
  if (!qVec || qVec.length !== manifest.e_model.dim) {
    throw new Error(`Remote dim ${qVec?.length ?? 0} != manifest dim ${manifest.e_model.dim}`);
  }
  normalizeInPlace(qVec);

  const hits = store.search(qVec, k);
  if (!hits.length) throw new Error("No retrieval hits.");

  logger.info("Top retrieval hits:");
  hits.forEach((h, i) => console.log(`  (${i + 1}) score=${h.score.toFixed(3)} :: ${h.text}`));

  // Prompt preview (in case you later send to an LLM)
  const ctx =
    "### Retrieved Knowledge\n" +
    hits.map((h, i) => `(${i + 1}) [${h.score.toFixed(3)}] ${h.text}`).join("\n");

  const prompt = [
    manifest.character?.name ? `You are ${manifest.character.name}.` : "",
    manifest.character?.system ? `System: ${manifest.character.system}` : "",
    ctx,
    "### Task",
    query,
    "",
    "Answer concisely and cite snippets (1..k) inline like [1],[2] where relevant."
  ].filter(Boolean).join("\n");

  console.log("\n--- Prompt Preview ---\n");
  console.log(prompt);

  logger.info("Done (remote-only, IPFS-only, no args, no fallbacks).");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
