// tests/run-plugin-e2e.ts
// Quick smoke test for 0G broker only (no manifest, vectors, or embedder).
// Run with:  npx tsx tests/run-plugin-e2e.ts
// Requires:  EVM_RPC, PRIVATE_KEY  (and optional OG_MODEL_HINT, QUERY)

import "dotenv/config";
import { logger } from "@elizaos/core";
import { OgBrokerService } from "../src/eliza/services/og-broker.service";

// Minimal runtime shim for OgBrokerService
type RuntimeLike = {
  getSetting: (k: string) => string | undefined;
  getService: <T = unknown>(_name: any) => T | null;
};

const runtime: RuntimeLike = {
  getSetting: (k) => process.env[k],
  getService: () => null,
};

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`Missing required env: ${name}`);
  return v;
}

async function main() {
  logger.info("=== 0G Broker Quick Test ===");

  // Hard-require the essentials for broker auth
  requireEnv("EVM_RPC");
  requireEnv("PRIVATE_KEY");

  // Use QUERY if provided, else a tiny default
  const prompt =
    process.env.QUERY ||
    "Give me a one-sentence greeting that mentions 0G.";

  // Optional: help pick a provider by model substring
  const modelHint = process.env.OG_MODEL_HINT;

  // Start service (normal imports, no dynamic require/import hacks)
  const og = await OgBrokerService.start(runtime as any);

  // One-shot chat completion
  const res = await og.infer({ prompt, modelHint });

  console.log("\n=== 0G Model Reply ===\n");
  console.log(res.text);

  console.log("\n--- meta ---");
  console.log(`provider: ${res.provider}`);
  console.log(`model:    ${res.model}`);
  if (res.chatID) console.log(`chatID:   ${res.chatID}`);

  logger.info("Quick test completed.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
