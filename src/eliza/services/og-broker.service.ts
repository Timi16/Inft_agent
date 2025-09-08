// src/eliza-og-plugin/services/og-broker.service.ts
import { Service, IAgentRuntime, logger, ServiceType } from "@elizaos/core";
import { Wallet, JsonRpcProvider } from "ethers";
import { createRequire } from "node:module";

// ---- Dynamic loader to avoid ESM/CJS export mismatch ----
type CreateBrokerFn = (signer: any) => Promise<any>;

async function loadCreateBroker(): Promise<CreateBrokerFn> {
  // 1) Try package ESM entry (named export)
  try {
    const esm = await import("@0glabs/0g-serving-broker");
    const fn =
      (esm as any).createZGComputeNetworkBroker ??
      (esm as any).default?.createZGComputeNetworkBroker;
    if (typeof fn === "function") return fn as CreateBrokerFn;
  } catch {
    // fall through to CJS
  }
  // 2) Try CommonJS build (in case the runtime resolves CJS)
  try {
    const require = createRequire(import.meta.url);
    const cjs = require("@0glabs/0g-serving-broker");
    const fn =
      cjs?.createZGComputeNetworkBroker ?? cjs?.default?.createZGComputeNetworkBroker;
    if (typeof fn === "function") return fn as CreateBrokerFn;
  } catch {
    // fall through
  }
  // 3) Hard fail with a helpful message
  throw new Error(
    "Failed to load @0glabs/0g-serving-broker. " +
      "It provides a named export `createZGComputeNetworkBroker` (no default). " +
      "Ensure the package is installed and your runtime supports ESM."
  );
}

// Minimal shape for a service entry (per 0G docs)
export type OgService = {
  provider: string;
  serviceType: string;
  url: string;
  inputPrice: bigint;
  outputPrice: bigint;
  updatedAt: bigint;
  model: string;
  verifiability: string;
  additionalInfo: string;
};

export type OgInferParams = {
  providerAddress?: string;
  prompt: string;
  messages?: Array<{ role: "system" | "user" | "assistant"; content: string }>;
  modelHint?: string;
};

export class OgBrokerService extends Service {
  static serviceType = ServiceType.TEE;
  capabilityDescription =
    "0G Serving Broker client for inference (OpenAI-compatible endpoints).";

  private rpcUrl!: string;
  private privateKey!: string;
  private provider!: JsonRpcProvider;
  private wallet!: Wallet;
  private broker!: any;
  private services: OgService[] = [];

  constructor(protected runtime: IAgentRuntime) {
    super(runtime);
  }

  static async start(runtime: IAgentRuntime): Promise<OgBrokerService> {
    const svc = new OgBrokerService(runtime);
    await svc.initialize();
    return svc;
  }

  async stop(): Promise<void> {}

  private requireEnv(key: string): string {
    const v = process.env[key] || this.runtime.getSetting?.(key);
    if (!v) throw new Error(`Missing required setting: ${key}`);
    return v;
  }

  private async initialize() {
    this.rpcUrl = this.requireEnv("OG_RPC_URL");
    this.privateKey = this.requireEnv("OG_PRIVATE_KEY"); // must be 0x-prefixed for ethers

    this.provider = new JsonRpcProvider(this.rpcUrl);
    this.wallet = new Wallet(this.privateKey, this.provider);

    const createZGComputeNetworkBroker = await loadCreateBroker();
    this.broker = await createZGComputeNetworkBroker(this.wallet);

    // list services (API moved under .inference in newer docs; support both)
    const listSvc =
      this.broker?.inference?.listService?.bind(this.broker.inference) ??
      this.broker?.listService?.bind(this.broker);
    if (!listSvc) throw new Error("Broker is missing listService()");
    this.services = await listSvc();
    logger.info(`OgBrokerService: loaded ${this.services.length} services`);
  }

  private selectProvider(opts: { providerAddress?: string; modelHint?: string }): OgService {
    if (opts.providerAddress) {
      const s = this.services.find(
        (x) => x.provider.toLowerCase() === opts.providerAddress!.toLowerCase()
      );
      if (!s) throw new Error(`0G provider not found: ${opts.providerAddress}`);
      return s;
    }
    if (opts.modelHint) {
      const s = this.services.find((x) =>
        (x.model || "").toLowerCase().includes(opts.modelHint!.toLowerCase())
      );
      if (s) return s;
    }
    if (!this.services.length) throw new Error("No 0G services available");
    return this.services[0];
  }

  async infer(params: OgInferParams): Promise<{ text: string; chatID?: string; provider: string; model: string }> {
    const svc = this.selectProvider({
      providerAddress: params.providerAddress,
      modelHint: params.modelHint,
    });

    // getServiceMetadata may be on broker.inference in newer docs; support both
    const getMeta =
      this.broker?.inference?.getServiceMetadata?.bind(this.broker.inference) ??
      this.broker?.getServiceMetadata?.bind(this.broker);
    if (!getMeta) throw new Error("Broker is missing getServiceMetadata()");
    const { endpoint, model } = await getMeta(svc.provider);

    // Acknowledge provider once before use
    await this.broker.inference.acknowledgeProviderSigner(svc.provider);

    // Billing headers are single-use; generate per request
    const contentForBilling = params.prompt;
    const headers = await this.broker.inference.getRequestHeaders(svc.provider, contentForBilling);

    // Build OpenAI-compatible body
    const messages =
      params.messages ?? [{ role: "system", content: params.prompt }];

    const res = await fetch(`${endpoint}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify({ model, messages }),
    });

    if (!res.ok) {
      const errBody = await res.text().catch(() => "");
      throw new Error(`0G inference failed: ${res.status} ${res.statusText} :: ${errBody}`);
    }

    const json = await res.json();
    const choice = json?.choices?.[0];
    const text: string =
      choice?.message?.content ??
      json?.content ??
      json?.output ??
      JSON.stringify(json);
    const chatID: string | undefined = json?.id || json?.chat_id;

    // Optional: verify proof for verifiable services
    // const valid = await this.broker.inference.processResponse(svc.provider, text, chatID);

    return { text, chatID, provider: svc.provider, model };
  }
}
