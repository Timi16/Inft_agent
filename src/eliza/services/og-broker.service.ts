// src/eliza/services/og-broker.service.ts
// Clean, “normal” imports (no ESM/CJS shenanigans).
import { Service, IAgentRuntime, logger, ServiceType } from "@elizaos/core";
import { Wallet, JsonRpcProvider } from "ethers";
import { createZGComputeNetworkBroker } from "@0glabs/0g-serving-broker";

// Types that mirror broker service listings
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
  providerAddress?: string; // optional: pick a specific provider
  prompt: string;           // main user prompt (used for header signing)
  // Optional chat history; if not provided we'll use { role: "user", content: prompt }
  messages?: Array<{ role: "system" | "user" | "assistant"; content: string }>;
  modelHint?: string;       // optional: pick a provider by model substring
};

export class OgBrokerService extends Service {
  static serviceType = ServiceType.TEE;
  capabilityDescription =
    "0G Serving Broker client for inference (OpenAI-compatible endpoints).";

  private rpcUrl!: string;
  private privateKey!: string;
  private provider!: JsonRpcProvider;
  private wallet!: Wallet;
  // Type-safe without importing broker types directly:
  private broker!: Awaited<ReturnType<typeof createZGComputeNetworkBroker>>;
  private services: OgService[] = [];
  private initialized = false;

  constructor(protected runtime: IAgentRuntime) {
    super(runtime);
  }

  static async start(runtime: IAgentRuntime): Promise<OgBrokerService> {
    const svc = new OgBrokerService(runtime);
    await svc.initialize();
    return svc;
  }

  async stop(): Promise<void> {
    // nothing to tear down
  }

  private getSetting(name: string, fallback?: string): string {
    // .env first, then runtime settings, then fallback
    const v = process.env[name] || this.runtime.getSetting?.(name) || fallback;
    if (!v) throw new Error(`Missing required env: ${name}`);
    return v;
  }

  private async initialize(): Promise<void> {
    if (this.initialized) return;

    // 1) RPC + signer
    this.rpcUrl = this.getSetting("EVM_RPC", "https://evmrpc-testnet.0g.ai");
    this.privateKey = this.getSetting("PRIVATE_KEY"); // must be 0x-prefixed hex

    logger.info(`[0G] RPC: ${this.rpcUrl}`);
    this.provider = new JsonRpcProvider(this.rpcUrl);
    this.wallet = new Wallet(this.privateKey, this.provider);

    // 2) Broker
    this.broker = await createZGComputeNetworkBroker(this.wallet);

    // 3) Discover services
    this.services = await this.broker.inference.listService();
    if (!this.services.length) {
      throw new Error("No 0G inference services available.");
    }
    logger.info(`OgBrokerService: loaded ${this.services.length} services`);
    this.initialized = true;
  }

  private selectProvider(opts?: { providerAddress?: string; modelHint?: string }): OgService {
    const { providerAddress, modelHint } = opts ?? {};
    if (!this.services.length) throw new Error("No 0G services available");

    if (providerAddress) {
      const match = this.services.find(
        (s) => s.provider.toLowerCase() === providerAddress.toLowerCase()
      );
      if (match) return match;
      throw new Error(`0G provider not found: ${providerAddress}`);
    }

    if (modelHint) {
      const byModel = this.services.find((s) =>
        (s.model ?? "").toLowerCase().includes(modelHint.toLowerCase())
      );
      if (byModel) return byModel;
    }

    // Default: first service
    return this.services[0];
  }

  /**
   * Perform a single chat completion request via 0G broker.
   * - Signs request headers with your wallet (billing/auth)
   * - Calls provider's OpenAI-compatible /chat/completions
   */
  async infer(
    params: OgInferParams
  ): Promise<{ text: string; chatID?: string; provider: string; model: string }> {
    if (!this.initialized) await this.initialize();

    const svc = this.selectProvider({
      providerAddress: params.providerAddress,
      modelHint: params.modelHint,
    });

    // Service endpoint + model metadata
    const { endpoint, model } = await this.broker.inference.getServiceMetadata(svc.provider);

    // Required once per signer per provider (safe to call repeatedly)
    await this.broker.inference.acknowledgeProviderSigner(svc.provider);

    // Headers must be derived from the content being billed
    const headers = await this.broker.inference.getRequestHeaders(
      svc.provider,
      params.prompt
    );

    // Messages payload (use prompt as user message if none provided)
    const messages =
      params.messages ?? [{ role: "user", content: params.prompt }];

    // Node 18+ has global fetch; no polyfills needed.
    const res = await fetch(`${endpoint}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify({ model, messages }),
    });

    if (!res.ok) {
      const errBody = await res.text().catch(() => "");
      throw new Error(
        `0G inference failed: ${res.status} ${res.statusText} :: ${errBody}`
      );
    }

    const json = await res.json();
    const text: string =
      json?.choices?.[0]?.message?.content ??
      json?.content ??
      json?.output ??
      JSON.stringify(json);
    const chatID: string | undefined = json?.id ?? json?.chat_id;

    return { text, chatID, provider: svc.provider, model };
  }
}
