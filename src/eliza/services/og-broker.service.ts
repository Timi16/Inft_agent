// Normal imports, no dynamic require/import hacks.
import { Service, IAgentRuntime, logger, ServiceType } from "@elizaos/core";
import { Wallet, JsonRpcProvider } from "ethers";
import { createZGComputeNetworkBroker } from "@0glabs/0g-serving-broker";
import OpenAI from "openai";
import { webcrypto as _webcrypto } from "node:crypto";

// Ensure WebCrypto is available for the broker's crypto adapter
if (!(globalThis as any).crypto) (globalThis as any).crypto = _webcrypto;

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

  async stop(): Promise<void> {}

  private getSetting(name: string, fallback?: string): string {
    const v = process.env[name] || this.runtime.getSetting?.(name) || fallback;
    if (!v) throw new Error(`Missing required env: ${name}`);
    return v;
  }

  private async initialize(): Promise<void> {
    if (this.initialized) return;

    this.rpcUrl = this.getSetting("EVM_RPC", "https://evmrpc-testnet.0g.ai");
    this.privateKey = this.getSetting("PRIVATE_KEY");

    logger.info(`[0G] RPC: ${this.rpcUrl}`);
    this.provider = new JsonRpcProvider(this.rpcUrl);
    this.wallet = new Wallet(this.privateKey, this.provider);

    this.broker = await createZGComputeNetworkBroker(this.wallet);

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

    return this.services[0];
  }

  /**
   * Mirror Code A flow:
   * - acknowledge signer
   * - get service metadata
   * - sign headers with getRequestHeaders(provider, message)
   * - call provider via OpenAI client using defaultHeaders
   */
  async infer(
    params: OgInferParams
  ): Promise<{ text: string; chatID?: string; provider: string; model: string }> {
    if (!this.initialized) await this.initialize();

    const svc = this.selectProvider({
      providerAddress: params.providerAddress,
      modelHint: params.modelHint || process.env.OG_MODEL_HINT,
    });

    const { endpoint, model } = await this.broker.inference.getServiceMetadata(svc.provider);
    await this.broker.inference.acknowledgeProviderSigner(svc.provider);

    // Use the billed content (same as Code A)
    const userMessage = params.prompt;

    // This call was crashing for you due to missing WebCrypto. Fixed by polyfill above.
    const headers = await this.broker.inference.getRequestHeaders(svc.provider, userMessage);

    const messages =
      params.messages ??
      [
        // Keep it simple & Code-A-like: one user message
        { role: "user", content: userMessage },
      ];

    const openai = new OpenAI({
      baseURL: endpoint,
      apiKey: "", // not used by 0G
      defaultHeaders: { ...headers } as Record<string, string>,
    });

    const completion = await openai.chat.completions.create({
      model,
      messages,
    });

    const text =
      completion.choices?.[0]?.message?.content ??
      JSON.stringify(completion);

    return {
      text,
      chatID: (completion as any)?.id,
      provider: svc.provider,
      model,
    };
  }
}
