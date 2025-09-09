// Code B rewritten in Code A style (no global crypto shims, no icons)

import "dotenv/config";
import { Service, IAgentRuntime, logger, ServiceType } from "@elizaos/core";
import { Wallet, JsonRpcProvider } from "ethers";
import { createZGComputeNetworkBroker } from "@0glabs/0g-serving-broker";
import OpenAI from "openai";

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

  // Code A–style core fields
  private provider!: JsonRpcProvider;
  private wallet!: Wallet;
  private broker!: Awaited<ReturnType<typeof createZGComputeNetworkBroker>>;

  // Service state (Code A pattern)
  private isInitialized = false;
  private providerAddress: string | null = null;
  private endpoint: string | null = null;
  private model: string | null = null;

  // Retain Code B’s cached list (optional)
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

  private getSetting(name: string, fallback?: string): string {
    const v = process.env[name] || this.runtime.getSetting?.(name) || fallback;
    if (!v) throw new Error(`Missing required env: ${name}`);
    return v;
  }

  private async initialize(): Promise<void> {
    if (this.isInitialized) return;

    this.rpcUrl = this.getSetting("EVM_RPC");
    this.privateKey = this.getSetting("PRIVATE_KEY");

    this.provider = new JsonRpcProvider(this.rpcUrl);
    this.wallet = new Wallet(this.privateKey, this.provider);

    // Code A: create broker with wallet signer
    this.broker = await createZGComputeNetworkBroker(this.wallet);

    // Discover services
    this.services = await this.broker.inference.listService();
    if (!this.services.length) throw new Error("No 0G inference services available.");

    // Code A: pick the first service
    this.providerAddress = this.services[0].provider;

    // Code A: acknowledge signer for this provider
    await this.broker.inference.acknowledgeProviderSigner(this.providerAddress);

    // Code A: get endpoint + model from metadata
    const meta = await this.broker.inference.getServiceMetadata(this.providerAddress);
    this.endpoint = meta.endpoint;
    this.model = meta.model;

    this.isInitialized = true;
    logger.info(`OgBrokerService: initialized; provider=${this.providerAddress}, model=${this.model}`);
  }
  /**
   * Code A flow inside Code B’s `infer`:
   * - ensure initialized
   * - use class-level providerAddress/endpoint/model
   * - sign headers with getRequestHeaders(providerAddress, billedContent)
   * - call OpenAI with defaultHeaders
   */
  async infer(
    params: OgInferParams
  ): Promise<{ text: string; chatID?: string; provider: string; model: string }> {
    if (!this.isInitialized) await this.initialize();

    // Allow override of provider for this call, still keeping Code A style fields
    if (params.providerAddress && params.providerAddress !== this.providerAddress) {
      // Switch provider to requested one, but keep Code A pattern
      await this.broker.inference.acknowledgeProviderSigner(params.providerAddress);
      const meta = await this.broker.inference.getServiceMetadata(params.providerAddress);
      this.providerAddress = params.providerAddress;
      this.endpoint = meta.endpoint;
      this.model = meta.model;
    }

    if (!this.providerAddress || !this.endpoint || !this.model) {
      throw new Error("OgBrokerService is not properly initialized.");
    }

    const billedContent = params.prompt;

    // Code A: get per-request signed headers
    const headers = await this.broker.inference.getRequestHeaders(
      this.providerAddress,
      billedContent
    );

    const messages =
      params.messages ?? [{ role: "user", content: billedContent }];

    const openai = new OpenAI({
      baseURL: this.endpoint,
      apiKey: "", // not required by 0G
      defaultHeaders: headers as unknown as Record<string, string>,
    });
    
    const completion = await openai.chat.completions.create({
      model: this.model,
      messages,
    });

    const text =
      completion.choices?.[0]?.message?.content ??
      JSON.stringify(completion);

    return {
      text,
      chatID: (completion as any)?.id,
      provider: this.providerAddress,
      model: this.model,
    };
  }
}
