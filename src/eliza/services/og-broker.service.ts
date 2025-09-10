// Code B rewritten in Code A style (no global crypto shims, no icons)

import "dotenv/config";
import { Service, IAgentRuntime, logger, ServiceType } from "@elizaos/core";
import { Wallet, JsonRpcProvider } from "ethers";
import { createZGComputeNetworkBroker } from "@0glabs/0g-serving-broker";
import OpenAI from "openai";

// Make sure ethers is fully imported for crypto functions
import * as ethers from "ethers";

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

  // Retain Code B's cached list (optional)
  private services: OgService[] = [];

  constructor(protected runtime: IAgentRuntime) {
    super(runtime);
  }

  static async start(runtime: IAgentRuntime): Promise<OgBrokerService> {
    const svc = new OgBrokerService(runtime);
    return svc;
  }

  async stop(): Promise<void> { }


 async initialize(signer: Wallet | ethers.JsonRpcSigner): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Create broker - this might need the full ethers context
      this.broker = await createZGComputeNetworkBroker(signer);

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
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : String(error);
      logger.error(`Inference error: ${msg}`);
      throw error; // preserves original stack
    }

  }

  /**
   * Code A flow inside Code B's `infer`:
   * - ensure initialized
   * - use class-level providerAddress/endpoint/model
   * - sign headers with getRequestHeaders(providerAddress, billedContent)
   * - call OpenAI with defaultHeaders
   */
  async infer(
    params: OgInferParams,
    signer?: Wallet | ethers.JsonRpcSigner
  ): Promise<{ text: string; chatID?: string; provider: string; model: string }> {
    if (!this.isInitialized) { if (!signer) { throw new Error("please pass in a signer, or call .initialize()") } await this.initialize(signer); }

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

    const content = params.prompt;

    try {
      // Code A: get per-request signed headers
      const headers = await this.broker.inference.getRequestHeaders(
        this.providerAddress,
        content
      );

      const messages =
        params.messages ?? [{ role: "user", content: content }];

      const openai = new OpenAI({
        baseURL: this.endpoint,
        apiKey: "dummy-key", // Use a dummy key instead of empty string
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
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : String(error);
      logger.error(`Inference error: ${msg}`);
      throw error; // preserves original stack
    }
  }
}