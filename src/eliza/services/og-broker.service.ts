// 0G Serving Broker integration as an Eliza Service.
// Flow per 0G docs: initialize with ethers signer -> list services -> get metadata
// -> acknowledgeProviderSigner -> getRequestHeaders -> call endpoint (OpenAI-compatible).

import { Service, IAgentRuntime, logger, ServiceType } from "@elizaos/core";
import { Wallet, JsonRpcProvider } from "ethers";
import { createZGComputeNetworkBroker } from "@0glabs/0g-serving-broker";

// Minimal shape for a service entry (per 0G docs page).
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
  providerAddress?: string;        // optional; we'll select if omitted
  prompt: string;                  // system/user content to send
  messages?: Array<{ role: "system" | "user" | "assistant"; content: string }>;
  modelHint?: string;              // filter by service.model includes(...)
};

export class OgBrokerService extends Service {
  static serviceType = ServiceType.TEE; // closest bucket for "compute/inference"
  capabilityDescription = "0G Serving Broker client for inference (OpenAI-compatible endpoints).";

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

  async stop(): Promise<void> {
    // broker has no long-lived handles we must close; noop
  }

  private requireEnv(key: string): string {
    const v = process.env[key] || this.runtime.getSetting(key);
    if (!v) throw new Error(`Missing required setting: ${key}`);
    return v;
  }

  private async initialize() {
    // Required: RPC + PRIVATE_KEY (Node) — for browser, swap for wagmi signer.
    this.rpcUrl = this.requireEnv("OG_RPC_URL");      // e.g., Galileo testnet RPC (see 0G Testnet page)
    this.privateKey = this.requireEnv("OG_PRIVATE_KEY");

    this.provider = new JsonRpcProvider(this.rpcUrl);
    this.wallet = new Wallet(this.privateKey, this.provider);

    this.broker = await createZGComputeNetworkBroker(this.wallet);

    // Warm list of services from contract
    this.services = await this.broker.listService();

    logger.info(`OgBrokerService: loaded ${this.services.length} services`);
  }

  /** Choose a provider by address (preferred) or by model hint substring. */
  private selectProvider(opts: { providerAddress?: string; modelHint?: string }): OgService {
    if (opts.providerAddress) {
      const s = this.services.find((x) => x.provider.toLowerCase() === opts.providerAddress!.toLowerCase());
      if (!s) throw new Error(`0G provider not found: ${opts.providerAddress}`);
      return s;
    }
    if (opts.modelHint) {
      const s = this.services.find((x) => (x.model || "").toLowerCase().includes(opts.modelHint!.toLowerCase()));
      if (s) return s;
    }
    if (!this.services.length) throw new Error("No 0G services available");
    return this.services[0];
  }

  /** One-shot chat completion using 0G billing headers + OpenAI-compatible endpoint. */
  async infer(params: OgInferParams): Promise<{ text: string; chatID?: string; provider: string; model: string }> {
    const svc = this.selectProvider({ providerAddress: params.providerAddress, modelHint: params.modelHint });

    // Fetch endpoint + model metadata from chain
    const { endpoint, model } = await this.broker.getServiceMetadata(svc.provider);

    // One-time acknowledgement per provider
    await this.broker.inference.acknowledgeProviderSigner(svc.provider);

    // Generate single-use billing headers for this request
    const contentForBilling = params.prompt;
    const headers = await this.broker.inference.getRequestHeaders(svc.provider, contentForBilling);

    // Build OpenAI-compatible body
    const messages = params.messages ?? [{ role: "system", content: params.prompt }];
    const res = await fetch(`${endpoint}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify({ model, messages }),
    });

    if (!res.ok) {
      const errBody = await res.text();
      throw new Error(`0G inference failed: ${res.status} ${res.statusText} :: ${errBody}`);
    }

    const json = await res.json();
    // Try to extract text + chatID if present
    const choice = json?.choices?.[0];
    const text: string =
      choice?.message?.content ??
      json?.content ??
      json?.output ??
      JSON.stringify(json);
    const chatID: string | undefined = json?.id || json?.chat_id;

    // Optional: verify TEE signatures if verifiable
    // const valid = await this.broker.inference.processResponse(svc.provider, text, chatID);
    // logger.info(`0G response verifiable=${svc.verifiability}, valid=${valid}`);

    return { text, chatID, provider: svc.provider, model };
  }
}
