// 0G Serving Broker integration as an Eliza Service.
// Works on Node (Windows/Linux/Mac) by dynamically loading the ESM entry
// and falling back to the CommonJS bundle if Node's ESM re-export trips up.

import { Service, IAgentRuntime, logger, ServiceType } from "@elizaos/core";
import { Wallet, JsonRpcProvider } from "ethers";

// ---- Dynamic loader to avoid ESM/CJS export mismatch ----
type CreateBrokerFn = (signer: any) => Promise<any>;

async function loadCreateBroker(): Promise<CreateBrokerFn> {
  // 1) Try package ESM entry
  try {
    const esm = await import("@0glabs/0g-serving-broker");
    if ("createZGComputeNetworkBroker" in esm) {
      return (esm as any).createZGComputeNetworkBroker as CreateBrokerFn;
    }
  } catch (e) {
    // swallow; we'll try CJS next
  }
  // 3) Hard fail with a helpful message
  throw new Error(
    "Failed to load @0glabs/0g-serving-broker. " +
      "Tried ESM and CJS bundles. Ensure the package is installed and not corrupted."
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
  providerAddress?: string; // optional; we'll pick if omitted
  prompt: string;           // system/user content
  messages?: Array<{ role: "system" | "user" | "assistant"; content: string }>;
  modelHint?: string;       // substring filter against service.model
};

export class OgBrokerService extends Service {
  static serviceType = ServiceType.TEE; // closest category for compute/inference
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
    // No persistent handles to close
  }

  private requireEnv(key: string): string {
    const v = process.env[key] || this.runtime.getSetting?.(key);
    if (!v) throw new Error(`Missing required setting: ${key}`);
    return v;
  }

  private async initialize() {
    // Required: RPC + PRIVATE_KEY (use a funded key on 0G)
    this.rpcUrl = this.requireEnv("OG_RPC_URL");
    this.privateKey = this.requireEnv("OG_PRIVATE_KEY");

    this.provider = new JsonRpcProvider(this.rpcUrl);
    this.wallet = new Wallet(this.privateKey, this.provider);

    // Load broker factory (ESM -> CJS fallback)
    const createZGComputeNetworkBroker = await loadCreateBroker();

    // Create broker with signer
    this.broker = await createZGComputeNetworkBroker(this.wallet);

    // Warm cache of available services
    this.services = await this.broker.listService();
    logger.info(`OgBrokerService: loaded ${this.services.length} services`);
  }

  /** Choose a provider by address (preferred) or by model hint substring. */
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

  /** One-shot chat completion using 0G billing headers + OpenAI-compatible endpoint. */
  async infer(params: OgInferParams): Promise<{ text: string; chatID?: string; provider: string; model: string }> {
    const svc = this.selectProvider({
      providerAddress: params.providerAddress,
      modelHint: params.modelHint,
    });

    // Fetch endpoint + model metadata from chain
    const { endpoint, model } = await this.broker.getServiceMetadata(svc.provider);

    // One-time acknowledgement per provider (required)
    await this.broker.inference.acknowledgeProviderSigner(svc.provider);

    // Generate single-use billing headers for this request
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

    // Optional proof verification (if service is verifiable)
    // const valid = await this.broker.inference.processResponse(svc.provider, text, chatID);
    // logger.info(`0G response verifiable=${svc.verifiability}, valid=${valid}`);

    return { text, chatID, provider: svc.provider, model };
  }
}
