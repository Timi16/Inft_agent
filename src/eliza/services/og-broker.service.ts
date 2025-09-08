// src/eliza-og-plugin/services/og-broker.service.ts
import { Service, IAgentRuntime, logger, ServiceType } from "@elizaos/core";
import { Wallet, JsonRpcProvider } from "ethers";
import { createRequire } from "node:module";

type CreateBrokerFn = (signer: any) => Promise<any>;

// Prefer package main via require() (CJS) to avoid ESM error-formatter path.
// Fall back to ESM only if CJS isn’t available.
async function loadCreateBrokerPreferCjs(): Promise<CreateBrokerFn> {
  try {
    const require = createRequire(import.meta.url);
    const cjs = require("@0glabs/0g-serving-broker");
    const fn = cjs?.createZGComputeNetworkBroker ?? cjs?.default?.createZGComputeNetworkBroker;
    if (typeof fn === "function") return fn as CreateBrokerFn;
  } catch {}
  const esm = await import("@0glabs/0g-serving-broker");
  const fn =
    (esm as any).createZGComputeNetworkBroker ??
    (esm as any).default?.createZGComputeNetworkBroker;
  if (typeof fn === "function") return fn as CreateBrokerFn;
  throw new Error("createZGComputeNetworkBroker not found in @0glabs/0g-serving-broker");
}

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

// Ensure a ledger exists & has funds in A0GI; idempotent-ish across SDK versions
async function ensureFundedLedger(broker: any, min = 0.01) {
  const getLedger = broker?.ledger?.getLedger?.bind(broker.ledger);
  let balance = 0;
  if (getLedger) {
    const info = await getLedger().catch(() => null);
    balance = Number((info && (info.balance ?? info?.data?.balance)) ?? 0);
  }
  if (!balance || balance < min) {
    await broker.ledger?.addLedger?.(String(min)).catch(() => {});
    if (!broker?.ledger?.depositFund) {
      throw new Error("Broker ledger API missing depositFund(amount). Update @0glabs/0g-serving-broker.");
    }
    await broker.ledger.depositFund(String(min));
  }
}

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

  private requireEnv(name: string): string {
    const v = process.env[name] || this.runtime.getSetting?.(name);
    if (!v) throw new Error(`Missing required env: ${name}`);
    return v;
  }

  private async initialize() {
    // ⬇⬇⬇ SINGLE SOURCE OF TRUTH: EVM_RPC & PRIVATE_KEY
    this.rpcUrl = this.requireEnv("EVM_RPC");
    this.privateKey = this.requireEnv("PRIVATE_KEY"); // must be 0x-prefixed

    logger.info(`[0G] RPC: ${this.rpcUrl}`);
    this.provider = new JsonRpcProvider(this.rpcUrl);
    this.wallet = new Wallet(this.privateKey, this.provider);

    const createBroker = await loadCreateBrokerPreferCjs();
    this.broker = await createBroker(this.wallet);

    // Avoid business-error path that triggers the ESM formatter crash
    await ensureFundedLedger(this.broker, 0.01);

    // list services (support both old top-level and new .inference)
    const listSvc =
      this.broker?.inference?.listService?.bind(this.broker.inference) ??
      this.broker?.listService?.bind(this.broker);
    if (!listSvc) throw new Error("Broker is missing listService()");
    this.services = await listSvc();
    logger.info(`OgBrokerService: loaded ${this.services.length} services`);
  }

  private selectProvider(opts: { providerAddress?: string; modelHint?: string }): OgService {
    if (!this.services.length) throw new Error("No 0G services available");
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
    return this.services[0];
  }

  async infer(params: OgInferParams): Promise<{ text: string; chatID?: string; provider: string; model: string }> {
    const svc = this.selectProvider({
      providerAddress: params.providerAddress,
      modelHint: params.modelHint ?? process.env.OG_MODEL_HINT,
    });

    // Get service meta
    const getMeta =
      this.broker?.inference?.getServiceMetadata?.bind(this.broker.inference) ??
      this.broker?.getServiceMetadata?.bind(this.broker);
    if (!getMeta) throw new Error("Broker is missing getServiceMetadata()");
    const { endpoint, model } = await getMeta(svc.provider);

    // Required once per signer
    await this.broker.inference.acknowledgeProviderSigner(svc.provider);

    // Generate single-use billing headers (catch the ESM keccak formatter case; recreate via CJS and retry)
    const contentForBilling = params.prompt;
    let headers: Record<string, string>;
    try {
      headers = await this.broker.inference.getRequestHeaders(svc.provider, contentForBilling);
    } catch (e: any) {
      const msg = String(e?.message ?? e);
      const esmFormatterCrashed = msg.includes("keccak256") || msg.includes("error-handler.ts");
      if (!esmFormatterCrashed) throw e;

      const createBroker = await loadCreateBrokerPreferCjs();
      this.broker = await createBroker(this.wallet);
      await ensureFundedLedger(this.broker, 0.01);
      await this.broker.inference.acknowledgeProviderSigner(svc.provider);
      headers = await this.broker.inference.getRequestHeaders(svc.provider, contentForBilling);
    }

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

    return { text, chatID, provider: svc.provider, model };
  }
}
