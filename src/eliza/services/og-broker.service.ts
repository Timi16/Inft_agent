import { Service, IAgentRuntime, logger, ServiceType } from "@elizaos/core";
import { Wallet, JsonRpcProvider, formatEther } from "ethers";
import { createRequire } from "node:module";

type CreateBrokerFn = (signer: any) => Promise<any>;

async function loadCreateBrokerPreferCjs(): Promise<CreateBrokerFn> {
  // Prefer package main via require() (CJS) to avoid the ESM error-formatter path.
  try {
    const require = createRequire(import.meta.url);
    const cjs = require("@0glabs/0g-serving-broker");
    const fn = cjs?.createZGComputeNetworkBroker ?? cjs?.default?.createZGComputeNetworkBroker;
    if (typeof fn === "function") return fn as CreateBrokerFn;
  } catch {}
  // Fallback to ESM
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

// ---------- Ledger helpers ----------

/** Get broker ledger balance (A0GI) and whether it exists. */
async function getLedgerBalance(broker: any): Promise<{ exists: boolean; balance: number }> {
  try {
    const info = await broker.ledger.getLedger();
    const raw = (info && (info.balance ?? info?.data?.balance)) ?? 0;
    return { exists: true, balance: Number(raw) };
  } catch {
    return { exists: false, balance: 0 };
  }
}

/** Ensure the broker ledger exists & is funded to at least `minimum` A0GI. */
async function ensureFundedLedger(broker: any, minimum = 0.1) {
  const target = Number(minimum);
  if (!Number.isFinite(target) || target <= 0) {
    throw new Error(`Invalid minimum fund amount: ${minimum}`);
  }

  let { exists, balance } = await getLedgerBalance(broker);

  if (!exists) {
    try {
      await broker.ledger.addLedger(target); // number in A0GI
      exists = true;
      balance = target;
    } catch (err: any) {
      const msg = String(err?.message ?? err);
      if (!/exist|already/i.test(msg)) throw err;
      // ledger already exists
      exists = true;
      ({ balance } = await getLedgerBalance(broker));
    }
  }

  if (balance < target) {
    const diff = target - balance;
    await broker.ledger.depositFund(diff); // number in A0GI
    ({ balance } = await getLedgerBalance(broker));
  }

  return { exists, balance };
}

// ---------- Service ----------

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
  private minFund: number = Number(process.env.OG_MIN_FUND ?? 0.1);

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

  /** Public: read the broker ledger balance (A0GI). */
  async ledgerInfo(): Promise<{ exists: boolean; balance: number }> {
    return getLedgerBalance(this.broker);
  }

  /** Public: read wallet address + native balance (formatted as ETH-style 18-decimals). */
  async walletInfo(): Promise<{ address: string; nativeBalance: string }> {
    const address = await this.wallet.getAddress();
    const wei = await this.provider.getBalance(address);
    return { address, nativeBalance: formatEther(wei) };
  }

  /** Public: list discovered services. */
  async listServices(): Promise<OgService[]> {
    return this.services.slice();
  }

  private async initialize() {
    this.rpcUrl = this.requireEnv("EVM_RPC");
    this.privateKey = this.requireEnv("PRIVATE_KEY"); // 0x-prefixed
    logger.info(`[0G] RPC: ${this.rpcUrl}`);

    this.provider = new JsonRpcProvider(this.rpcUrl);
    this.wallet = new Wallet(this.privateKey, this.provider);

    const createBroker = await loadCreateBrokerPreferCjs();
    this.broker = await createBroker(this.wallet);

    // Fund ledger before billable actions (prevents formatter crash path)
    const pre = await ensureFundedLedger(this.broker, this.minFund);
    logger.info(`[0G] Ledger balance (pre-list): ${pre.balance.toFixed(4)} A0GI (exists=${pre.exists})`);

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

    // Fetch meta
    const getMeta =
      this.broker?.inference?.getServiceMetadata?.bind(this.broker.inference) ??
      this.broker?.getServiceMetadata?.bind(this.broker);
    if (!getMeta) throw new Error("Broker is missing getServiceMetadata()");
    const { endpoint, model } = await getMeta(svc.provider);

    // Ack once per signer/provider
    await this.broker.inference.acknowledgeProviderSigner(svc.provider);

    // Generate single-use billing headers
    const contentForBilling = params.prompt;
    let headers: Record<string, string>;
    try {
      // Top-up if needed just before header gen
      await ensureFundedLedger(this.broker, this.minFund);
      headers = await this.broker.inference.getRequestHeaders(svc.provider, contentForBilling);
    } catch (e: any) {
      const msg = String(e?.message ?? e);
      const esmFormatterCrashed = msg.includes("keccak256") || msg.includes("error-handler.ts");
      if (!esmFormatterCrashed) throw e;

      // Recreate via CJS + refund, then retry once
      const createBroker = await loadCreateBrokerPreferCjs();
      this.broker = await createBroker(this.wallet);
      await ensureFundedLedger(this.broker, this.minFund);
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
