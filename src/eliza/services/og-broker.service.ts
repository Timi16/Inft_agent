import { Service, IAgentRuntime, logger, ServiceType } from "@elizaos/core";
import { Wallet, JsonRpcProvider, formatEther } from "ethers";
import { createRequire } from "node:module";
import { webcrypto as _crypto } from "node:crypto";
if (!(globalThis as any).crypto) (globalThis as any).crypto = _crypto;
import fetch from "node-fetch"; // Node 18+ has global fetch; but not in all runtimes
type CreateBrokerFn = (signer: any) => Promise<any>;

/** Prefer package main via require() (CJS) to avoid the ESM error-formatter path; fallback to ESM. */

async function loadCreateBrokerPreferCjs(): Promise<CreateBrokerFn> {
  try {
    const require = createRequire(import.meta.url);
    const cjs = require("@0glabs/0g-serving-broker");
    const fn =
      cjs.createZGComputeNetworkBroker;
    if (typeof fn === "function") return fn as CreateBrokerFn;
  } catch {}
  const esm = await import("@0glabs/0g-serving-broker");
  const fn =
    (esm).createZGComputeNetworkBroker;
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

/**
 * Fund only what the wallet can afford after reserving gas.
 * Numbers only (A0GI). Creates ledger if missing; otherwise deposits delta.
 * IMPORTANT: If ledger exists but wallet has no "available" above reserve, we **skip** top-up (no throw).
 */
async function ensureFundedLedger(
  broker: any,
  provider: JsonRpcProvider,
  walletAddr: string,
  min = 0.01,          // desired target on-ledger balance
  gasReserve = 0.005  // keep this much in wallet for gas fees
) {
  const target = Number(min);
  const reserve = Math.max(0, Number(gasReserve));
  if (!Number.isFinite(target) || target <= 0) {
    throw new Error(`ensureFundedLedger: invalid target amount: ${min}`);
  }

  // 1) Wallet native balance → available for funding after gas reserve
  const balWei = await provider.getBalance(walletAddr);
  const walletBal = Number(formatEther(balWei));
  const available = Math.max(0, +(walletBal - reserve).toFixed(6));

  // 2) Try to read existing ledger balance (SDK variants differ)
  let current: number | null = null;
  const getLedger = broker?.ledger?.getLedger?.bind(broker.ledger);
  if (typeof getLedger === "function") {
    const info = await getLedger().catch(() => null);
    const raw = info?.balance ?? info?.data?.balance;
    if (raw !== undefined) {
      const n = Number(raw);
      if (Number.isFinite(n)) current = n;
    }
  }

  // Helper: deposit a clamped Number amount
  const deposit = async (amount: number) => {
    const amt = Math.max(0, +amount.toFixed(6));
    if (amt <= 0) return;
    await broker.ledger.depositFund(amt);
  };

  // 3) If we can see a balance, top up only the needed delta (but no more than available).
  //    If available is 0 or negative, just proceed (no throw); the ledger already exists.
  if (current !== null) {
    if (current >= target) return; // already funded enough
    const need = Math.max(0, +(target - current).toFixed(6));
    if (available <= 0) {
      logger.info(
        `[0G] Ledger exists (balance=${current.toFixed(6)}), ` +
        `but wallet has no available above reserve (reserve=${reserve}). Skipping top-up.`
      );
      return;
    }
    const fund = Math.min(need, available);
    await deposit(fund);
    return;
  }

  // 4) No readable balance → try to create with an affordable initial amount
  if (available <= 0) {
    throw new Error(
      `Insufficient wallet balance to create ledger: have ${walletBal.toFixed(6)} A0GI, ` +
      `reserve ${reserve}, available ${available}. Top up or lower OG_MIN_LEDGER.`
    );
  }

  const init = Math.min(target, available);
  try {
    await broker.ledger.addLedger(+init.toFixed(6));
    return;
  } catch (e: any) {
    // If creation failed (already exists or masked by formatter), try a deposit fallback.
    try {
      await deposit(init);
      return;
    } catch {
      throw e;
    }
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
    // === Single source of truth: EVM_RPC & PRIVATE_KEY ===
    this.rpcUrl = this.requireEnv("EVM_RPC");
    this.privateKey = this.requireEnv("PRIVATE_KEY"); // must be 0x-prefixed

    logger.info(`[0G] RPC: ${this.rpcUrl}`);
    this.provider = new JsonRpcProvider(this.rpcUrl);
    this.wallet = new Wallet(this.privateKey, this.provider);

    // Preflight: network & wallet balance
    const [net, bal] = await Promise.all([
      this.provider.getNetwork(),
      this.provider.getBalance(this.wallet.address),
    ]);
    logger.info(`[0G] chainId=${net.chainId} balance=${formatEther(bal)} A0GI`);
    if (bal === 0n) {
      throw new Error(
        `0G wallet has 0 A0GI on chainId=${net.chainId}. Fund ${this.wallet.address} on 0G testnet, then rerun.`
      );
    }

    // Create broker
    const createBroker = await loadCreateBrokerPreferCjs();
    this.broker = await createBroker(this.wallet);

    // Ensure ledger exists and is funded (affordable funding)
    const gasReserve = Number(process.env.OG_GAS_RESERVE || "0.0015");
    await ensureFundedLedger(
      this.broker,
      this.provider,
      this.wallet.address,
      Number( "0.01"),
      gasReserve
    );

    // List services (supports old top-level and new .inference)
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

    // Get service metadata (endpoint, model id)
    const getMeta =
      this.broker?.inference?.getServiceMetadata?.bind(this.broker.inference) ??
      this.broker?.getServiceMetadata?.bind(this.broker);
    if (!getMeta) throw new Error("Broker is missing getServiceMetadata()");
    const { endpoint, model } = await getMeta(svc.provider);

    // Required once per signer
    await this.broker.inference.acknowledgeProviderSigner(svc.provider);
    
    const contentForBilling = params.prompt;
    const headers = await this.broker.inference.getRequestHeaders(svc.provider, contentForBilling);

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
