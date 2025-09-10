// Wrap your iNFT loader + vector search as an Eliza Service.
// Provides: loadManifest(uri), search(queryText, k).

import { Service, IAgentRuntime, logger, ServiceType } from "@elizaos/core";
import { loadManifest } from "../../services/inft-loader-service";   // you already have this
import { ExternalVectorStore } from "../../services/external-store"; // you already have this
import { LocalEmbedder } from "../../services/local-embedder";       // offline embedder we added
import { ensureModelContract } from "../../services/contracts";       // model contract checks
import { normalizeInPlace } from "../../utils/cosine";
import type { InftManifest } from "../../types";

export class InftKnowledgeService extends Service {
  static serviceType = ServiceType.WEB_SEARCH; // closest built-in category for "KB retrieval"
  capabilityDescription = "Loads iNFT manifests + vectors and performs local similarity search.";

  private manifest: InftManifest | null = null;
  private store: ExternalVectorStore | null = null;
  private embedder: LocalEmbedder | null = null;

  constructor(protected runtime: IAgentRuntime) {
    super(runtime);
  }

  static async start(runtime: IAgentRuntime): Promise<InftKnowledgeService> {
    const svc = new InftKnowledgeService(runtime);
    // Lazy init; nothing to do here.
    return svc;
  }

  async stop(): Promise<void> {
    // no long-lived handles to close
  }

  async loadManifest(uri: string) {
    this.manifest = await loadManifest(uri);
    this.store = await ExternalVectorStore.fromManifestWithExternalVectors(this.manifest);
    this.embedder = new LocalEmbedder(process.env.MODEL_ID || this.manifest.e_model.id);

    // Check model contract (id/dim)
    const info = await this.embedder.info();
    ensureModelContract(this.manifest.e_model, info);

    logger.info(`InftKnowledgeService: loaded ${this.store.size()} chunks, model=${info.id}`);
  }

  /** Embed query offline and return top-k text chunks (+scores + meta). */
  async search(query: string, k = 5) {
    if (!this.store || !this.embedder || !this.manifest) {
      throw new Error("InftKnowledgeService not initialized. Call loadManifest(uri) first.");
    }
    const [q] = await this.embedder.embed([query], {
      mode: "query",
      instruction: "e5",
      normalize: true,
    });
    normalizeInPlace(q);
    return this.store.search(q, k);
  }
}
