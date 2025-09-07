// Action to load an iNFT manifest by URI (ipfs://, https://, file://).
// Usage example in chat: "inft load ipfs://<CID>/manifest.json"

import { Action, IAgentRuntime, Memory, State } from "@elizaos/core";
import { InftKnowledgeService } from "../services/inft-knowledge.service";

export const inftLoadAction: Action = {
  name: "INFT_LOAD",
  description: "Load an iNFT manifest and vectors into the knowledge store.",
  similes: ["LOAD_INFT", "LOAD_MANIFEST", "LOAD_KB"],

  validate: async (_runtime: IAgentRuntime, message: Memory) => {
    const text = message?.content?.text || "";
    return /(^|\s)inft\s+load\s+/i.test(text) || /(^|\s)load\s+(ipfs:\/\/|https?:\/\/|file:\/\/)/i.test(text);
  },

  handler: async (runtime: IAgentRuntime, message: Memory, _state?: State) => {
    const text = message?.content?.text || "";
    const m = text.match(/(ipfs:\/\/\S+|https?:\/\/\S+|file:\/\/\S+)/i);
    if (!m) {
      return { success: false, error: "No manifest URI found. Example: inft load ipfs://<CID>/manifest.json" };
    }
    const uri = m[1];

    let svc = runtime.getService<InftKnowledgeService>(InftKnowledgeService.name);
    if (!svc) {
      // Eliza runtime constructs services from plugin.services; but guard anyway.
      svc = await InftKnowledgeService.start(runtime);
    }

    await svc.loadManifest(uri);

    return {
      success: true,
      text: `Loaded iNFT manifest from ${uri}`,
      values: { inft_loaded: true, inft_manifest_uri: uri },
    };
  },
};
