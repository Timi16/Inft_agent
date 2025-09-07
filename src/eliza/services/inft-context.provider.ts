// Adds retrieved snippets to Eliza's composed state BEFORE decisions.
// Keep it small: topK and an obvious "Retrieved Knowledge" section.

import { Provider, IAgentRuntime, Memory, State } from "@elizaos/core";
import { InftKnowledgeService } from "../services/inft-knowledge.service";

export const inftContextProvider: Provider = {
  name: "inft-context",
  description: "RAG provider: retrieves top-k snippets from the iNFT knowledge store.",
  position: -10, // run early

  get: async (runtime: IAgentRuntime, message: Memory, _state: State) => {
    const svc = runtime.getService<InftKnowledgeService>("inft-context" as any) 
              || runtime.getService<InftKnowledgeService>(InftKnowledgeService.serviceType as any)
              || (runtime.getService as any)?.(InftKnowledgeService.name);

    // In practice you'll register & fetch by class
    const store = runtime.getService<InftKnowledgeService>(InftKnowledgeService.name) || svc;
    if (!store) {
      // Provider returns nothing if not loaded yet.
      return { text: "", data: {} };
    }

    const userText = message?.content?.text || "";
    if (!userText.trim()) return { text: "", data: {} };

    const hits = await store.search(userText, Number(process.env.INFT_TOP_K || 4));

    const text = [
      "### Retrieved Knowledge",
      ...hits.map((h: { score: number; text: any; }, i: number) => `(${i + 1}) [${h.score.toFixed(3)}] ${h.text}`),
      "",
    ].join("\n");

    return {
      text,
      data: { hits },
      values: { inft_hits: hits },
    };
  },
};
