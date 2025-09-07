// Eliza Plugin object that registers our components.
// Add this plugin's name to your character.plugins or register via runtime.registerPlugin.

import { Plugin } from "@elizaos/core";
import { inftContextProvider } from "../eliza/providers/inft-context.provider";
import { inftLoadAction } from "./actions/inft-load.action";
import { ogChatAction } from "./actions/og-chat.action";
import { InftKnowledgeService } from "./services/inft-knowledge.service";
import { OgBrokerService } from "./services/og-broker.service";

const plugin: Plugin = {
  name: "plugin-inft-og",
  description: "iNFT knowledge retrieval + 0G inference integration",
  services: [InftKnowledgeService, OgBrokerService],
  providers: [inftContextProvider],
  actions: [inftLoadAction, ogChatAction],
  // evaluators: [], // optional later (e.g., fact extraction)
};

export default plugin;