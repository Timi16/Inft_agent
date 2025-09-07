// Action that invokes 0G model using the OgBrokerService.
// It composes a simple prompt from the current state and user message.

import { Action, IAgentRuntime, Memory, State, composePromptFromState } from "@elizaos/core";
import { OgBrokerService } from "../services/og-broker.service";

export const ogChatAction: Action = {
  name: "OG_CHAT",
  description: "Send the composed prompt to a 0G model and return the reply.",
  similes: ["ASK_0G", "OG_INFER", "OG_COMPLETION", "ASK"],

  validate: async (_runtime: IAgentRuntime, message: Memory) => {
    const t = (message?.content?.text || "").toLowerCase();
    // Only fire when user explicitly asks (keep control tight for now)
    return t.startsWith("ask ") || t.startsWith("/ask ") || t.includes(" og:");
  },

  handler: async (runtime: IAgentRuntime, message: Memory, state?: State) => {
    const svc = runtime.getService<OgBrokerService>(OgBrokerService.name);
    if (!svc) {
      return { success: false, error: "OgBrokerService not available (check plugin.services + env)" };
    }

    // Ensure we always pass a full `State`
    const safeState: State = state ?? { text: "", values: {}, data: {} };

    // Inject the user's message into `values` so the template can use {{message}}
    const stateWithMessage: State = {
      ...safeState,
      values: {
        ...(safeState.values ?? {}),
        message: message?.content?.text ?? "",
      },
    };

    const template = [
      "You are {{character.name}}.",
      "Adhere to the style and constraints.",
      "",
      "### Conversation",
      "{{state.text}}",
      "",
      "### Task",
      "{{message}}",
    ].join("\n");

    const composed = composePromptFromState({
      state: stateWithMessage,
      template,
    });

    const res = await svc.infer({
      prompt: composed,
      modelHint: process.env.OG_MODEL_HINT, // optional: e.g. "llama-3.3-70b-instruct"
    });

    return {
      success: true,
      text: res.text,
      data: { provider: res.provider, model: res.model, chatID: res.chatID },
    };
  },
};
