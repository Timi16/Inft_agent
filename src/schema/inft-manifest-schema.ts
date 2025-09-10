import { z } from "zod";

export const InftModelEmbedSchema = z.object({
  id: z.string(),
  dim: z.number().int().positive(),
  normalize: z.boolean(),
  instruction: z.enum(["e5", "none"]),
  mode: z.literal("passage"),
  quantization: z.enum(["fp32", "fp16", "int8"]).optional(),
  checksum: z.string().optional(),
});

export const InftModelSchema = z.object({
  providerId: z.string(),
  name: z.number().int().positive().optional(),
});

// 1) Remove the refine here (allow entries without inline vectors)
export const InftEntrySchema = z.object({
  id: z.string(),
  type: z.string(),
  text: z.string(),
  meta: z.record(z.string(), z.unknown()).optional(),
  embedding_b64: z.string().optional(),
  embedding: z.array(z.number()).optional(),
});

export const InftCharacterSchema = z.object({
  id: z.string(),
  name: z.string(),
  username: z.string().optional(),
  bio: z.union([z.string(), z.array(z.string())]).optional(),
  system: z.string().optional(),
  adjectives: z.array(z.string()).optional(),
  topics: z.array(z.string()).optional(),
  knowledge: z.array(z.string()).optional(),
  messageExamples: z.array(z.array(z.string())).optional(),
  postExamples: z.array(z.string()).optional(),
  style: z.record(z.string(), z.unknown()).optional(),
  plugins: z.array(z.string()).optional(),
  settings: z.record(z.string(), z.unknown()).optional(),
  secrets: z.record(z.string(), z.unknown()).optional(),
});

// 2) Enforce “inline vectors required” ONLY when there is no external vectors URI/index
export const InftManifestSchema = z.object({
  version: z.string(),
  character: InftCharacterSchema,
  model: InftModelSchema,
  e_model: InftModelEmbedSchema,
  entries: z.array(InftEntrySchema).min(1),
  vectors_uri: z.string().optional(),
  vectors_index: z.string().optional(),
  license: z.string().optional(),
}).superRefine((m, ctx) => {
  const usesExternal = Boolean(m.vectors_uri || m.vectors_index);
  if (!usesExternal) {
    m.entries.forEach((e, i) => {
      if (!e.embedding && !e.embedding_b64) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["entries", i],
          message: "Entry must include either embedding_b64 or embedding[] (no vectors_uri provided)",
        });
      }
    });
  }
});

export type InftManifestParsed = z.infer<typeof InftManifestSchema>;
