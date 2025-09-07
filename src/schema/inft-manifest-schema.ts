// src/schema/inft-manifest.schema.ts
import { z } from "zod";

export const InftModelCardSchema = z.object({
  id: z.string(),
  dim: z.number().int().positive(),
  normalize: z.boolean(),
  instruction: z.enum(["e5", "none"]),
  mode: z.literal("passage"),
  quantization: z.enum(["fp32", "fp16", "int8"]).optional(),
  checksum: z.string().optional(),
});

export const InftEntrySchema = z.object({
  id: z.string(),
  type: z.string(),
  text: z.string(),
  meta: z.record(z.string(), z.unknown()).optional(),      
  embedding_b64: z.string().optional(),
  embedding: z.array(z.number()).optional(),
}).refine(
  (e) => !!e.embedding_b64 || !!e.embedding,
  { message: "Entry must include either embedding_b64 or embedding[]" }
);

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

export const InftManifestSchema = z.object({
  version: z.string(),
  character: InftCharacterSchema,
  model: InftModelCardSchema,
  entries: z.array(InftEntrySchema).min(1),
  vectors_uri: z.string().optional(),
  vectors_index: z.string().optional(),
  vectors_checksum: z.string().optional(),   // ← NEW
  license: z.string().optional(),
});

export type InftManifestParsed = z.infer<typeof InftManifestSchema>;
