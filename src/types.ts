// src/types.ts
// Core types for iNFT manifest and runtime.

export type Quantization = "fp32" | "fp16" | "int8";

export interface InftModelCard {
  id: string;              // e.g. "Xenova/bge-small-en-v1.5"
  dim: number;             // 384/768/1024...
  normalize: boolean;      // passages are L2-normalized?
  instruction: "e5" | "none";
  mode: "passage";         // passages embedded as "passage"
  quantization?: Quantization;
  checksum?: string;
}

export interface InftCharacter {
  id: string;
  name: string;
  username?: string;
  bio?: string | string[];
  system?: string;
  adjectives?: string[];
  topics?: string[];
  knowledge?: string[];
  messageExamples?: string[][];
  postExamples?: string[];
  style?: Record<string, unknown>;
  plugins?: string[];
  settings?: Record<string, unknown>;
  secrets?: Record<string, unknown>;
}

export interface InftEntry {
  id: string;
  type: "knowledge" | "example" | "post" | string;
  text: string;
  meta?: Record<string, unknown>;
  embedding_b64?: string;  // base64(Float32Array dim)
  embedding?: number[];    // small inline arrays
}

export interface InftManifest {
  version: string;
  character: InftCharacter;
  model: InftModelCard;
  entries: InftEntry[];
  vectors_uri?: string;
  vectors_index?: string;
  vectors_checksum?: string;   // ← NEW
  license?: string;
}


export interface VectorHit {
  id: string;
  score: number;           // cosine
  text: string;
  meta?: Record<string, unknown>;
}
