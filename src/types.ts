
export type Quantization = "fp32" | "fp16" | "int8";

export interface InftModelCard {
  id: string;              // e.g., "Xenova/bge-small-en-v1.5" or "custom/debug"
  dim: number;             // vector dimension (e.g., 384/768/1024 or 3 in test)
  normalize: boolean;      // whether vectors are unit-length
  instruction: "e5" | "none";
  mode: "passage";         // passages are pre-embedded as "passage"
  quantization?: Quantization;
  checksum?: string;       // optional integrity tag for vectors blob(s)
}

export interface InftCharacter {
  id: string;              // uuid or slug
  name: string;
  username?: string;
  bio?: string | string[];
  system?: string;
  adjectives?: string[];
  topics?: string[];
  knowledge?: string[];    // refs (facts/files/dirs), optional
  messageExamples?: string[][];
  postExamples?: string[];
  style?: Record<string, unknown>;
  plugins?: string[];
  settings?: Record<string, unknown>;
  secrets?: Record<string, unknown>;
}

export interface InftEntry {
  id: string;              // unique id for this chunk
  type: "knowledge" | "example" | "post" | string;
  text: string;            // the chunk text
  meta?: Record<string, unknown>;
  // One of:
  embedding_b64?: string;  // base64(Float32Array) of length model.dim
  embedding?: number[];    // optional plain numbers for small sets
}

export interface InftManifest {
  version: string;         // "1.0"
  character: InftCharacter;
  model: InftModelCard;
  entries: InftEntry[];
  // Optional external binary/index artifacts for large corpora
  vectors_uri?: string;    // ipfs://... or https://... to a big binary
  vectors_index?: string;  // prebuilt HNSW index (optional future)
  license?: string;
}

// Search result shape returned by the store
export interface VectorHit {
  id: string;
  score: number;           // cosine similarity
  text: string;
  meta?: Record<string, unknown>;
}
