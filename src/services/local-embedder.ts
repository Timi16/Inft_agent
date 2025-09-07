// Offline, in-process query embedder using @xenova/transformers.
// One-time model download required; after that it reads from cache.
// Set TRANSFORMERS_CACHE to a persistent folder.

export type EmbedMode = "query" | "passage";
export type Instruction = "e5" | "none";

export interface EmbedOptions {
  mode?: EmbedMode;
  instruction?: Instruction;
  normalize?: boolean;
}

export interface EmbedderInfo {
  id: string;
  dim: number | null;
}

export interface IEmbedder {
  embed(texts: string[], opts?: EmbedOptions): Promise<Float32Array[]>;
  info(): Promise<EmbedderInfo>;
}

export class LocalEmbedder implements IEmbedder {
  private lazy: Promise<any> | null = null;
  private modelDim: number | null = null;

  constructor(private modelId = process.env.MODEL_ID || "Xenova/bge-small-en-v1.5") {}

  private async getPipe() {
    if (!this.lazy) {
      this.lazy = import("@xenova/transformers").then(({ pipeline }) =>
        pipeline("feature-extraction", this.modelId)
      );
    }
    return this.lazy;
  }

  private withPrefix(texts: string[], mode: EmbedMode, instruction: Instruction) {
    if (instruction === "e5") {
      const prefix = mode === "query" ? "query: " : "passage: ";
      return texts.map((t) => prefix + t);
    }
    return texts;
  }

  async embed(texts: string[], opts: EmbedOptions = {}): Promise<Float32Array[]> {
    const mode = opts.mode ?? "query";
    const instruction = opts.instruction ?? "e5";
    const normalize = opts.normalize ?? true;

    const pipe = await this.getPipe();
    const prefixed = this.withPrefix(texts, mode, instruction);
    const out = await pipe(prefixed, { pooling: "mean", normalize });
    // transformers.js returns a Tensor with .data (Float32Array) and dims [n, d]
    const data: Float32Array = out.data;
    const dims: number[] = out.dims;
    const n = dims.length === 2 ? dims[0] : 1;
    const d = dims.length === 2 ? dims[1] : data.length;
    this.modelDim = d;
    const result: Float32Array[] = [];
    for (let i = 0; i < n; i++) {
      const slice = data.subarray(i * d, (i + 1) * d);
      result.push(new Float32Array(slice));
    }
    return result;
  }

  async info(): Promise<EmbedderInfo> {
    if (this.modelDim == null) {
      await this.embed(["warmup"], { mode: "query", instruction: "e5", normalize: true });
    }
    return { id: this.modelId, dim: this.modelDim };
  }
}
