// Reusable embedding interfaces. You can plug in:
// - RemoteEmbedder: calls your existing /v1/embeddings server
// - MockEmbedder3D: tiny deterministic test embedder for the demo
//
// If you later want local embeddings, add a LocalEmbedder class that
// uses @xenova/transformers inside the process.

export type EmbedMode = "query" | "passage";
export type Instruction = "e5" | "none";

export interface EmbedOptions {
  mode?: EmbedMode;
  instruction?: Instruction;
  normalize?: boolean;
}

export interface IEmbedder {
  embed(texts: string[], opts?: EmbedOptions): Promise<Float32Array[]>;
}

/** Calls your TS embedding server's /v1/embeddings endpoint. */
export class RemoteEmbedder implements IEmbedder {
  constructor(private baseUrl: string) {}

  async embed(texts: string[], opts: EmbedOptions = {}): Promise<Float32Array[]> {
    const body = {
      input: texts,
      mode: opts.mode ?? "query",
      instruction: opts.instruction ?? "e5",
      normalize: opts.normalize ?? true,
      encoding_format: "base64",
    };
    const res = await fetch(`${this.baseUrl.replace(/\/$/, "")}/v1/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`RemoteEmbedder error: ${res.status} ${res.statusText} ${text}`);
    }
    const json = await res.json();
    // Expect { data: [{ embedding: base64 }, ...] }
    if (!json?.data || !Array.isArray(json.data)) throw new Error("Bad embedding response shape");

    return json.data.map((item: any) => {
      const b64 = String(item.embedding);
      const buf = Buffer.from(b64, "base64");
      return new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
    });
  }
}

/**
 * Mock 3D embedder for tests. It maps text to a 3D vector via a tiny
 * keyword-feature trick so we can demo retrieval without a model.
 * This ONLY exists for the test harness (sample-manifest.json uses dim=3).
 */
export class MockEmbedder3D implements IEmbedder {
  private vocab = ["feedback", "incentives", "policy"];

  async embed(texts: string[], _opts: EmbedOptions = {}): Promise<Float32Array[]> {
    return texts.map((t) => {
      const lower = t.toLowerCase();
      const v = new Float32Array(3);
      v[0] = this.count(lower, "feedback");
      v[1] = this.count(lower, "incentive");
      v[2] = this.count(lower, "policy");
      // normalize for cosine
      const n = Math.hypot(v[0], v[1], v[2]) || 1;
      v[0] /= n; v[1] /= n; v[2] /= n;
      return v;
    });
  }

  private count(text: string, token: string): number {
    // naive frequency count
    let c = 0, idx = 0;
    while ((idx = text.indexOf(token, idx)) !== -1) { c++; idx += token.length; }
    return c;
  }
}
