// tests/build-manifest-from-folder.ts
// Build a REAL iNFT manifest from a folder of .md/.txt files.
// Usage (LOCAL embedder):
//   set MODEL_ID=Xenova/bge-small-en-v1.5
//   set TRANSFORMERS_CACHE=C:\models\transformers
//   npx tsx tests/build-manifest-from-folder.ts tests/data real/manifest.inline.json
//
// Usage (REMOTE embedder hitting your /v1/embeddings):
//   set EMBED_URL=http://127.0.0.1:8080
//   set USE_REMOTE=1
//   npx tsx tests/build-manifest-from-folder.ts tests/data real/manifest.inline.json

import { readdir, readFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { InftManifestSchema } from "../src/schema/inft-manifest-schema";
import type { InftManifest, InftEntry } from "../src/types";
import { LocalEmbedder } from "../src/services/local-embedder";
import { RemoteEmbedder } from "../src/services/embedding.service";
import { normalizeInPlace } from "../src/utils/cosine";

// ------------ Config --------------
const INPUT_DIR = process.argv[2] ?? "tests/data";
const OUT_PATH = process.argv[3] ?? "tests/real/manifest.inline.json";
const BATCH = Number(process.env.BATCH ?? 64);
const MODEL_ID = process.env.MODEL_ID || "Xenova/bge-small-en-v1.5";
const USE_REMOTE = process.env.USE_REMOTE === "1";
const EMBED_URL = process.env.EMBED_URL || "http://127.0.0.1:8080";

// Chunking config (simple, robust for prod baseline)
const MAX_CHARS = Number(process.env.MAX_CHARS ?? 2500); // ~ around 500 tokens
const OVERLAP = Number(process.env.OVERLAP ?? 300);

// ------------ Helpers --------------
function isDoc(f: string) {
  return f.endsWith(".md") || f.endsWith(".txt");
}
function chunkText(s: string, max = MAX_CHARS, overlap = OVERLAP): string[] {
  const chunks: string[] = [];
  let i = 0;
  const n = s.length;
  while (i < n) {
    const end = Math.min(i + max, n);
    let cut = end;
    // try to cut on paragraph boundary
    const back = s.lastIndexOf("\n\n", end);
    if (back > i + 200) cut = back + 2;
    chunks.push(s.slice(i, cut).trim());
    i = Math.max(cut - overlap, i + 1);
  }
  return chunks.filter(Boolean);
}

async function* walk(dir: string): AsyncGenerator<string> {
  const entries = await readdir(dir, { withFileTypes: true });
  for (const e of entries) {
    const fp = path.join(dir, e.name);
    if (e.isDirectory()) yield* walk(fp);
    else if (isDoc(e.name)) yield fp;
  }
}

// ------------ Main --------------
async function main() {
  const embedder = USE_REMOTE ? new RemoteEmbedder(EMBED_URL) : new LocalEmbedder(MODEL_ID);

  // Gather chunks
  const passages: { id: string; text: string; meta: any }[] = [];
  for await (const fp of walk(INPUT_DIR)) {
    const rel = path.relative(INPUT_DIR, fp);
    const txt = await readFile(fp, "utf8");
    const chs = chunkText(txt);
    chs.forEach((t, idx) => {
      passages.push({
        id: `${rel}#${idx}`,
        text: t,
        meta: { source: rel, idx },
      });
    });
  }
  if (passages.length === 0) throw new Error(`No .md/.txt found in ${INPUT_DIR}`);
  console.log(`Collected ${passages.length} chunks from ${INPUT_DIR}`);

  // Embed passages in batches (mode: passage, instruction: e5, normalize: true)
  const vectors: Float32Array[] = [];
  for (let i = 0; i < passages.length; i += BATCH) {
    const batch = passages.slice(i, i + BATCH);
    const vecs = await embedder.embed(
      batch.map((p) => p.text),
      { mode: "passage", instruction: "e5", normalize: true }
    );
    vecs.forEach((v) => normalizeInPlace(v)); // just to be safe
    vectors.push(...vecs);
    process.stdout.write(`Embedded ${Math.min(i + BATCH, passages.length)}/${passages.length}\r`);
  }
  process.stdout.write("\n");

  // Determine model dim
  const dim = vectors[0].length;

  // Build inline manifest (production: good for small/medium)
  const entries: InftEntry[] = passages.map((p, i) => ({
    id: p.id,
    type: "knowledge",
    text: p.text,
    meta: p.meta,
    embedding: Array.from(vectors[i]), // inline numeric (you can switch to base64 if you prefer)
  }));

  const manifest: InftManifest = {
    version: "1.0",
    character: {
      id: path.basename(INPUT_DIR),
      name: path.basename(INPUT_DIR),
      topics: [],
      settings: { k: 8 },
    },
    model: {
      id: USE_REMOTE ? MODEL_ID : (await new LocalEmbedder(MODEL_ID).info()).id || MODEL_ID,
      dim,
      normalize: true,
      instruction: "e5",
      mode: "passage",
      quantization: "fp32",
    },
    entries,
  };

  // Validate & write
  const parsed = InftManifestSchema.parse(manifest);
  await mkdir(path.dirname(OUT_PATH), { recursive: true });
  await writeFile(OUT_PATH, JSON.stringify(parsed, null, 2), "utf8");
  console.log(`Wrote manifest: ${OUT_PATH}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
