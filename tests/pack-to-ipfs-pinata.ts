// tests/pack-to-ipfs-pinata.ts
// Convert a REAL inline manifest to external vectors blob (fp16), upload to Pinata,
// update manifest with vectors_uri + vectors_checksum, and save a new file.
//
// Env:
//   PINATA_JWT=... (required)
//   PINATA_GATEWAY=https://<sub>.mypinata.cloud/ipfs/ (recommended)
// Usage:
//   npx tsx tests/pack-to-ipfs-pinata.ts tests/real/manifest.inline.json tests/real/manifest.external.json

import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { InftManifestSchema } from "../src/schema/inft-manifest-schema";
import { encodeVectorsBlob } from "../src/services/binary.vector";
import { sha256Hex } from "../src/utils/hash";

// Your Pinata service wrapper (as you pasted)
import { IpfsService } from "../src/services/ipfs.service";

async function main() {
  const IN = process.argv[2] ?? "tests/sample-manifest.json";
  const OUT = process.argv[3] ?? "tests/sample-manifest.external.json";

  const raw = JSON.parse(await readFile(IN, "utf8"));
  const manifest = InftManifestSchema.parse(raw);

  if (!manifest.entries.every((e) => Array.isArray(e.embedding))) {
    throw new Error("All entries must have inline numeric embeddings to pack.");
  }

  // Gather vectors (Float32) in order
  const dim = manifest.model.dim;
  const vectors = manifest.entries.map((e) => new Float32Array(e.embedding as number[]));
  if (!vectors.every((v) => v.length === dim)) throw new Error("Dimension mismatch in embeddings.");

  // Encode fp16 blob
  const blob = encodeVectorsBlob(vectors, 1); // 1 = fp16
  const checksum = sha256Hex(blob);

  // Upload to Pinata
  const ipfs = new IpfsService(); // expects PINATA_JWT + PINATA_GATEWAY
  const up = await ipfs.addBytes(new Uint8Array(blob), "vectors.bin");
  const vectors_uri = up.uri;

  // Strip inline embeddings and write externalized manifest
  const external = {
    ...manifest,
    vectors_uri,
    vectors_checksum: checksum,
    model: { ...manifest.model, quantization: "fp16" as const },
    entries: manifest.entries.map(({ embedding, embedding_b64, ...rest }) => rest),
  };

  await mkdir(path.dirname(OUT), { recursive: true });
  await writeFile(OUT, JSON.stringify(external, null, 2), "utf8");
  console.log("vectors_uri:", vectors_uri);
  console.log("vectors_checksum:", checksum);
  console.log("Wrote:", OUT);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
