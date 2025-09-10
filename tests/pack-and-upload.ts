import "dotenv/config"; // <-- ensure .env is loaded
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { InftManifestSchema } from "../src/schema/inft-manifest-schema";
import { encodeVectorsBlob } from "../src/services/binary.vector";
import { IpfsService } from "../src/services/ipfs.service";
import { sha256Hex } from "../src/utils/hash";

const QUANT: 0 | 1 = (process.env.QUANT === "fp16") ? 1 : 0;

async function main() {
  const manifestUrl = new URL("./sample-manifest.json", import.meta.url);
  const manifestPath = fileURLToPath(manifestUrl);

  const json = JSON.parse(await readFile(manifestPath, "utf8"));
  const manifest = InftManifestSchema.parse(json);

  // Collect vectors
  const vectors: Float32Array[] = manifest.entries.map((e) => {
    if (Array.isArray(e.embedding)) return new Float32Array(e.embedding);
    if (e.embedding_b64) {
      const buf = Buffer.from(e.embedding_b64, "base64");
      return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    }
    throw new Error(`Entry ${e.id} has no embedding (embedding[] or embedding_b64)`);
  });

  const blob = encodeVectorsBlob(vectors, QUANT);
  console.log(`Encoded vectors blob: dim=${manifest.e_model.dim} count=${vectors.length} bytes=${blob.byteLength}`);

  const checksum = sha256Hex(blob);
  console.log("SHA-256 checksum:", checksum);
  // No args needed; service uses PINATA_JWT + PINATA_GATEWAY from .env
  const ipfs = new IpfsService();
  const vectorsUp = await ipfs.addBytes(blob, "vectors.bin");
  console.log("vectors_uri:", vectorsUp.uri);
  const stripped = {
    ...manifest,
    model: {
        ...manifest.model,
        quantization: QUANT === 1 ? "fp16" : "fp32",
    },
    vectors_uri: vectorsUp.uri,
    vectors_checksum: checksum,       
    entries: manifest.entries.map(({ embedding, embedding_b64, ...rest }) => rest),
    };

  const manUp = await ipfs.addJSON(stripped, "manifest.json");
  console.log("manifest_uri:", manUp.uri);

  const outPath = manifestPath.replace(".json", ".external.json");
  await writeFile(outPath, JSON.stringify(stripped, null, 2), "utf8");
  console.log("Wrote:", outPath);
}

main().catch((e) => { console.error(e); process.exit(1); });
