// 1) Loads a manifest with inline embeddings
// 2) Packs vectors into a binary blob (fp32 or fp16)
// 3) Uploads blob to IPFS
// 4) Rewrites manifest to use vectors_uri and removes inline embeddings
// 5) Uploads manifest JSON to IPFS
// Prints both URIs for minting (tokenURI = manifest ipfs://CID)

import { readFile, writeFile } from "node:fs/promises";
import { InftManifestSchema } from "../src/schema/inft-manifest-schema";
import { encodeVectorsBlob } from "../src/services/binary.vector";
import { IpfsService } from "../src/services/ipfs.service";

const QUANT: 0 | 1 = (process.env.QUANT === "fp16") ? 1 : 0; // default fp32

async function main() {
  const manifestPath = new URL("./sample-manifest.json", import.meta.url).pathname;
  const json = JSON.parse(await readFile(manifestPath, "utf8"));
  const manifest = InftManifestSchema.parse(json);

  // Collect vectors in entry order
  const vectors: Float32Array[] = manifest.entries.map((e, i) => {
    if (Array.isArray(e.embedding)) return new Float32Array(e.embedding);
    throw new Error(`Entry ${e.id} has no inline embedding[] for packing`);
  });

  // Encode blob
  const blob = encodeVectorsBlob(vectors, QUANT);
  console.log(`Encoded vectors blob: dim=${manifest.model.dim} count=${vectors.length} bytes=${blob.byteLength}`);

  // Upload blob to IPFS
  const ipfs = new IpfsService({
    apiUrl: process.env.IPFS_API_URL,          // e.g. http://127.0.0.1:5001/api/v0
    projectId: process.env.IPFS_PROJECT_ID,    // for Infura/Pinata if needed
    projectSecret: process.env.IPFS_PROJECT_SECRET,
  });

  const vectorsUp = await ipfs.addBytes(blob, "vectors.bin");
  console.log("vectors_uri:", vectorsUp.uri);

  // Rewrite manifest: remove inline embeddings, set vectors_uri
  const stripped = {
    ...manifest,
    model: {
      ...manifest.model,
      quantization: QUANT === 1 ? "fp16" : "fp32",
    },
    vectors_uri: vectorsUp.uri,
    entries: manifest.entries.map(({ embedding, embedding_b64, ...rest }) => rest),
  };

  // Upload manifest
  const manUp = await ipfs.addJSON(stripped);
  console.log("manifest_uri:", manUp.uri);

  // Optional: write local copy of the final manifest for debugging
  await writeFile(manifestPath.replace(".json", ".external.json"), JSON.stringify(stripped, null, 2), "utf8");
  console.log("Wrote:", manifestPath.replace(".json", ".external.json"));
}

main().catch((e) => { console.error(e); process.exit(1); });
