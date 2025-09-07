import "dotenv/config";
import { readFile } from "node:fs/promises";

/** Ensure we have a usable gateway URL (adds https://, trims, ensures trailing /) */
function normalizeGateway(input?: string): string {
  let gw = (input ?? process.env.PINATA_GATEWAY ?? "").trim();
  if (!gw) {
    throw new Error(
      "PINATA_GATEWAY not set. Put your domain (e.g. violet-...mypinata.cloud) in .env"
    );
  }
  if (!/^https?:\/\//i.test(gw)) {
    gw = "https://" + gw;
  }
  gw = gw.replace(/\/+$/, "") + "/"; // ensure single trailing slash
  return gw;
}

/** Resolve ipfs:// or ipns:// to your HTTP(S) gateway */
export function resolveToHttp(uri: string, gateway?: string): string {
  const gw = normalizeGateway(gateway);

  if (uri.startsWith("ipfs://")) {
    // ipfs://<cid>/...  ->  https://<gw>/ipfs/<cid>/...
    const rest = uri.slice("ipfs://".length).replace(/^ipfs\/+/i, "");
    return gw + "ipfs/" + rest.replace(/^\/+/, "");
  }

  if (uri.startsWith("ipns://")) {
    // ipns://<name>/... ->  https://<gw>/ipns/<name>/...
    const rest = uri.slice("ipns://".length).replace(/^ipns\/+/i, "");
    return gw + "ipns/" + rest.replace(/^\/+/, "");
  }

  // Already a gateway path like /ipfs/<cid> or /ipns/<name>
  if (uri.startsWith("/ipfs/") || uri.startsWith("/ipns/")) {
    return gw + uri.replace(/^\/+/, "");
  }

  // http(s) or anything else: leave unchanged
  return uri;
}

/** Fetch bytes from ipfs://, ipns://, /ipfs/, http(s)://, file://, or a bare local path. */
export async function fetchBytesSmart(uriOrPath: string, gateway?: string): Promise<Uint8Array> {
  if (
    uriOrPath.startsWith("ipfs://") ||
    uriOrPath.startsWith("ipns://") ||
    uriOrPath.startsWith("/ipfs/") ||
    uriOrPath.startsWith("/ipns/")
  ) {
    const url = resolveToHttp(uriOrPath, gateway);
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Failed to fetch via gateway: ${r.status} ${r.statusText}`);
    return new Uint8Array(await r.arrayBuffer());
  }

  if (uriOrPath.startsWith("http://") || uriOrPath.startsWith("https://")) {
    const r = await fetch(uriOrPath);
    if (!r.ok) throw new Error(`Failed to fetch HTTP(S): ${r.status} ${r.statusText}`);
    return new Uint8Array(await r.arrayBuffer());
  }

  if (uriOrPath.startsWith("file://")) {
    const p = uriOrPath.replace(/^file:\/\//, "");
    const buf = await readFile(p);
    return new Uint8Array(buf);
  }

  // bare filesystem path (relative/absolute)
  const buf = await readFile(uriOrPath);
  return new Uint8Array(buf);
}
