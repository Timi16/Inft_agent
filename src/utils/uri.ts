// Resolve ipfs://CID[/path] to HTTP gateway URL.
export function resolveToHttp(uri: string, gateway = process.env.IPFS_GATEWAY || "https://ipfs.io/ipfs/"): string {
  if (uri.startsWith("ipfs://")) {
    const path = uri.replace("ipfs://", "");
    return gateway.replace(/\/+$/, "") + "/" + path.replace(/^\/+/, "");
  }
  return uri; // already http(s)
}