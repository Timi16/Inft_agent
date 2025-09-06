// Minimal IPFS client using ipfs-http-client. Works with:
//  - Local IPFS node (set IPFS_API_URL=http://127.0.0.1:5001/api/v0)
//  - Hosted providers (Infura, Pinata) using Basic auth headers.
//
// Returns CID strings and ipfs:// URIs for convenience.

import { create, IPFSHTTPClient } from "ipfs-http-client";

export interface IpfsConfig {
  apiUrl?: string;    // e.g., http://127.0.0.1:5001/api/v0 or https://ipfs.infura.io:5001/api/v0
  projectId?: string; // for providers that need auth (Infura)
  projectSecret?: string;
}

export class IpfsService {
  private client: IPFSHTTPClient;

  constructor(cfg: IpfsConfig = {}) {
    const url = cfg.apiUrl || process.env.IPFS_API_URL || "http://127.0.0.1:5001/api/v0";
    const headers: Record<string, string> = {};
    if (cfg.projectId && cfg.projectSecret) {
      const token = Buffer.from(`${cfg.projectId}:${cfg.projectSecret}`).toString("base64");
      headers["Authorization"] = `Basic ${token}`;
    }
    this.client = create({ url, headers });
  }

  async addJSON(obj: unknown): Promise<{ cid: string; uri: string }> {
    const { cid } = await this.client.add(JSON.stringify(obj));
    const c = cid.toString();
    return { cid: c, uri: `ipfs://${c}` };
  }

  async addBytes(bytes: Uint8Array, path?: string): Promise<{ cid: string; uri: string }> {
    const { cid } = await this.client.add({ content: bytes, path });
    const c = cid.toString();
    return { cid: c, uri: `ipfs://${c}${path ? "/" + path : ""}` };
  }
}