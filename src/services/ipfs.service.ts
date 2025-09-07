// src/services/ipfs.service.ts
import { PinataSDK } from "pinata";

export interface IpfsConfig {
  jwt?: string;
  gateway?: string;
}

export class IpfsService {
  private pinata: InstanceType<typeof PinataSDK>;

  constructor(cfg: IpfsConfig = {}) {
    const pinataJwt =process.env.PINATA_JWT!;
    const pinataGateway =process.env.PINATA_GATEWAY!;
    if (!pinataJwt) throw new Error("PINATA_JWT missing");
    if (!pinataGateway) throw new Error("PINATA_GATEWAY missing");

    this.pinata = new PinataSDK({ pinataJwt, pinataGateway });
  }

  async addBytes(bytes: Uint8Array, name = "vectors.bin") {
    const b64 = Buffer.from(bytes).toString("base64");
    const up = await this.pinata.upload.public.base64(b64).name(name);
    const cid = up.cid;
    return { cid, uri: `ipfs://${cid}` };
  }

  async addJSON(obj: Record<string, unknown>, name = "manifest.json") {
    const up = await this.pinata.upload.public.json(obj).name(name);
    const cid = up.cid;
    return { cid, uri: `ipfs://${cid}` };
  }
}
