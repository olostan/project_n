/**
 * Project N: NCCPC Pain Checklist Scoring API Client.
 */

import { request } from "./client";
import { NCCPCResponseValue, NCCPCScoreResponse } from "../types/nccpc";

export interface NCCPCSubmissionPayload {
  instrument: "nccpc_pv" | "nccpc_r";
  scores: Record<string, NCCPCResponseValue>;
  duration_min?: number;
}

export async function submitNCCPCChecklist(
  episodeId: string,
  payload: NCCPCSubmissionPayload
): Promise<NCCPCScoreResponse> {
  return request<NCCPCScoreResponse>(
    `/api/v1/episodes/${encodeURIComponent(episodeId)}/nccpc`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}
