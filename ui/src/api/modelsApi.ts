/**
 * Project N: Model Checkpoints & Promotion Gate API Client.
 */

import { request } from "./client";
import { ModelCheckpoint, PromotionResponse } from "../types/models";

export async function fetchModelCheckpoints(): Promise<{ checkpoints: ModelCheckpoint[] }> {
  return request<{ checkpoints: ModelCheckpoint[] }>("/api/v1/models/checkpoints");
}

export async function promoteCandidateModel(
  candidateId: string
): Promise<PromotionResponse> {
  return request<PromotionResponse>("/api/v1/models/promote", {
    method: "POST",
    body: JSON.stringify({ candidate_id: candidateId }),
  });
}

export async function rollbackProductionModel(): Promise<PromotionResponse> {
  return request<PromotionResponse>("/api/v1/models/rollback", {
    method: "POST",
    body: JSON.stringify({}),
  });
}
