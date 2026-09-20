/**
 * Project N: Facts and Clinical Techniques API Client.
 */

import { request } from "./client";
import {
  ChildProfileFact,
  FactConfirmRequest,
  FactCreateRequest,
} from "../types/facts";

export interface FactFilterParams {
  category?: string;
  source_type?: string;
  confirmed_only?: boolean;
}

export async function fetchFacts(
  params: FactFilterParams = {}
): Promise<{ facts: ChildProfileFact[]; total: number }> {
  const query = new URLSearchParams();
  if (params.category) query.set("category", params.category);
  if (params.source_type) query.set("source_type", params.source_type);
  if (params.confirmed_only !== undefined) {
    query.set("confirmed_only", String(params.confirmed_only));
  }

  const qs = query.toString();
  return request<{ facts: ChildProfileFact[]; total: number }>(
    `/api/v1/facts${qs ? `?${qs}` : ""}`
  );
}

export async function createFact(
  data: FactCreateRequest
): Promise<ChildProfileFact> {
  return request<ChildProfileFact>("/api/v1/facts", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function confirmFact(
  factId: string,
  data: FactConfirmRequest
): Promise<{ fact_id: string; confirmed_by_caregiver: boolean }> {
  return request(`/api/v1/facts/${encodeURIComponent(factId)}/confirm`, {
    method: "POST",
    body: JSON.stringify(data),
  });
}
