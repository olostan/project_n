/**
 * Project N: Clips API Client.
 */

import { request } from "./client";

export interface AnalyzeClipParams {
  antecedent_id: string;
  antecedent_notes?: string;
  caregiver_hypothesis?: string;
  setting?: string;
  observer?: string;
  captured_at?: string;
  generate_render?: boolean;
}

export interface AnalyzeClipResponse {
  task_id: string;
  stream_url: string;
}

export async function triggerClipAnalysis(
  clipId: string,
  params: AnalyzeClipParams
): Promise<AnalyzeClipResponse> {
  return request<AnalyzeClipResponse>(`/api/v1/clips/${encodeURIComponent(clipId)}/analyze`, {
    method: "POST",
    body: JSON.stringify(params),
  });
}
