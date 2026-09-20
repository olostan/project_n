/**
 * Project N: Episodes API Client.
 */

import { request } from "./client";
import {
  Episode,
  EpisodeListResponse,
  EpisodeOutcomeRequest,
} from "../types/episodes";

export interface EpisodeFilterParams {
  limit?: number;
  offset?: number;
  antecedent_id?: string;
}

export async function fetchEpisodes(
  params: EpisodeFilterParams = {}
): Promise<EpisodeListResponse> {
  const query = new URLSearchParams();
  if (params.limit !== undefined) query.set("limit", String(params.limit));
  if (params.offset !== undefined) query.set("offset", String(params.offset));
  if (params.antecedent_id) query.set("antecedent_id", params.antecedent_id);

  const qs = query.toString();
  return request<EpisodeListResponse>(`/api/v1/episodes${qs ? `?${qs}` : ""}`);
}

export async function fetchEpisode(episodeId: string): Promise<Episode> {
  return request<Episode>(`/api/v1/episodes/${encodeURIComponent(episodeId)}`);
}

export function getEpisodeMediaUrl(episodeId: string): string {
  return `/api/v1/episodes/${encodeURIComponent(episodeId)}/media`;
}

export async function confirmEpisodeOutcome(
  episodeId: string,
  data: EpisodeOutcomeRequest
): Promise<{ episode_id: string; status: string; indexed_in_vector_store: boolean }> {
  return request(`/api/v1/episodes/${encodeURIComponent(episodeId)}/outcome`, {
    method: "POST",
    body: JSON.stringify(data),
  });
}
