/**
 * Project N: Episodes Data Hook.
 * Handles filterable episode pagination, selection, and cache invalidation.
 */

import { useCallback, useEffect, useState } from "react";
import { fetchEpisodes, fetchEpisode } from "../api/episodesApi";
import { ControlledAntecedent, Episode } from "../types/episodes";

export function useEpisodes() {
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [antecedentFilter, setAntecedentFilter] = useState<ControlledAntecedent | "all">("all");
  const [selectedEpisodeId, setSelectedEpisodeId] = useState<string | null>(null);
  const [selectedEpisode, setSelectedEpisode] = useState<Episode | null>(null);
  const [loadingSelected, setLoadingSelected] = useState<boolean>(false);

  const loadEpisodes = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchEpisodes({
        limit: 50,
        offset: 0,
        antecedent_id: antecedentFilter === "all" ? undefined : antecedentFilter,
      });
      setEpisodes(res.episodes);
      setTotal(res.total);
      if (!selectedEpisodeId && res.episodes.length > 0) {
        setSelectedEpisodeId(res.episodes[0].id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load episodes.");
    } finally {
      setLoading(false);
    }
  }, [antecedentFilter, selectedEpisodeId]);

  useEffect(() => {
    loadEpisodes();
  }, [loadEpisodes]);

  // Load full episode details when selection changes
  useEffect(() => {
    if (!selectedEpisodeId) {
      setSelectedEpisode(null);
      return;
    }
    let isCurrent = true;
    setLoadingSelected(true);
    fetchEpisode(selectedEpisodeId)
      .then((ep) => {
        if (isCurrent) setSelectedEpisode(ep);
      })
      .catch((err) => {
        if (isCurrent) console.error("Failed to load episode details", err);
      })
      .finally(() => {
        if (isCurrent) setLoadingSelected(false);
      });

    return () => {
      isCurrent = false;
    };
  }, [selectedEpisodeId]);

  return {
    episodes,
    total,
    loading,
    error,
    antecedentFilter,
    setAntecedentFilter,
    selectedEpisodeId,
    setSelectedEpisodeId,
    selectedEpisode,
    loadingSelected,
    refetch: loadEpisodes,
  };
}
