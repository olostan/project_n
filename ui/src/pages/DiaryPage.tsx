/**
 * Project N: Historical Episode Diary Page.
 */

import React from "react";
import { Film, HeartPulse } from "lucide-react";
import { EpisodeCard } from "../components/diary/EpisodeCard";
import { EpisodeFilters } from "../components/diary/EpisodeFilters";
import { Button } from "../components/common/Button";
import { ControlledAntecedent, Episode } from "../types/episodes";

interface DiaryPageProps {
  episodes: Episode[];
  total: number;
  loading: boolean;
  antecedentFilter: ControlledAntecedent | "all";
  onSelectAntecedent: (ant: ControlledAntecedent | "all") => void;
  selectedEpisodeId: string | null;
  onSelectEpisodeId: (id: string) => void;
  onRefresh: () => void;
  onNavigateToInspector: () => void;
  onNavigateToTriage: (episodeId: string) => void;
}

export const DiaryPage: React.FC<DiaryPageProps> = ({
  episodes,
  total,
  loading,
  antecedentFilter,
  onSelectAntecedent,
  selectedEpisodeId,
  onSelectEpisodeId,
  onRefresh,
  onNavigateToInspector,
  onNavigateToTriage,
}) => {
  const selectedEpisode = episodes.find((e) => e.id === selectedEpisodeId) || episodes[0];

  return (
    <div className="flex flex-col gap-4 max-w-6xl mx-auto p-4 sm:p-6 text-slate-100">
      {/* Top Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-slate-100">Episodic Behavioral Diary</h2>
          <p className="text-xs text-slate-400">
            Historical log of verified episodes and recorded co-regulatory outcomes for Child N.
          </p>
        </div>
      </div>

      {/* Filter Controls */}
      <EpisodeFilters
        currentAntecedent={antecedentFilter}
        onSelectAntecedent={onSelectAntecedent}
        onRefresh={onRefresh}
        loading={loading}
        totalCount={total}
      />

      {/* Main Content Layout: Episode List + Selected Preview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Episode Card Grid (2 cols on lg) */}
        <div className="lg:col-span-2 flex flex-col gap-3 max-h-[calc(100vh-250px)] overflow-y-auto pr-1">
          {episodes.length === 0 && !loading && (
            <div className="p-8 rounded-xl bg-slate-900/60 border border-slate-800 text-center text-slate-500 text-xs">
              No episodes found matching the selected antecedent filter.
            </div>
          )}

          {episodes.map((ep) => (
            <EpisodeCard
              key={ep.id}
              episode={ep}
              isSelected={ep.id === selectedEpisodeId}
              onSelect={() => onSelectEpisodeId(ep.id)}
              onOpenTriage={() => onNavigateToTriage(ep.id)}
            />
          ))}
        </div>

        {/* Selected Episode Summary Sidebar */}
        <div className="flex flex-col gap-4">
          {selectedEpisode ? (
            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl flex flex-col gap-4 sticky top-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <span className="text-xs font-semibold text-slate-200">Selected Episode</span>
                <span className="text-[10px] font-mono text-slate-400">{selectedEpisode.id.slice(0, 8)}...</span>
              </div>

              <div className="flex flex-col gap-2 text-xs">
                <div>
                  <span className="text-slate-500 block text-[10px]">ANTECEDENT CONTEXT</span>
                  <strong className="text-slate-200 capitalize">
                    {selectedEpisode.antecedent_id.replace(/_/g, " ")}
                  </strong>
                </div>

                <div>
                  <span className="text-slate-500 block text-[10px]">ACTION LOG</span>
                  <strong className="text-sky-400 capitalize">
                    {selectedEpisode.action_performed?.replace(/_/g, " ") || selectedEpisode.action_offered.replace(/_/g, " ")}
                  </strong>
                </div>

                <div>
                  <span className="text-slate-500 block text-[10px]">OUTCOME</span>
                  <strong className="text-emerald-400 capitalize">
                    {selectedEpisode.outcome_state?.replace(/_/g, " ") || "Pending Confirmation"}
                  </strong>
                </div>

                {selectedEpisode.observed_f0_mean && (
                  <div className="pt-2 border-t border-slate-800 flex justify-between font-mono text-[11px]">
                    <span className="text-slate-500">Acoustic Pitch F₀:</span>
                    <span className="text-slate-200">{Math.round(selectedEpisode.observed_f0_mean)} Hz</span>
                  </div>
                )}
              </div>

              <div className="flex flex-col gap-2 pt-2 border-t border-slate-800">
                <Button
                  variant="primary"
                  size="md"
                  onClick={onNavigateToInspector}
                  className="gap-2 w-full"
                >
                  <Film className="w-4 h-4" />
                  <span>Open in Video Inspector</span>
                </Button>

                <Button
                  variant="secondary"
                  size="md"
                  onClick={() => onNavigateToTriage(selectedEpisode.id)}
                  className="gap-2 w-full text-slate-300"
                >
                  <HeartPulse className="w-4 h-4 text-amber-400" />
                  <span>Pain & NCCPC Triage</span>
                </Button>
              </div>
            </div>
          ) : (
            <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 text-center text-slate-500 text-xs">
              Select an episode to view quick details.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
