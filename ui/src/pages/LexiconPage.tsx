/**
 * Project N: 2D Behavioral Lexicon Visualizer Page.
 */

import React, { useState } from "react";
import { Compass, Film } from "lucide-react";
import { Episode } from "../types/episodes";
import { UMAPCanvas } from "../components/lexicon/UMAPCanvas";
import { LexiconLegend } from "../components/lexicon/LexiconLegend";
import { Button } from "../components/common/Button";
import { NonDiagnosticBanner } from "../components/common/NonDiagnosticBanner";

interface LexiconPageProps {
  episodes: Episode[];
  onSelectEpisodeForInspector: (episodeId: string) => void;
}

export const LexiconPage: React.FC<LexiconPageProps> = ({
  episodes,
  onSelectEpisodeForInspector,
}) => {
  const [selectedEpisode, setSelectedEpisode] = useState<Episode | null>(
    episodes.length > 0 ? episodes[0] : null
  );

  return (
    <div className="flex flex-col gap-5 max-w-6xl mx-auto p-4 sm:p-6 text-slate-100">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Compass className="w-5 h-5 text-sky-400" />
            <h2 className="text-base font-semibold text-slate-100">
              2D Behavioral Lexicon Visualizer
            </h2>
          </div>
          <p className="text-xs text-slate-400">
            Interactive UMAP geometry of Child N's 128-dimensional multimodal communication embeddings.
          </p>
        </div>
      </div>

      <LexiconLegend />

      {/* Main Visualizer & Details Drawer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Canvas Plot (8 cols) */}
        <div className="lg:col-span-8 flex flex-col gap-2">
          <UMAPCanvas
            episodes={episodes}
            selectedEpisodeId={selectedEpisode?.id || null}
            onSelectEpisode={setSelectedEpisode}
          />
          <span className="text-[11px] text-slate-500 font-mono">
            Projection: Metric Space (128d &rarr; 2D) • Total Episodes: {episodes.length}
          </span>
        </div>

        {/* Selected Cluster Info Drawer (4 cols) */}
        <div className="lg:col-span-4 flex flex-col gap-4">
          {selectedEpisode ? (
            <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl flex flex-col gap-3 text-xs">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="font-semibold text-slate-200">Episode Cluster Point</span>
                <span className="font-mono text-slate-500 text-[10px]">{selectedEpisode.id.slice(0, 8)}...</span>
              </div>

              <div className="flex flex-col gap-2">
                <div>
                  <span className="text-slate-500 block text-[10px]">ANTECEDENT CONTEXT</span>
                  <strong className="text-slate-200 capitalize">
                    {selectedEpisode.antecedent_id.replace(/_/g, " ")}
                  </strong>
                </div>

                <div>
                  <span className="text-slate-500 block text-[10px]">ACTION RESOLUTION</span>
                  <strong className="text-sky-400 capitalize">
                    {selectedEpisode.action_performed?.replace(/_/g, " ") || selectedEpisode.action_offered.replace(/_/g, " ")}
                  </strong>
                </div>

                <div>
                  <span className="text-slate-500 block text-[10px]">OBSERVED OUTCOME</span>
                  <strong className="text-emerald-400 capitalize">
                    {selectedEpisode.outcome_state?.replace(/_/g, " ") || "Pending Confirmation"}
                  </strong>
                </div>

                {selectedEpisode.observed_f0_mean && (
                  <div>
                    <span className="text-slate-500 block text-[10px]">ACOUSTIC PITCH MEAN</span>
                    <strong className="text-slate-300 font-mono">
                      {Math.round(selectedEpisode.observed_f0_mean)} Hz
                    </strong>
                  </div>
                )}
              </div>

              <div className="pt-2 border-t border-slate-800">
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => onSelectEpisodeForInspector(selectedEpisode.id)}
                  className="w-full gap-2"
                >
                  <Film className="w-3.5 h-3.5" />
                  <span>Inspect Multimodal Media</span>
                </Button>
              </div>
            </div>
          ) : (
            <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 text-center text-slate-500 text-xs">
              Hover or click any dot on the map to view episode details.
            </div>
          )}
        </div>
      </div>

      <NonDiagnosticBanner />
    </div>
  );
};
