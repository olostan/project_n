/**
 * Project N: 2D Lexicon Cluster Hover Tooltip.
 */

import React from "react";
import { Episode } from "../../types/episodes";

interface ClusterTooltipProps {
  episode: Episode;
  x: number;
  y: number;
}

export const ClusterTooltip: React.FC<ClusterTooltipProps> = ({ episode, x, y }) => {
  return (
    <div
      className="absolute z-30 p-2.5 rounded-lg bg-slate-950/95 border border-slate-700 shadow-xl text-xs flex flex-col gap-1 pointer-events-none transform -translate-x-1/2 -translate-y-full mb-2 w-48"
      style={{ left: `${x}px`, top: `${y}px` }}
    >
      <div className="flex items-center justify-between border-b border-slate-800 pb-1">
        <span className="font-semibold text-sky-400 capitalize">
          {episode.antecedent_id.replace(/_/g, " ")}
        </span>
        <span className="text-[10px] text-slate-500 font-mono">
          {Math.round(episode.duration_ms / 1000)}s
        </span>
      </div>

      <div className="text-[11px] text-slate-300">
        <div>Action: <strong className="text-white capitalize">{episode.action_performed?.replace(/_/g, " ") || episode.action_offered.replace(/_/g, " ")}</strong></div>
        <div>Outcome: <strong className="text-emerald-400 capitalize">{episode.outcome_state?.replace(/_/g, " ") || "Unconfirmed"}</strong></div>
        {episode.observed_f0_mean && (
          <div>F₀: <strong className="text-slate-200">{Math.round(episode.observed_f0_mean)} Hz</strong></div>
        )}
      </div>

      <div className="text-[9px] text-sky-500/80 pt-0.5">Click dot to inspect episode</div>
    </div>
  );
};
