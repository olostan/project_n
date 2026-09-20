/**
 * Project N: Episode Diary Card Component.
 */

import React from "react";
import { AlertTriangle, Clock, Film, HeartPulse } from "lucide-react";
import { Episode } from "../../types/episodes";
import { Badge } from "../common/Badge";
import { OutcomeBadge } from "./OutcomeBadge";

interface EpisodeCardProps {
  episode: Episode;
  isSelected: boolean;
  onSelect: () => void;
  onOpenTriage: () => void;
}

export const EpisodeCard: React.FC<EpisodeCardProps> = ({
  episode,
  isSelected,
  onSelect,
  onOpenTriage,
}) => {
  const durationSec = Math.round(episode.duration_ms / 1000);
  const formattedDate = new Date(episode.captured_at).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  const isPainBreached = episode.pain_cutoff_breached === 1;

  return (
    <div
      onClick={onSelect}
      className={`p-4 rounded-xl border transition-all cursor-pointer flex flex-col gap-3 ${
        isSelected
          ? "bg-slate-800/90 border-sky-500/80 shadow-md ring-1 ring-sky-500/50"
          : "bg-slate-900/70 border-slate-800 hover:border-slate-700 hover:bg-slate-850"
      }`}
    >
      {/* Top row: Timestamp & Outcome */}
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5 text-slate-400 font-mono text-[11px]">
          <Clock className="w-3.5 h-3.5 text-slate-500" />
          <span>{formattedDate}</span>
          <span>({durationSec}s)</span>
        </div>
        <div className="flex items-center gap-1.5">
          {isPainBreached && (
            <Badge variant="danger" size="sm" className="gap-1">
              <AlertTriangle className="w-3 h-3" />
              Pain Flag
            </Badge>
          )}
          <OutcomeBadge outcome={episode.outcome_state} />
        </div>
      </div>

      {/* Middle row: Antecedent and Action */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-slate-200 text-sm capitalize">
            {episode.antecedent_id.replace(/_/g, " ")}
          </span>
          <span className="text-slate-500 text-xs">•</span>
          <span className="text-sky-400 text-xs font-medium">
            {episode.action_performed
              ? `Action: ${episode.action_performed.replace(/_/g, " ")}`
              : `Offered: ${episode.action_offered.replace(/_/g, " ")}`}
          </span>
        </div>
        {episode.antecedent_notes && (
          <p className="text-xs text-slate-400 line-clamp-2 italic">
            "{episode.antecedent_notes}"
          </p>
        )}
      </div>

      {/* Bottom metrics row: F0, motion rhythm, triage button */}
      <div className="flex items-center justify-between pt-2 border-t border-slate-800/80 text-[11px] text-slate-400 font-mono">
        <div className="flex items-center gap-3">
          {episode.observed_f0_mean !== null && (
            <span title="Mean Acoustic Pitch (F0)">
              F₀: <strong className="text-slate-200">{Math.round(episode.observed_f0_mean)} Hz</strong>
            </span>
          )}
          {episode.observed_motion_rhythm_hz !== null && (
            <span title="Kinematic Rhythm Frequency">
              Motion: <strong className="text-slate-200">{episode.observed_motion_rhythm_hz.toFixed(1)} Hz</strong>
            </span>
          )}
        </div>

        <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
          <button
            onClick={onOpenTriage}
            className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-amber-400 px-2 py-1 rounded hover:bg-slate-800 transition-colors"
            title="Perform or view NCCPC pain screening"
          >
            <HeartPulse className="w-3.5 h-3.5" />
            <span>NCCPC</span>
          </button>
          <button
            onClick={onSelect}
            className="flex items-center gap-1 text-[11px] text-sky-400 hover:text-sky-300 px-2 py-1 rounded hover:bg-slate-800 transition-colors"
          >
            <Film className="w-3.5 h-3.5" />
            <span>Inspect</span>
          </button>
        </div>
      </div>
    </div>
  );
};
