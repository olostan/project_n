/**
 * Project N: Four-Layer Insight Card Container & Perspective Switcher.
 * Enforces Invariants 6 & 8 with Parent View and Therapist View modes.
 */

import React, { useState } from "react";
import { AlertCircle, Eye, Microscope } from "lucide-react";
import { Episode, FourLayerCardData } from "../../types/episodes";
import { ParentViewContent } from "./ParentViewContent";
import { TherapistViewContent } from "./TherapistViewContent";

interface FourLayerCardProps {
  episode: Episode;
  cardData?: FourLayerCardData | null;
  onOpenOutcomeLogger: () => void;
}

export const FourLayerCard: React.FC<FourLayerCardProps> = ({
  episode,
  cardData,
  onOpenOutcomeLogger,
}) => {
  const [viewMode, setViewMode] = useState<"parent" | "therapist">("parent");

  // Fallback content when SSE four_layer_card is not yet generated
  const l1Text =
    cardData?.L1_measured ||
    `Vocalization Duration: ${Math.round(episode.duration_ms / 1000)}s, F₀ mean: ${
      episode.observed_f0_mean ? Math.round(episode.observed_f0_mean) + " Hz" : "Uncalibrated"
    }. Kinematic rhythm: ${
      episode.observed_motion_rhythm_hz
        ? episode.observed_motion_rhythm_hz.toFixed(1) + " Hz oscillation"
        : "Low motion"
    }.`;

  const l2Text =
    cardData?.L2_historical ||
    (episode.action_performed
      ? `Prior confirmed resolution: ${episode.action_performed.replace(/_/g, " ")} (${
          episode.outcome_state?.replace(/_/g, " ") || "recorded"
        }).`
      : "Episodic prototype indexing pending caregiver outcome confirmation.");

  const l3Text =
    cardData?.L3_context ||
    `Antecedent: ${episode.antecedent_id.replace(/_/g, " ")}. Notes: ${
      episode.antecedent_notes || "None logged"
    }.`;

  const l4Text =
    cardData?.L4_evidence ||
    "Breau et al. (2002) - NCCPC validated pain observation protocol. Van de Cruys et al. (2014) - sensory prediction error & motor stimming as uncertainty reduction.";

  const parentSignals =
    cardData?.parent_view_text ||
    `Child N showed elevated vocal pitch (${
      episode.observed_f0_mean ? Math.round(episode.observed_f0_mean) + " Hz" : "typical range"
    }) and rhythmic hand movement (${
      episode.observed_motion_rhythm_hz ? episode.observed_motion_rhythm_hz.toFixed(1) + " Hz" : "subtle"
    }).`;

  const parentPrecedent = episode.action_performed
    ? `In past episodes following ${episode.antecedent_id.replace(/_/g, " ")}, offering ${episode.action_performed.replace(/_/g, " ")} helped him settle.`
    : "Similar moments were noted during transitions; review past successful calming cues below.";

  const gentlePossibilities = [
    {
      action: "deep_pressure_proprioceptive",
      reason: "Firm compression or weighted blanket may provide calming proprioceptive input.",
    },
    {
      action: "hydration_water",
      reason: "Offer a familiar cup of water to assist physiological regulation.",
    },
    {
      action: "quiet_refuge",
      reason: "Reduce ambient room stimulation and lower lighting for 3 minutes.",
    },
  ];

  return (
    <div className="flex flex-col gap-3 rounded-2xl bg-slate-900 border border-slate-800 p-4 shadow-lg">
      {/* Header & Perspective Toggle */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-slate-100">Behavioral Insight Card</h2>
          <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
            Four-Layer Contract
          </span>
        </div>

        {/* View Switcher Toggle */}
        <div className="flex items-center p-0.5 bg-slate-950 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => setViewMode("parent")}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition-all font-medium ${
              viewMode === "parent"
                ? "bg-emerald-600 text-white shadow-xs"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>Parent View</span>
          </button>
          <button
            onClick={() => setViewMode("therapist")}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition-all font-medium ${
              viewMode === "therapist"
                ? "bg-sky-600 text-white shadow-xs"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Microscope className="w-3.5 h-3.5" />
            <span>Therapist View</span>
          </button>
        </div>
      </div>

      {/* Abstention alert if applicable */}
      {cardData?.abstained && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-950/40 border border-amber-800/60 text-amber-200 text-xs">
          <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold">Unrecognized Behavioral Pattern:</span> Metric distance exceeds calibrated confidence threshold. The system abstains from automated classification; open-ended caregiver observation is recommended.
          </div>
        </div>
      )}

      {/* Perspective Content */}
      {viewMode === "parent" ? (
        <ParentViewContent
          observedSignals={parentSignals}
          precedentSummary={parentPrecedent}
          contextNotes={episode.antecedent_notes}
          gentlePossibilities={gentlePossibilities}
          onOpenOutcomeLogger={onOpenOutcomeLogger}
        />
      ) : (
        <TherapistViewContent
          l1Measured={l1Text}
          l2Historical={l2Text}
          l3Context={l3Text}
          l4Evidence={l4Text}
          f0MeanHz={episode.observed_f0_mean}
          motionRhythmHz={episode.observed_motion_rhythm_hz}
        />
      )}
    </div>
  );
};
