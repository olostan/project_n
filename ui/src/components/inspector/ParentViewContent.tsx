/**
 * Project N: Parent View Perspective Component.
 * Delivers warm, accessible English co-regulatory suggestions and gentle possibilities.
 */

import React from "react";
import { Check, Heart, Lightbulb, Sparkles } from "lucide-react";
import { NonDiagnosticBanner } from "../common/NonDiagnosticBanner";

interface ParentViewContentProps {
  observedSignals: string;
  precedentSummary: string;
  contextNotes?: string | null;
  gentlePossibilities: Array<{ action: string; reason: string }>;
  onOpenOutcomeLogger: () => void;
}

export const ParentViewContent: React.FC<ParentViewContentProps> = ({
  observedSignals,
  precedentSummary,
  contextNotes,
  gentlePossibilities,
  onOpenOutcomeLogger,
}) => {
  return (
    <div className="flex flex-col gap-4 text-slate-200">
      {/* Observed Signals Summary */}
      <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col gap-1.5">
        <div className="flex items-center gap-2 text-xs font-semibold text-sky-400">
          <Sparkles className="w-4 h-4" />
          <span>What We Noticed:</span>
        </div>
        <p className="text-sm text-slate-200 leading-relaxed">{observedSignals}</p>
      </div>

      {/* Comparable History / Precedent */}
      <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col gap-1.5">
        <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400">
          <Heart className="w-4 h-4" />
          <span>What Helped in Past Similar Moments:</span>
        </div>
        <p className="text-sm text-slate-200 leading-relaxed">{precedentSummary}</p>
        {contextNotes && (
          <p className="text-xs text-slate-400 mt-1 italic">Context: {contextNotes}</p>
        )}
      </div>

      {/* Gentle Co-Regulatory Possibilities to Explore */}
      <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col gap-2.5">
        <div className="flex items-center gap-2 text-xs font-semibold text-amber-400">
          <Lightbulb className="w-4 h-4" />
          <span>Gentle Things You Might Try:</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {gentlePossibilities.map((pos, idx) => (
            <div
              key={idx}
              className="p-3 rounded-lg bg-slate-800/70 border border-slate-700/80 flex flex-col gap-1"
            >
              <div className="flex items-center gap-1.5 font-medium text-slate-100 text-xs capitalize">
                <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                <span>{pos.action.replace(/_/g, " ")}</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-normal">{pos.reason}</p>
            </div>
          ))}
        </div>

        <div className="pt-2 flex justify-end">
          <button
            onClick={onOpenOutcomeLogger}
            className="px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-medium shadow-sm transition-colors"
          >
            Record What Happened
          </button>
        </div>
      </div>

      <NonDiagnosticBanner />
    </div>
  );
};
