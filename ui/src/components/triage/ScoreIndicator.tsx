/**
 * Project N: NCCPC Running Score & Subscales Indicator.
 */

import React from "react";
import { AlertCircle, CheckCircle2 } from "lucide-react";

interface ScoreIndicatorProps {
  totalScore: number;
  cutoffThreshold: number;
  cutoffBreached: boolean;
  naCount: number;
  subscaleScores: Record<string, number>;
  instrument: string;
}

export const ScoreIndicator: React.FC<ScoreIndicatorProps> = ({
  totalScore,
  cutoffThreshold,
  cutoffBreached,
  naCount,
  subscaleScores,
  instrument,
}) => {
  return (
    <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex flex-col gap-3">
      {/* Top row: Total vs Cutoff */}
      <div className="flex items-center justify-between">
        <div>
          <span className="text-xs text-slate-400 block font-medium">Total Pain Score:</span>
          <div className="flex items-baseline gap-2">
            <span
              className={`text-3xl font-bold font-mono ${
                cutoffBreached ? "text-rose-400" : "text-sky-400"
              }`}
            >
              {totalScore}
            </span>
            <span className="text-xs text-slate-500 font-mono">
              / Cut-off: &ge;{cutoffThreshold}
            </span>
          </div>
        </div>

        <div>
          {cutoffBreached ? (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-950/80 border border-rose-700 text-rose-300 text-xs font-semibold">
              <AlertCircle className="w-4 h-4 text-rose-400" />
              <span>Pain Threshold Breached</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-950/80 border border-emerald-800 text-emerald-300 text-xs font-semibold">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Below Pain Cut-off</span>
            </div>
          )}
        </div>
      </div>

      {/* Subscales Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pt-2 border-t border-slate-800 text-xs font-mono">
        {Object.entries(subscaleScores).map(([subId, score]) => (
          <div
            key={subId}
            className="p-2 rounded-md bg-slate-800/60 border border-slate-700/60 flex items-center justify-between"
          >
            <span className="text-slate-400 text-[11px] capitalize">
              {subId.replace(/_/g, " ")}:
            </span>
            <strong className="text-slate-200">{score}</strong>
          </div>
        ))}
      </div>

      <div className="flex justify-between items-center text-[11px] text-slate-500 pt-1">
        <span>Instrument: {instrument.toUpperCase()}</span>
        <span>Items Marked NA: {naCount}</span>
      </div>
    </div>
  );
};
