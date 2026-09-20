/**
 * Project N: Behavioral Lexicon Color Legend.
 */

import React from "react";

export const ACTION_CLUSTER_COLORS: Record<string, string> = {
  deep_pressure_proprioceptive: "#38bdf8", // Sky
  hydration_water: "#34d399", // Emerald
  quiet_refuge: "#a855f7", // Purple
  dimmed_lighting: "#818cf8", // Indigo
  vestibular_rocking: "#f59e0b", // Amber
  preferred_comfort_object: "#ec4899", // Pink
  open_observation: "#94a3b8", // Slate
  other_custom: "#64748b",
};

export const LexiconLegend: React.FC = () => {
  return (
    <div className="flex flex-wrap items-center gap-3 p-3 bg-slate-900/80 border border-slate-800 rounded-lg text-xs">
      <span className="font-semibold text-slate-300">Cluster Color Map:</span>
      {Object.entries(ACTION_CLUSTER_COLORS).map(([action, color]) => (
        <div key={action} className="flex items-center gap-1.5 text-[11px] text-slate-400 capitalize">
          <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
          <span>{action.replace(/_/g, " ")}</span>
        </div>
      ))}
    </div>
  );
};
