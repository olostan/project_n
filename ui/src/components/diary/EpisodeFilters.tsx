/**
 * Project N: Episode Filter Controls Component.
 */

import React from "react";
import { Filter, RefreshCw } from "lucide-react";
import { ControlledAntecedent } from "../../types/episodes";
import { Button } from "../common/Button";

interface EpisodeFiltersProps {
  currentAntecedent: ControlledAntecedent | "all";
  onSelectAntecedent: (ant: ControlledAntecedent | "all") => void;
  onRefresh: () => void;
  loading: boolean;
  totalCount: number;
}

const antecedentOptions: Array<{ id: ControlledAntecedent | "all"; label: string }> = [
  { id: "all", label: "All Antecedents" },
  { id: "post_school_transition", label: "Post-School Transition" },
  { id: "mealtime", label: "Mealtime" },
  { id: "bedtime_routine", label: "Bedtime Routine" },
  { id: "loud_environment", label: "Loud Environment" },
  { id: "unfamiliar_setting", label: "Unfamiliar Setting" },
  { id: "physical_transition", label: "Physical Transition" },
  { id: "preferred_activity_ended", label: "Preferred Activity Ended" },
  { id: "unknown", label: "Unknown / Spontaneous" },
];

export const EpisodeFilters: React.FC<EpisodeFiltersProps> = ({
  currentAntecedent,
  onSelectAntecedent,
  onRefresh,
  loading,
  totalCount,
}) => {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-slate-900/80 border border-slate-800 rounded-lg text-xs">
      <div className="flex items-center gap-2">
        <Filter className="w-3.5 h-3.5 text-slate-400" />
        <span className="font-medium text-slate-300">Filter Antecedent:</span>
        <select
          value={currentAntecedent}
          onChange={(e) =>
            onSelectAntecedent(e.target.value as ControlledAntecedent | "all")
          }
          className="bg-slate-800 border border-slate-700 text-slate-200 rounded-md px-2.5 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-sky-500"
        >
          {antecedentOptions.map((opt) => (
            <option key={opt.id} value={opt.id}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-3">
        <span className="text-slate-400 font-mono">
          {totalCount} {totalCount === 1 ? "episode" : "episodes"} recorded
        </span>
        <Button
          variant="secondary"
          size="sm"
          onClick={onRefresh}
          loading={loading}
          title="Refresh episodes"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </Button>
      </div>
    </div>
  );
};
