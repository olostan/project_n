/**
 * Project N: Fact Library Filters Component.
 */

import React from "react";
import { Filter, Plus } from "lucide-react";
import { FactCategory } from "../../types/facts";
import { Button } from "../common/Button";

interface FactFiltersProps {
  selectedCategory: FactCategory | "all";
  onSelectCategory: (cat: FactCategory | "all") => void;
  confirmedOnly: boolean;
  onToggleConfirmedOnly: (val: boolean) => void;
  onOpenCreateModal: () => void;
}

const categories: Array<{ id: FactCategory | "all"; label: string }> = [
  { id: "all", label: "All Categories" },
  { id: "comfort_object", label: "Comfort Objects" },
  { id: "calming_cue", label: "Calming Cues" },
  { id: "sensory_trigger", label: "Sensory Triggers" },
  { id: "therapist_technique", label: "Therapist Techniques" },
  { id: "communication_routine", label: "Communication Routines" },
];

export const FactFilters: React.FC<FactFiltersProps> = ({
  selectedCategory,
  onSelectCategory,
  confirmedOnly,
  onToggleConfirmedOnly,
  onOpenCreateModal,
}) => {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-slate-900/80 border border-slate-800 rounded-lg text-xs">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <Filter className="w-3.5 h-3.5 text-slate-400" />
          <select
            value={selectedCategory}
            onChange={(e) => onSelectCategory(e.target.value as FactCategory | "all")}
            className="bg-slate-800 border border-slate-700 text-slate-200 rounded-md px-2.5 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-sky-500"
          >
            {categories.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </div>

        <label className="flex items-center gap-1.5 cursor-pointer text-slate-300 select-none">
          <input
            type="checkbox"
            checked={confirmedOnly}
            onChange={(e) => onToggleConfirmedOnly(e.target.checked)}
            className="rounded bg-slate-800 border-slate-700 text-sky-500 focus:ring-0 cursor-pointer"
          />
          <span>Confirmed Only</span>
        </label>
      </div>

      <Button variant="primary" size="sm" onClick={onOpenCreateModal} className="gap-1">
        <Plus className="w-3.5 h-3.5" />
        <span>Stage Technique</span>
      </Button>
    </div>
  );
};
