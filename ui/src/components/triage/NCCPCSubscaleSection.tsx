/**
 * Project N: NCCPC Subscale Section Component.
 * Touch-optimized score buttons (0, 1, 2, 3, NA) for each clinical item.
 */

import React from "react";
import { NCCPCItem, NCCPCResponseValue, NCCPCSubscale } from "../../types/nccpc";

interface NCCPCSubscaleSectionProps {
  subscale: NCCPCSubscale;
  scores: Record<string, NCCPCResponseValue>;
  onSelectScore: (itemId: string, val: NCCPCResponseValue) => void;
}

const scoreButtons: Array<{ value: NCCPCResponseValue; label: string }> = [
  { value: 0, label: "0 (Not at all)" },
  { value: 1, label: "1 (Just little)" },
  { value: 2, label: "2 (Fairly often)" },
  { value: 3, label: "3 (Very often)" },
  { value: "NA", label: "NA" },
];

export const NCCPCSubscaleSection: React.FC<NCCPCSubscaleSectionProps> = ({
  subscale,
  scores,
  onSelectScore,
}) => {
  return (
    <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex flex-col gap-3">
      <h3 className="text-xs font-semibold text-sky-400 tracking-wide">
        {subscale.title}
      </h3>

      <div className="flex flex-col divide-y divide-slate-800/80">
        {subscale.items.map((item: NCCPCItem) => {
          const currentVal = scores[item.id];
          return (
            <div
              key={item.id}
              className="py-2.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs"
            >
              <span className="text-slate-300 font-medium sm:max-w-md">
                {item.description}
              </span>

              {/* Touch score buttons */}
              <div className="flex items-center gap-1 shrink-0">
                {scoreButtons.map((btn) => {
                  const isSelected = currentVal === btn.value;
                  return (
                    <button
                      key={String(btn.value)}
                      type="button"
                      onClick={() => onSelectScore(item.id, btn.value)}
                      className={`px-2 py-1 rounded text-[11px] font-mono transition-colors border ${
                        isSelected
                          ? btn.value === "NA"
                            ? "bg-slate-700 border-slate-500 text-white font-bold"
                            : btn.value >= 2
                            ? "bg-amber-600 border-amber-400 text-white font-bold"
                            : "bg-sky-600 border-sky-400 text-white font-bold"
                          : "bg-slate-800/80 hover:bg-slate-700/80 border-slate-700 text-slate-400"
                      }`}
                    >
                      {btn.value}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
