/**
 * Project N: Child Profile Fact & Clinical Technique Item Card.
 */

import React from "react";
import { Check, Clock, ThumbsUp, X } from "lucide-react";
import { ChildProfileFact } from "../../types/facts";
import { Badge } from "../common/Badge";
import { Button } from "../common/Button";

interface FactItemCardProps {
  fact: ChildProfileFact;
  onConfirm: (factId: string, confirmed: boolean) => void;
  confirming: boolean;
}

export const FactItemCard: React.FC<FactItemCardProps> = ({
  fact,
  onConfirm,
  confirming,
}) => {
  const isConfirmed = fact.confirmed_by_caregiver === 1;

  return (
    <div
      className={`p-4 rounded-xl border flex flex-col gap-3 transition-colors ${
        isConfirmed
          ? "bg-slate-900/80 border-slate-800"
          : "bg-slate-900/95 border-amber-500/50 shadow-md ring-1 ring-amber-500/20"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-semibold text-slate-100">{fact.fact_title}</h4>
            <Badge variant={isConfirmed ? "success" : "warning"} size="sm">
              {isConfirmed ? "Confirmed" : "Needs Review"}
            </Badge>
          </div>
          <span className="text-[11px] font-mono text-sky-400 capitalize">
            {fact.category.replace(/_/g, " ")} • Source: {fact.source_type.replace(/_/g, " ")}
          </span>
        </div>

        {/* Action buttons for unconfirmed technique */}
        {!isConfirmed && (
          <div className="flex items-center gap-1.5 shrink-0">
            <Button
              variant="primary"
              size="sm"
              loading={confirming}
              onClick={() => onConfirm(fact.id, true)}
              className="gap-1 bg-emerald-600 hover:bg-emerald-500 text-white"
            >
              <Check className="w-3.5 h-3.5" />
              <span>Confirm</span>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              loading={confirming}
              onClick={() => onConfirm(fact.id, false)}
              className="text-slate-400 hover:text-rose-400"
            >
              <X className="w-3.5 h-3.5" />
            </Button>
          </div>
        )}
      </div>

      <p className="text-xs text-slate-300 leading-relaxed">{fact.description}</p>

      {/* Footer metrics */}
      <div className="flex items-center justify-between pt-2 border-t border-slate-800/80 text-[11px] text-slate-500 font-mono">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <Clock className="w-3.5 h-3.5 text-slate-400" />
            Tried: <strong className="text-slate-300">{fact.times_tried}</strong>
          </span>
          <span className="flex items-center gap-1">
            <ThumbsUp className="w-3.5 h-3.5 text-slate-400" />
            Helpful: <strong className="text-emerald-400">{fact.times_helpful}</strong>
          </span>
        </div>

        {fact.clinician_role && (
          <span className="text-slate-400">
            Logged by: {fact.clinician_role} {fact.clinician_id ? `(${fact.clinician_id})` : ""}
          </span>
        )}
      </div>
    </div>
  );
};
