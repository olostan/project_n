/**
 * Project N: Model Checkpoint Row Component.
 */

import React from "react";
import { Sparkles } from "lucide-react";
import { ModelCheckpoint } from "../../types/models";
import { Badge } from "../common/Badge";
import { Button } from "../common/Button";

interface CheckpointRowProps {
  checkpoint: ModelCheckpoint;
  isSelected: boolean;
  onSelect: () => void;
  onPromoteClick: () => void;
}

export const CheckpointRow: React.FC<CheckpointRowProps> = ({
  checkpoint,
  isSelected,
  onSelect,
  onPromoteClick,
}) => {
  const isProd = checkpoint.is_production === 1;
  const isPassed = checkpoint.evaluation_status === "passed";

  return (
    <div
      onClick={onSelect}
      className={`p-4 rounded-xl border transition-all cursor-pointer flex flex-col gap-3 ${
        isSelected
          ? "bg-slate-800/90 border-sky-500 shadow-md ring-1 ring-sky-500/40"
          : "bg-slate-900/70 border-slate-800 hover:border-slate-700"
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-mono font-bold text-slate-100 text-sm">{checkpoint.id}</span>
          {isProd && (
            <Badge variant="success" size="sm" className="gap-1">
              <Sparkles className="w-3 h-3" />
              Active Production
            </Badge>
          )}
          <Badge
            variant={isPassed ? "info" : checkpoint.evaluation_status === "failed" ? "danger" : "warning"}
            size="sm"
          >
            {checkpoint.evaluation_status.toUpperCase()}
          </Badge>
        </div>

        {!isProd && isPassed && (
          <div onClick={(e) => e.stopPropagation()}>
            <Button variant="primary" size="sm" onClick={onPromoteClick}>
              Promote to Active
            </Button>
          </div>
        )}
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-slate-800/80 text-xs font-mono">
        <div className="p-2 rounded bg-slate-800/60 border border-slate-700/60">
          <span className="text-slate-400 block text-[10px]">Retrieval MRR</span>
          <strong className={checkpoint.retrieval_mrr >= 0.65 ? "text-emerald-400" : "text-rose-400"}>
            {checkpoint.retrieval_mrr.toFixed(3)}
          </strong>
        </div>

        <div className="p-2 rounded bg-slate-800/60 border border-slate-700/60">
          <span className="text-slate-400 block text-[10px]">Holdout Coverage</span>
          <strong className="text-slate-200">
            {(checkpoint.holdout_coverage * 100).toFixed(1)}%
          </strong>
        </div>

        <div className="p-2 rounded bg-slate-800/60 border border-slate-700/60">
          <span className="text-slate-400 block text-[10px]">Expected Calib. Error</span>
          <strong className={checkpoint.ece_score && checkpoint.ece_score <= 0.12 ? "text-emerald-400" : "text-amber-400"}>
            {checkpoint.ece_score ? checkpoint.ece_score.toFixed(3) : "N/A"}
          </strong>
        </div>

        <div className="p-2 rounded bg-slate-800/60 border border-slate-700/60">
          <span className="text-slate-400 block text-[10px]">Distress Misses</span>
          <strong className={checkpoint.zero_distress_misses === 0 ? "text-emerald-400" : "text-rose-400"}>
            {checkpoint.zero_distress_misses}
          </strong>
        </div>
      </div>
    </div>
  );
};
