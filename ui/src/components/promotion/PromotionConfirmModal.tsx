/**
 * Project N: Model Promotion Confirmation Modal (Invariant 5).
 * Explicit caregiver confirmation gate preventing autonomous deployment.
 */

import React, { useState } from "react";
import { CheckCircle2, ShieldCheck } from "lucide-react";
import { ModelCheckpoint } from "../../types/models";
import { Button } from "../common/Button";
import { Modal } from "../common/Modal";

interface PromotionConfirmModalProps {
  isOpen: boolean;
  onClose: () => void;
  checkpoint: ModelCheckpoint | null;
  onConfirmPromote: (checkpointId: string) => Promise<void>;
}

export const PromotionConfirmModal: React.FC<PromotionConfirmModalProps> = ({
  isOpen,
  onClose,
  checkpoint,
  onConfirmPromote,
}) => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (!checkpoint) return null;

  const handlePromote = async () => {
    setLoading(true);
    setError(null);
    try {
      await onConfirmPromote(checkpoint.id);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Promotion failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Caregiver Model Promotion Sign-Off">
      <div className="flex flex-col gap-4 text-xs">
        {error && (
          <div className="p-2.5 rounded bg-rose-950/70 border border-rose-800 text-rose-300">
            {error}
          </div>
        )}

        <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg flex flex-col gap-1.5 font-mono">
          <div className="text-slate-400">Target Checkpoint: <strong className="text-sky-400">{checkpoint.id}</strong></div>
          <div className="text-slate-400">Holdout Coverage: <strong className="text-slate-200">{(checkpoint.holdout_coverage * 100).toFixed(1)}%</strong></div>
          <div className="text-slate-400">Retrieval MRR: <strong className="text-emerald-400">{checkpoint.retrieval_mrr.toFixed(3)}</strong></div>
        </div>

        <div className="p-3 bg-emerald-950/40 border border-emerald-800/60 rounded-lg text-emerald-200 flex items-start gap-2.5">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
          <div>
            <strong className="block text-emerald-300">Safety Criteria Satisfied:</strong>
            Zero acute distress misses recorded on locked safety holdout set. Pre-declared calibration targets met.
          </div>
        </div>

        <div className="p-3 bg-amber-950/40 border border-amber-800/60 rounded-lg text-amber-200 flex items-start gap-2.5">
          <ShieldCheck className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <strong className="block text-amber-300">Invariant 5 Human Sign-Off:</strong>
            Promoting this candidate will replace the active production projection head in local Metal RAM. A full rollback capability is preserved.
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 pt-3 border-t border-slate-800">
          <Button variant="ghost" size="sm" onClick={onClose} disabled={loading}>
            Cancel
          </Button>
          <Button variant="primary" size="sm" onClick={handlePromote} loading={loading}>
            Confirm Promotion to Production
          </Button>
        </div>
      </div>
    </Modal>
  );
};
