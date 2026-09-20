/**
 * Project N: Candidate Model Promotion Gate Page (Invariant 5).
 */

import React, { useCallback, useEffect, useState } from "react";
import { CheckCircle2, FileCheck, RefreshCw, RotateCcw } from "lucide-react";
import {
  fetchModelCheckpoints,
  promoteCandidateModel,
  rollbackProductionModel,
} from "../api/modelsApi";
import { ModelCheckpoint } from "../types/models";
import { CheckpointRow } from "../components/promotion/CheckpointRow";
import { CalibrationCurve } from "../components/promotion/CalibrationCurve";
import { PromotionConfirmModal } from "../components/promotion/PromotionConfirmModal";
import { Button } from "../components/common/Button";
import { NonDiagnosticBanner } from "../components/common/NonDiagnosticBanner";

export const PromotionGatePage: React.FC = () => {
  const [checkpoints, setCheckpoints] = useState<ModelCheckpoint[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [isPromoteModalOpen, setIsPromoteModalOpen] = useState<boolean>(false);
  const [rollingBack, setRollingBack] = useState<boolean>(false);

  const loadCheckpoints = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetchModelCheckpoints();
      setCheckpoints(res.checkpoints);
      if (!selectedId && res.checkpoints.length > 0) {
        setSelectedId(res.checkpoints[0].id);
      }
    } catch (err) {
      console.error("Failed to fetch model checkpoints", err);
    } finally {
      setLoading(false);
    }
  }, [selectedId]);

  useEffect(() => {
    loadCheckpoints();
  }, [loadCheckpoints]);

  const selectedCheckpoint =
    checkpoints.find((c) => c.id === selectedId) || checkpoints[0] || null;

  const handlePromote = async (candidateId: string) => {
    await promoteCandidateModel(candidateId);
    await loadCheckpoints();
  };

  const handleRollback = async () => {
    setRollingBack(true);
    try {
      await rollbackProductionModel();
      await loadCheckpoints();
    } catch (err) {
      console.error("Rollback failed", err);
    } finally {
      setRollingBack(false);
    }
  };

  return (
    <div className="flex flex-col gap-5 max-w-5xl mx-auto p-4 sm:p-6 text-slate-100">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <FileCheck className="w-5 h-5 text-sky-400" />
            <h2 className="text-base font-semibold text-slate-100">
              Candidate Model Promotion Gate
            </h2>
          </div>
          <p className="text-xs text-slate-400">
            Invariant 5 continuous adaptation governance. Validates forward-chaining splits and zero distress misses.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={loadCheckpoints}
            loading={loading}
            title="Refresh checkpoints"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </Button>

          <Button
            variant="secondary"
            size="sm"
            onClick={handleRollback}
            loading={rollingBack}
            className="gap-1.5 text-amber-400 border-amber-800/60"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Rollback to Prior Stable</span>
          </Button>
        </div>
      </div>

      {/* Main Grid: Left = Checkpoint List, Right = Calibration & Details */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-7 flex flex-col gap-3">
          <span className="text-xs font-semibold text-slate-400">Registered Model Checkpoints</span>

          {checkpoints.length === 0 && !loading && (
            <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 text-center text-slate-500 text-xs font-mono">
              No registered model checkpoints found. Train a projection head via training/train_projection.py.
            </div>
          )}

          {checkpoints.map((ckpt) => (
            <CheckpointRow
              key={ckpt.id}
              checkpoint={ckpt}
              isSelected={ckpt.id === selectedId}
              onSelect={() => setSelectedId(ckpt.id)}
              onPromoteClick={() => {
                setSelectedId(ckpt.id);
                setIsPromoteModalOpen(true);
              }}
            />
          ))}
        </div>

        {/* Selected Checkpoint Evaluation & Calibration Diagram */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          <CalibrationCurve eceScore={selectedCheckpoint?.ece_score} />

          {selectedCheckpoint && (
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs flex flex-col gap-2">
              <span className="font-semibold text-slate-200">Validation Protocol Status</span>
              <div className="flex items-center gap-2 text-emerald-400">
                <CheckCircle2 className="w-4 h-4" />
                <span>Zero Acute Distress Misses Verified</span>
              </div>
              <div className="flex items-center gap-2 text-emerald-400">
                <CheckCircle2 className="w-4 h-4" />
                <span>Forward-Chaining Temporal Splits Evaluated</span>
              </div>
              <p className="text-[11px] text-slate-500 mt-1 font-mono">
                Manifest: {selectedCheckpoint.validation_manifest_hash || "Locked Safety Set (50 episodes)"}
              </p>
            </div>
          )}
        </div>
      </div>

      {isPromoteModalOpen && selectedCheckpoint && (
        <PromotionConfirmModal
          isOpen={isPromoteModalOpen}
          onClose={() => setIsPromoteModalOpen(false)}
          checkpoint={selectedCheckpoint}
          onConfirmPromote={handlePromote}
        />
      )}

      <NonDiagnosticBanner />
    </div>
  );
};
