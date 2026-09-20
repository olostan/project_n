/**
 * Project N: Caregiver Pain & NCCPC Triage Page (Invariant 7).
 */

import React from "react";
import { ArrowLeft, CheckCircle2, HeartPulse, RotateCcw } from "lucide-react";
import { useNCCPCForm } from "../hooks/useNCCPCForm";
import { ScoreIndicator } from "../components/triage/ScoreIndicator";
import { MedicalEscalationBanner } from "../components/triage/MedicalEscalationBanner";
import { NCCPCSubscaleSection } from "../components/triage/NCCPCSubscaleSection";
import { Button } from "../components/common/Button";
import { NonDiagnosticBanner } from "../components/common/NonDiagnosticBanner";

interface PainTriagePageProps {
  episodeId: string;
  onBack: () => void;
  onSubmitted?: () => void;
}

export const PainTriagePage: React.FC<PainTriagePageProps> = ({
  episodeId,
  onBack,
  onSubmitted,
}) => {
  const {
    instrument,
    setInstrument,
    scores,
    setItemScore,
    subscales,
    totalScore,
    naCount,
    subscaleScores,
    cutoffThreshold,
    cutoffBreached,
    submitting,
    submissionResult,
    error,
    submit,
    reset,
  } = useNCCPCForm(episodeId);

  const handleSubmit = async () => {
    try {
      await submit();
      if (onSubmitted) onSubmitted();
    } catch {
      // Error handled in hook
    }
  };

  return (
    <div className="flex flex-col gap-5 max-w-4xl mx-auto p-4 sm:p-6 text-slate-100">
      {/* Top action header */}
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back</span>
        </button>

        {/* Instrument Switcher */}
        <div className="flex items-center p-0.5 bg-slate-900 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => setInstrument("nccpc_pv")}
            className={`px-3 py-1 rounded-md font-medium transition-colors ${
              instrument === "nccpc_pv"
                ? "bg-sky-600 text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            NCCPC-PV (27 Items, Cut-off &ge; 11)
          </button>
          <button
            onClick={() => setInstrument("nccpc_r")}
            className={`px-3 py-1 rounded-md font-medium transition-colors ${
              instrument === "nccpc_r"
                ? "bg-sky-600 text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            NCCPC-R (30 Items, Cut-off &ge; 7)
          </button>
        </div>
      </div>

      {/* Safety Escalation Banner if Cut-off Breached */}
      {cutoffBreached && (
        <MedicalEscalationBanner
          score={totalScore}
          cutoff={cutoffThreshold}
          instrument={instrument}
        />
      )}

      {/* Running Score Indicator Card */}
      <ScoreIndicator
        totalScore={totalScore}
        cutoffThreshold={cutoffThreshold}
        cutoffBreached={cutoffBreached}
        naCount={naCount}
        subscaleScores={subscaleScores}
        instrument={instrument}
      />

      {error && (
        <div className="p-3 rounded-lg bg-rose-950/70 border border-rose-800 text-rose-300 text-xs">
          {error}
        </div>
      )}

      {submissionResult && (
        <div className="p-4 rounded-xl bg-emerald-950/60 border border-emerald-800 text-emerald-200 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>Checklist recorded successfully. Triage guidance: {submissionResult.guidance}</span>
          </div>
          <Button variant="secondary" size="sm" onClick={reset}>
            Reset Form
          </Button>
        </div>
      )}

      {/* Subscale Item Sections */}
      <div className="flex flex-col gap-4">
        {subscales.map((subscale) => (
          <NCCPCSubscaleSection
            key={subscale.id}
            subscale={subscale}
            scores={scores}
            onSelectScore={setItemScore}
          />
        ))}
      </div>

      {/* Submit Action Bar */}
      <div className="flex items-center justify-between p-4 bg-slate-900 border border-slate-800 rounded-xl">
        <Button variant="ghost" size="sm" onClick={reset} className="gap-1.5 text-slate-400">
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Reset Scores</span>
        </Button>

        <Button
          variant={cutoffBreached ? "danger" : "primary"}
          size="md"
          onClick={handleSubmit}
          loading={submitting}
          className="gap-2"
        >
          <HeartPulse className="w-4 h-4" />
          <span>Save Checklist & Clinical Score</span>
        </Button>
      </div>

      <NonDiagnosticBanner />
    </div>
  );
};
