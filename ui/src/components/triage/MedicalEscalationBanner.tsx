/**
 * Project N: Medical Safety & Pediatric Escalation Banner (Invariant 7).
 * Renders high-urgency pediatric protocol and locks out behavioral interpretations.
 */

import React from "react";
import { AlertTriangle, ShieldAlert } from "lucide-react";

interface MedicalEscalationBannerProps {
  score: number;
  cutoff: number;
  instrument: string;
}

export const MedicalEscalationBanner: React.FC<MedicalEscalationBannerProps> = ({
  score,
  cutoff,
  instrument,
}) => {
  return (
    <div className="p-4 rounded-xl bg-rose-950/90 border-2 border-rose-600 shadow-2xl flex flex-col gap-3 text-rose-100 animate-pulse">
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-rose-600 text-white shrink-0">
          <AlertTriangle className="w-6 h-6" />
        </div>
        <div>
          <h3 className="text-base font-bold text-white tracking-tight">
            MEDICAL SAFETY ESCALATION REQUIRED
          </h3>
          <p className="text-xs text-rose-200">
            Caregiver pain observation score ({score}) exceeds validated clinical cut-off threshold ({cutoff}) for {instrument.toUpperCase()}.
          </p>
        </div>
      </div>

      <div className="p-3 bg-black/40 rounded-lg border border-rose-800/80 text-xs flex flex-col gap-2">
        <div className="flex items-center gap-1.5 font-semibold text-rose-300">
          <ShieldAlert className="w-4 h-4" />
          <span>Mandatory Safety Precedence Protocol:</span>
        </div>
        <ul className="list-disc list-inside space-y-1 text-slate-200 text-xs">
          <li><strong>All behavioral interpretations are locked out.</strong> Somatic distress must be evaluated first.</li>
          <li>Examine Child N for acute somatic symptoms: fever, ear pull, abdominal guarding, dental issues, or physical trauma.</li>
          <li>Perform your family pediatrician-approved physical comfort and positioning protocol.</li>
          <li>If symptoms persist, contact your pediatrician or emergency services immediately.</li>
        </ul>
      </div>

      <div className="flex items-center justify-between pt-1 text-xs">
        <span className="text-[11px] text-rose-300 font-mono">
          Pediatrician Protocol Active · Behavioral Matching Suppressed
        </span>
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-white">Emergency Call: 911 / Local ER</span>
        </div>
      </div>
    </div>
  );
};
