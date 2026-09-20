/**
 * Project N: Non-Diagnostic Epistemic Notice (Invariant 6 & 8).
 */

import React from "react";
import { ShieldAlert } from "lucide-react";

export const NonDiagnosticBanner: React.FC<{ className?: string }> = ({ className = "" }) => {
  return (
    <div
      className={`flex items-start gap-2.5 p-3 rounded-lg bg-sky-950/40 border border-sky-900/60 text-sky-200/90 text-xs leading-relaxed ${className}`}
    >
      <ShieldAlert className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
      <div>
        <span className="font-semibold text-sky-300">Non-Diagnostic Support Notice:</span>{" "}
        Project N is a caregiver-operated observational notebook and co-regulatory scaffolding assistant. It does not diagnose medical conditions, translate speech into definitive intent, or replace clinical assessment. Potential somatic distress always takes priority over behavioral interpretations.
      </div>
    </div>
  );
};
