/**
 * Project N: Thermal State & MLX Engine Indicator Component.
 */

import React from "react";
import { Activity, Flame, ShieldCheck } from "lucide-react";
import { ThermalState } from "../../types/telemetry";

interface ThermalIndicatorProps {
  thermalState: ThermalState;
  checkpointId: string;
  isTrained: boolean;
}

const thermalConfig: Record<ThermalState, { label: string; color: string; badge: string }> = {
  nominal: { label: "Nominal", color: "text-emerald-400", badge: "bg-emerald-950/60 border-emerald-800/80 text-emerald-300" },
  fair: { label: "Fair", color: "text-yellow-400", badge: "bg-yellow-950/60 border-yellow-800/80 text-yellow-300" },
  serious: { label: "Warm", color: "text-amber-400", badge: "bg-amber-950/60 border-amber-800/80 text-amber-300" },
  critical: { label: "Throttled", color: "text-rose-400", badge: "bg-rose-950/60 border-rose-800/80 text-rose-300" },
};

export const ThermalIndicator: React.FC<ThermalIndicatorProps> = ({
  thermalState,
  checkpointId,
  isTrained,
}) => {
  const t = thermalConfig[thermalState] || thermalConfig.nominal;

  return (
    <div className="flex items-center gap-2 text-xs">
      {/* Thermal State */}
      <div
        className={`flex items-center gap-1 px-2.5 py-1 rounded-md border text-[11px] font-medium ${t.badge}`}
        title={`Apple Silicon Thermal State: ${t.label}`}
      >
        <Flame className={`w-3.5 h-3.5 ${t.color}`} />
        <span>{t.label}</span>
      </div>

      {/* Checkpoint ID & Trained status */}
      <div
        className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-300 text-[11px]"
        title={isTrained ? "Metric Head Calibrated" : "Uncalibrated Baseline"}
      >
        {isTrained ? (
          <ShieldCheck className="w-3.5 h-3.5 text-sky-400" />
        ) : (
          <Activity className="w-3.5 h-3.5 text-amber-400" />
        )}
        <span className="font-mono">{checkpointId}</span>
      </div>
    </div>
  );
};
