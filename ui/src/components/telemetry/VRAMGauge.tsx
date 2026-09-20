/**
 * Project N: Metal Unified VRAM Gauge Component.
 * Visualizes active vs peak unified memory against the measured 36.0 GB invariant ceiling.
 */

import React from "react";
import { Cpu } from "lucide-react";

interface VRAMGaugeProps {
  activeGb: number;
  peakGb: number;
  ceilingGb?: number;
}

export const VRAMGauge: React.FC<VRAMGaugeProps> = ({
  activeGb,
  peakGb,
  ceilingGb = 36.0,
}) => {
  const activePct = Math.min(100, Math.max(0, (activeGb / ceilingGb) * 100));
  const peakPct = Math.min(100, Math.max(0, (peakGb / ceilingGb) * 100));

  // Invariant 2 color boundaries: green <= 28GB, amber 28-36GB, red > 36GB
  let statusColor = "bg-emerald-500";
  let textColor = "text-emerald-400";
  if (activeGb > 36.0) {
    statusColor = "bg-rose-500";
    textColor = "text-rose-400";
  } else if (activeGb > 28.0) {
    statusColor = "bg-amber-500";
    textColor = "text-amber-400";
  }

  return (
    <div className="flex items-center gap-3 bg-slate-900/90 border border-slate-800 rounded-lg px-3 py-2 text-xs">
      <div className="flex items-center gap-1.5 text-slate-400">
        <Cpu className="w-4 h-4 text-sky-400" />
        <span className="font-semibold text-slate-200">Metal VRAM:</span>
      </div>

      <div className="flex flex-col gap-1 w-32 sm:w-40">
        <div className="flex justify-between items-center text-[11px] text-slate-400">
          <span className={`font-mono font-medium ${textColor}`}>
            {activeGb.toFixed(1)} GB
          </span>
          <span className="text-slate-500 font-mono">
            / {ceilingGb.toFixed(0)} GB Cap
          </span>
        </div>

        {/* Bar container */}
        <div className="relative h-2 w-full bg-slate-800 rounded-full overflow-hidden">
          {/* Peak watermark */}
          <div
            className="absolute top-0 bottom-0 bg-sky-900/60 transition-all duration-300"
            style={{ width: `${peakPct}%` }}
            title={`Peak VRAM: ${peakGb.toFixed(1)} GB`}
          />
          {/* Active memory fill */}
          <div
            className={`absolute top-0 bottom-0 ${statusColor} transition-all duration-300 rounded-full`}
            style={{ width: `${activePct}%` }}
          />
        </div>
      </div>

      <div className="hidden lg:flex flex-col text-[10px] text-slate-500 leading-tight">
        <span>Peak: {peakGb.toFixed(1)} GB</span>
        <span>Baseline: &le;28 GB</span>
      </div>
    </div>
  );
};
