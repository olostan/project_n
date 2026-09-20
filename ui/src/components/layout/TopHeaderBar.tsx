/**
 * Project N: Top Navigation & Real-Time Telemetry Bar.
 */

import React from "react";
import { Radio } from "lucide-react";
import { UnifiedTelemetry } from "../../hooks/useSystemTelemetry";
import { ThermalIndicator } from "../telemetry/ThermalIndicator";
import { VRAMGauge } from "../telemetry/VRAMGauge";

interface TopHeaderBarProps {
  telemetry: UnifiedTelemetry;
  sseConnected: boolean;
}

export const TopHeaderBar: React.FC<TopHeaderBarProps> = ({
  telemetry,
  sseConnected,
}) => {
  return (
    <header className="h-16 border-b border-slate-800 bg-[#0f172a] px-4 sm:px-6 flex items-center justify-between shrink-0 z-30">
      {/* Brand & Child Identity */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-sky-600/20 border border-sky-500/40 flex items-center justify-center font-bold text-sky-400 text-sm">
          N
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-semibold text-slate-100 tracking-tight">
              Project N
            </h1>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-sky-950/80 text-sky-400 border border-sky-800/60">
              Child N
            </span>
          </div>
          <p className="text-[11px] text-slate-400 hidden sm:block">
            Local Co-Regulatory Assistant & Observation Notebook
          </p>
        </div>
      </div>

      {/* Telemetry Gauges */}
      <div className="flex items-center gap-3">
        {/* Metal Unified VRAM Gauge */}
        <VRAMGauge
          activeGb={telemetry.activeMemoryGb}
          peakGb={telemetry.peakMemoryGb}
          ceilingGb={telemetry.memoryCeilingGb}
        />

        {/* Thermal & Engine state */}
        <div className="hidden md:flex items-center">
          <ThermalIndicator
            thermalState={telemetry.thermalState}
            checkpointId={telemetry.checkpointId}
            isTrained={telemetry.isTrained}
          />
        </div>

        {/* SSE Connection Pulse */}
        <div
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-[11px] font-medium transition-colors ${
            sseConnected
              ? "bg-emerald-950/40 border-emerald-800/60 text-emerald-400"
              : "bg-slate-900 border-slate-800 text-slate-500"
          }`}
          title={sseConnected ? "Live SSE Stream Active" : "SSE Stream Disconnected"}
        >
          <Radio className={`w-3.5 h-3.5 ${sseConnected ? "animate-pulse" : ""}`} />
          <span className="hidden sm:inline">
            {sseConnected ? "Live Daemon" : "Connecting..."}
          </span>
        </div>
      </div>
    </header>
  );
};
