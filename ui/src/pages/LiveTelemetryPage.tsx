/**
 * Project N: Live Telemetry & Ingestion Pipeline Page.
 */

import React from "react";
import { Activity, Clock, Cpu, Radio } from "lucide-react";
import { SSEStreamState } from "../hooks/useSSEStream";
import { UnifiedTelemetry } from "../hooks/useSystemTelemetry";
import { PipelineTaskTracker } from "../components/telemetry/PipelineTaskTracker";
import { NonDiagnosticBanner } from "../components/common/NonDiagnosticBanner";

interface LiveTelemetryPageProps {
  telemetry: UnifiedTelemetry;
  sseState: SSEStreamState;
}

export const LiveTelemetryPage: React.FC<LiveTelemetryPageProps> = ({
  telemetry,
  sseState,
}) => {
  return (
    <div className="flex flex-col gap-6 max-w-5xl mx-auto p-4 sm:p-6 text-slate-100">
      {/* Top Welcome & System Status Card */}
      <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-lg flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-sky-600/20 border border-sky-500/30 text-sky-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-slate-100">Apple Silicon MLX Daemon</h2>
              <p className="text-xs text-slate-400 font-mono">
                Engine: {telemetry.engine} • Active Checkpoint: {telemetry.checkpointId}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-950/60 border border-emerald-800 text-emerald-300 text-xs font-medium">
              <Radio className="w-3.5 h-3.5 animate-pulse" />
              <span>{sseState.connected ? "Stream Active" : "Disconnected"}</span>
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-3 border-t border-slate-800 text-xs font-mono">
          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <span className="text-slate-500 block text-[10px]">ACTIVE METAL VRAM</span>
            <strong className="text-emerald-400 text-base">{telemetry.activeMemoryGb.toFixed(2)} GB</strong>
            <span className="text-slate-500 block text-[10px] mt-0.5">&le; 28.0 GB Operational Base</span>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <span className="text-slate-500 block text-[10px]">PEAK METAL ALLOCATION</span>
            <strong className="text-sky-400 text-base">{telemetry.peakMemoryGb.toFixed(2)} GB</strong>
            <span className="text-slate-500 block text-[10px] mt-0.5">36.0 GB Invariant Ceiling</span>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <span className="text-slate-500 block text-[10px]">METRIC HEAD STATUS</span>
            <strong className={telemetry.isTrained ? "text-emerald-400 text-base" : "text-amber-400 text-base"}>
              {telemetry.isTrained ? "Calibrated (128d)" : "Uncalibrated Baseline"}
            </strong>
            <span className="text-slate-500 block text-[10px] mt-0.5">Fail-closed gating enabled</span>
          </div>
        </div>
      </div>

      {/* Active Pipeline Ingestion Tasks */}
      <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-lg flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-sky-400" />
            <h3 className="text-sm font-semibold text-slate-100">Live Ingestion & Processing Queue</h3>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            {Object.keys(sseState.activeTasks).length} active
          </span>
        </div>

        <PipelineTaskTracker tasks={sseState.activeTasks} />
      </div>

      {/* Real-time SSE Event Log */}
      <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-lg flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-slate-400" />
            <h3 className="text-sm font-semibold text-slate-100">Real-Time Event Stream Log</h3>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">Buffered SSE Events</span>
        </div>

        {sseState.eventLog.length === 0 ? (
          <div className="p-4 rounded-lg bg-slate-950/50 border border-slate-800/60 text-center text-slate-500 text-xs font-mono">
            Awaiting streaming events from /api/v1/events/stream...
          </div>
        ) : (
          <div className="flex flex-col gap-1.5 max-h-56 overflow-y-auto font-mono text-xs divide-y divide-slate-800/60">
            {sseState.eventLog.map((ev) => (
              <div key={ev.id} className="pt-1.5 pb-1 flex items-baseline gap-2.5">
                <span className="text-slate-500 text-[10px] shrink-0">{ev.timestamp}</span>
                <span className="text-sky-400 text-[10px] uppercase font-bold shrink-0">[{ev.type}]</span>
                <span className="text-slate-300 text-[11px] truncate">{ev.summary}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <NonDiagnosticBanner />
    </div>
  );
};
