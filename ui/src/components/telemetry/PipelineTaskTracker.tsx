/**
 * Project N: Live Pipeline Processing Task Tracker.
 * Displays real-time progress across extraction stages from SSE events.
 */

import React from "react";
import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { ProcessingProgressEvent } from "../../types/events";

interface PipelineTaskTrackerProps {
  tasks: Record<string, ProcessingProgressEvent>;
}

const stageLabels: Record<string, string> = {
  started: "Ingest Started",
  vault_stored: "Encrypted to Vault",
  demuxing: "Demuxing Video & Audio",
  kinematic_flow_extraction: "MediaPipe & Optical Flow",
  audio_spectral_extraction: "STFT, CQT & F0 Pitch",
  temporal_pooling: "Attention Pooling",
  metric_projection: "128-Dim Metric Space",
  retrieval_matching: "ChromaDB Episodic Retrieval",
  completed: "Analysis Completed",
  demux_failed: "Media Decode Failed",
  analysis_error: "Analysis Error",
};

export const PipelineTaskTracker: React.FC<PipelineTaskTrackerProps> = ({ tasks }) => {
  const taskList = Object.values(tasks);

  if (taskList.length === 0) {
    return (
      <div className="p-4 rounded-lg bg-slate-900/60 border border-slate-800 text-center text-slate-500 text-xs">
        No active background extraction tasks. Ingested clips will appear here in real time.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2.5">
      {taskList.map((task) => {
        const isComplete = task.stage === "completed";
        const isFailed = task.stage === "demux_failed" || task.stage === "analysis_error";
        const pct = Math.round(task.progress * 100);

        return (
          <div
            key={task.task_id}
            className="p-3 bg-slate-900/80 border border-slate-800 rounded-lg flex flex-col gap-2"
          >
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                {isComplete ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : isFailed ? (
                  <XCircle className="w-4 h-4 text-rose-400" />
                ) : (
                  <Loader2 className="w-4 h-4 text-sky-400 animate-spin" />
                )}
                <span className="font-medium text-slate-200">
                  {stageLabels[task.stage] || task.stage}
                </span>
              </div>
              <span className="font-mono text-slate-400 text-[11px]">{pct}%</span>
            </div>

            {/* Progress bar */}
            <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-300 rounded-full ${
                  isFailed
                    ? "bg-rose-500"
                    : isComplete
                    ? "bg-emerald-500"
                    : "bg-sky-500"
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>

            <div className="flex justify-between items-center text-[10px] text-slate-500 font-mono">
              <span>Task: {task.task_id.slice(0, 8)}...</span>
              {task.windows_processed !== undefined && (
                <span>
                  Windows: {task.windows_processed} / {task.total_windows || "?"}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};
