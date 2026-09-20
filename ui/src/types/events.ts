export type { FourLayerCardData } from "./episodes";
export type { SystemTelemetryEvent } from "./telemetry";
import { FourLayerCardData } from "./episodes";
import { SystemTelemetryEvent } from "./telemetry";

export interface ProcessingProgressEvent {
  task_id: string;
  stage:
    | "started"
    | "vault_stored"
    | "demuxing"
    | "kinematic_flow_extraction"
    | "audio_spectral_extraction"
    | "temporal_pooling"
    | "metric_projection"
    | "retrieval_matching"
    | "completed"
    | "demux_failed"
    | "analysis_error";
  progress: number;
  clip_id?: string;
  windows_processed?: number;
  total_windows?: number;
  timestamp?: string;
}

export interface SignalQualityAbstainedEvent {
  task_id: string;
  clip_id?: string;
  breaches: string[];
  timestamp: string;
}

export interface ProcessingFailedEvent {
  task_id: string;
  clip_id?: string;
  stage: string;
  error: string;
}

export type ProjectNEvent =
  | { type: "processing_progress"; data: ProcessingProgressEvent }
  | { type: "system_telemetry"; data: SystemTelemetryEvent }
  | { type: "four_layer_card"; data: FourLayerCardData }
  | { type: "signal_quality_abstained"; data: SignalQualityAbstainedEvent }
  | { type: "processing_failed"; data: ProcessingFailedEvent };
