/**
 * Project N: Telemetry and System State Type Definitions.
 */

export interface SystemHealth {
  status: "healthy" | "degraded" | "error";
  engine: string;
  active_metal_memory_gb: number;
  peak_metal_memory_gb: number;
  checkpoint_id: string;
  is_trained: boolean;
}

export type ThermalState = "nominal" | "fair" | "serious" | "critical";

export interface SystemTelemetryEvent {
  active_vram_gb: number;
  peak_vram_gb: number;
  memory_ceiling_gb: number;
  thermal_state: ThermalState;
  mlx_device: string;
}
