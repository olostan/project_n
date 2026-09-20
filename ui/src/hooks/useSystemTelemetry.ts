/**
 * Project N: Live System Telemetry Hook.
 * Merges /api/v1/health base polling with real-time SSE telemetry updates.
 */

import { useCallback, useEffect, useState } from "react";
import { fetchSystemHealth } from "../api/telemetryApi";
import { SystemHealth, SystemTelemetryEvent, ThermalState } from "../types/telemetry";

export interface UnifiedTelemetry {
  status: "healthy" | "degraded" | "error";
  engine: string;
  activeMemoryGb: number;
  peakMemoryGb: number;
  memoryCeilingGb: number;
  thermalState: ThermalState;
  checkpointId: string;
  isTrained: boolean;
  lastUpdated: Date;
}

export function useSystemTelemetry(sseTelemetry: SystemTelemetryEvent | null) {
  const [telemetry, setTelemetry] = useState<UnifiedTelemetry>({
    status: "healthy",
    engine: "Apple MLX",
    activeMemoryGb: 0,
    peakMemoryGb: 0,
    memoryCeilingGb: 36.0,
    thermalState: "nominal",
    checkpointId: "uncalibrated_v0",
    isTrained: false,
    lastUpdated: new Date(),
  });

  const loadHealth = useCallback(async () => {
    try {
      const h: SystemHealth = await fetchSystemHealth();
      setTelemetry((prev) => ({
        ...prev,
        status: h.status,
        engine: h.engine,
        activeMemoryGb: h.active_metal_memory_gb,
        peakMemoryGb: h.peak_metal_memory_gb,
        checkpointId: h.checkpoint_id,
        isTrained: h.is_trained,
        lastUpdated: new Date(),
      }));
    } catch {
      setTelemetry((prev) => ({
        ...prev,
        status: "degraded",
      }));
    }
  }, []);

  useEffect(() => {
    loadHealth();
    const interval = setInterval(loadHealth, 20000);
    return () => clearInterval(interval);
  }, [loadHealth]);

  // Update immediately when real-time SSE telemetry arrives
  useEffect(() => {
    if (!sseTelemetry) return;
    setTelemetry((prev) => ({
      ...prev,
      activeMemoryGb: sseTelemetry.active_vram_gb,
      peakMemoryGb: sseTelemetry.peak_vram_gb,
      memoryCeilingGb: sseTelemetry.memory_ceiling_gb || 36.0,
      thermalState: sseTelemetry.thermal_state,
      lastUpdated: new Date(),
    }));
  }, [sseTelemetry]);

  return { telemetry, refetch: loadHealth };
}
