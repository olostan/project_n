/**
 * Project N: Resilient Server-Sent Events (SSE) Hook.
 * Connects to /api/v1/events/stream with reconnect logic and typed event dispatching.
 */

import { useEffect, useRef, useState } from "react";
import {
  FourLayerCardData,
  ProcessingProgressEvent,
  SignalQualityAbstainedEvent,
  SystemTelemetryEvent,
} from "../types/events";

export interface SSEStreamState {
  connected: boolean;
  activeTasks: Record<string, ProcessingProgressEvent>;
  latestTelemetry: SystemTelemetryEvent | null;
  latestFourLayerCard: FourLayerCardData | null;
  latestAbstention: SignalQualityAbstainedEvent | null;
  eventLog: Array<{ id: string; timestamp: string; type: string; summary: string }>;
}

export function useSSEStream() {
  const [state, setState] = useState<SSEStreamState>({
    connected: false,
    activeTasks: {},
    latestTelemetry: null,
    latestFourLayerCard: null,
    latestAbstention: null,
    eventLog: [],
  });

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    let isMounted = true;

    function connect() {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      const es = new EventSource("/api/v1/events/stream");
      eventSourceRef.current = es;

      es.onopen = () => {
        if (!isMounted) return;
        setState((prev) => ({ ...prev, connected: true }));
      };

      es.onerror = () => {
        if (!isMounted) return;
        setState((prev) => ({ ...prev, connected: false }));
        es.close();
        // Exponential or 3s retry
        reconnectTimeoutRef.current = window.setTimeout(connect, 3000);
      };

      // 1. processing_progress
      es.addEventListener("processing_progress", (e: MessageEvent) => {
        if (!isMounted) return;
        try {
          const data = JSON.parse(e.data) as ProcessingProgressEvent;
          setState((prev) => {
            const nextTasks = { ...prev.activeTasks, [data.task_id]: data };
            if (data.stage === "completed" || data.stage === "analysis_error") {
              // Mark complete, keep in view
            }
            const logEntry = {
              id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
              timestamp: new Date().toLocaleTimeString(),
              type: "progress",
              summary: `[${data.stage}] ${Math.round(data.progress * 100)}% (Task ${data.task_id.slice(0, 8)})`,
            };
            return {
              ...prev,
              activeTasks: nextTasks,
              eventLog: [logEntry, ...prev.eventLog].slice(0, 50),
            };
          });
        } catch {
          // Ignore JSON parse error
        }
      });

      // 2. system_telemetry
      es.addEventListener("system_telemetry", (e: MessageEvent) => {
        if (!isMounted) return;
        try {
          const data = JSON.parse(e.data) as SystemTelemetryEvent;
          setState((prev) => ({
            ...prev,
            latestTelemetry: data,
          }));
        } catch {
          // Ignore
        }
      });

      // 3. four_layer_card
      es.addEventListener("four_layer_card", (e: MessageEvent) => {
        if (!isMounted) return;
        try {
          const data = JSON.parse(e.data) as FourLayerCardData;
          const logEntry = {
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            timestamp: new Date().toLocaleTimeString(),
            type: "card",
            summary: `Four-Layer Card: ${data.abstained ? "ABSTAINED" : "Emitted for task " + data.task_id.slice(0, 8)}`,
          };
          setState((prev) => ({
            ...prev,
            latestFourLayerCard: data,
            eventLog: [logEntry, ...prev.eventLog].slice(0, 50),
          }));
        } catch {
          // Ignore
        }
      });

      // 4. signal_quality_abstained
      es.addEventListener("signal_quality_abstained", (e: MessageEvent) => {
        if (!isMounted) return;
        try {
          const data = JSON.parse(e.data) as SignalQualityAbstainedEvent;
          const logEntry = {
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            timestamp: new Date().toLocaleTimeString(),
            type: "abstention",
            summary: `Signal Quality Breach: ${data.breaches.join("; ")}`,
          };
          setState((prev) => ({
            ...prev,
            latestAbstention: data,
            eventLog: [logEntry, ...prev.eventLog].slice(0, 50),
          }));
        } catch {
          // Ignore
        }
      });
    }

    connect();

    return () => {
      isMounted = false;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (eventSourceRef.current) eventSourceRef.current.close();
    };
  }, []);

  return state;
}
