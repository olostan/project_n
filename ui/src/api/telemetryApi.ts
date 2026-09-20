/**
 * Project N: Telemetry API Service.
 */

import { request } from "./client";
import { SystemHealth } from "../types/telemetry";

export async function fetchSystemHealth(): Promise<SystemHealth> {
  return request<SystemHealth>("/api/v1/health");
}
