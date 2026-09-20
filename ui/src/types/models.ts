/**
 * Project N: Model Governance and Checkpoint Evaluation Types.
 * Parity with storage/db_schema.py and docs/evaluation_protocol.md.
 */

export interface ModelCheckpoint {
  id: string;
  checkpoint_path: string;
  extractor_version: string;
  schema_version: string;
  dataset_version: string;
  validation_manifest_hash?: string | null;
  retrieval_mrr: number;
  holdout_coverage: number;
  ece_score?: number | null;
  evaluation_status: "pending" | "passed" | "failed";
  distress_cases_evaluated: number;
  zero_distress_misses: number;
  validation_timestamp?: string | null;
  caregiver_utility_score?: number | null;
  is_production: number; // 0 or 1
  promoted_at?: string | null;
  notes?: string | null;
}

export interface PromotionRequest {
  candidate_id: string;
}

export interface PromotionResponse {
  status: "promoted" | "rolled_back";
  checkpoint_id: string;
  timestamp: string;
}
