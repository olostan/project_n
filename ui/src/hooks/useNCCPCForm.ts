/**
 * Project N: NCCPC Pain Checklist Form Hook.
 * Implements real-time scoring, subscale computation, and safety cut-off alerts.
 */

import { useCallback, useMemo, useState } from "react";
import { submitNCCPCChecklist } from "../api/nccpcApi";
import {
  NCCPC_PV_SUBSCALES,
  NCCPC_R_EXTRA_SUBSCALE,
  NCCPCResponseValue,
  NCCPCScoreResponse,
  NCCPCSubscale,
} from "../types/nccpc";

export function useNCCPCForm(episodeId: string) {
  const [instrument, setInstrument] = useState<"nccpc_pv" | "nccpc_r">("nccpc_pv");
  const [scores, setScores] = useState<Record<string, NCCPCResponseValue>>({});
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [submissionResult, setSubmissionResult] = useState<NCCPCScoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const subscales: NCCPCSubscale[] = useMemo(() => {
    return instrument === "nccpc_pv"
      ? NCCPC_PV_SUBSCALES
      : [...NCCPC_PV_SUBSCALES, NCCPC_R_EXTRA_SUBSCALE];
  }, [instrument]);

  const allItems = useMemo(() => {
    return subscales.flatMap((s) => s.items);
  }, [subscales]);

  const cutoffThreshold = instrument === "nccpc_pv" ? 11 : 7;

  // Set single item score
  const setItemScore = useCallback((itemId: string, value: NCCPCResponseValue) => {
    setScores((prev) => ({ ...prev, [itemId]: value }));
  }, []);

  // Compute live subscale scores and total
  const { totalScore, naCount, subscaleScores, cutoffBreached } = useMemo(() => {
    let total = 0;
    let na = 0;
    const subTotals: Record<string, number> = {};

    for (const subscale of subscales) {
      let subSum = 0;
      for (const item of subscale.items) {
        const val = scores[item.id];
        if (typeof val === "number") {
          subSum += val;
        } else if (val === "NA") {
          na += 1;
        }
      }
      subTotals[subscale.id] = subSum;
      total += subSum;
    }

    return {
      totalScore: total,
      naCount: na,
      subscaleScores: subTotals,
      cutoffBreached: total >= cutoffThreshold,
    };
  }, [scores, subscales, cutoffThreshold]);

  const submit = useCallback(async () => {
    setSubmitting(true);
    setError(null);
    try {
      const res = await submitNCCPCChecklist(episodeId, {
        instrument,
        scores,
      });
      setSubmissionResult(res);
      return res;
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Submission failed.";
      setError(msg);
      throw err;
    } finally {
      setSubmitting(false);
    }
  }, [episodeId, instrument, scores]);

  const reset = useCallback(() => {
    setScores({});
    setSubmissionResult(null);
    setError(null);
  }, []);

  return {
    instrument,
    setInstrument,
    scores,
    setItemScore,
    subscales,
    allItems,
    totalScore,
    naCount,
    subscaleScores,
    cutoffThreshold,
    cutoffBreached,
    submitting,
    submissionResult,
    error,
    submit,
    reset,
  };
}
