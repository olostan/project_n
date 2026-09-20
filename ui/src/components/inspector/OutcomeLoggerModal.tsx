/**
 * Project N: Dyadic Outcome Recording Modal.
 * Records caregiver intervention outcome and child response (Invariant 8) to trigger ChromaDB indexing.
 */

import React, { useState } from "react";
import { confirmEpisodeOutcome } from "../../api/episodesApi";
import {
  CaregiverDecision,
  ChildResponse,
  ControlledAction,
  Episode,
  OutcomeState,
  PerformanceStatus,
  ResponseChannel,
  ResponseIndependence,
} from "../../types/episodes";
import { Button } from "../common/Button";
import { Modal } from "../common/Modal";

interface OutcomeLoggerModalProps {
  isOpen: boolean;
  onClose: () => void;
  episode: Episode;
  onSuccess: () => void;
}

export const OutcomeLoggerModal: React.FC<OutcomeLoggerModalProps> = ({
  isOpen,
  onClose,
  episode,
  onSuccess,
}) => {
  const [actionPerformed, setActionPerformed] = useState<ControlledAction>(
    episode.action_performed || episode.action_offered
  );
  const [caregiverDecision, setCaregiverDecision] = useState<CaregiverDecision>(
    episode.caregiver_decision || "accepted"
  );
  const [outcomeState, setOutcomeState] = useState<OutcomeState>(
    episode.outcome_state || "settled_immediately"
  );
  const [settledWithinSec, setSettledWithinSec] = useState<number>(
    episode.settled_within_sec || 60
  );
  const [childResponse, setChildResponse] = useState<ChildResponse>(
    episode.child_response || "none"
  );
  const [responseChannel, setResponseChannel] = useState<ResponseChannel>(
    episode.response_channel || "none"
  );
  const [responseIndependence, setResponseIndependence] = useState<ResponseIndependence>(
    episode.response_independence || "independent"
  );
  const [performanceStatus, setPerformanceStatus] = useState<PerformanceStatus>(
    episode.performance_status || "completed"
  );

  const [submitting, setSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await confirmEpisodeOutcome(episode.id, {
        action_performed: actionPerformed,
        caregiver_decision: caregiverDecision,
        outcome_state: outcomeState,
        settled_within_sec: settledWithinSec,
        child_response: childResponse,
        response_channel: responseChannel,
        response_independence: responseIndependence,
        performance_status: performanceStatus,
      });
      onSuccess();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to record outcome.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Record Dyadic Outcome & Child Response" maxWidth="lg">
      <form onSubmit={handleSubmit} className="flex flex-col gap-4 text-xs">
        {error && (
          <div className="p-2.5 rounded-md bg-rose-950/70 border border-rose-800 text-rose-300">
            {error}
          </div>
        )}

        {/* Action Performed & Caregiver Decision */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <label className="block text-slate-400 font-medium mb-1">Co-Regulatory Action Performed</label>
            <select
              value={actionPerformed}
              onChange={(e) => setActionPerformed(e.target.value as ControlledAction)}
              className="w-full bg-slate-800 border border-slate-700 rounded-md p-2 text-slate-200"
            >
              <option value="deep_pressure_proprioceptive">Deep Pressure / Proprioceptive</option>
              <option value="hydration_water">Hydration (Water)</option>
              <option value="quiet_refuge">Quiet Refuge / Sensory Break</option>
              <option value="dimmed_lighting">Dimmed Lighting</option>
              <option value="vestibular_rocking">Vestibular Rocking</option>
              <option value="preferred_comfort_object">Preferred Comfort Object</option>
              <option value="aac_choice_board">AAC Choice Board</option>
              <option value="open_observation">Open Observation</option>
              <option value="other_custom">Other Custom Action</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-400 font-medium mb-1">Caregiver Decision</label>
            <select
              value={caregiverDecision}
              onChange={(e) => setCaregiverDecision(e.target.value as CaregiverDecision)}
              className="w-full bg-slate-800 border border-slate-700 rounded-md p-2 text-slate-200"
            >
              <option value="accepted">Accepted Suggestion</option>
              <option value="modified">Modified Action</option>
              <option value="rejected">Rejected / Different Path</option>
              <option value="open_observation">Open Observation Only</option>
            </select>
          </div>
        </div>

        {/* Outcome State & Settling Duration */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div>
            <label className="block text-slate-400 font-medium mb-1">Observed Outcome</label>
            <select
              value={outcomeState}
              onChange={(e) => setOutcomeState(e.target.value as OutcomeState)}
              className="w-full bg-slate-800 border border-slate-700 rounded-md p-2 text-slate-200"
            >
              <option value="settled_immediately">Settled Immediately (&lt; 2 min)</option>
              <option value="settled_delayed">Settled with Delay</option>
              <option value="no_change">No Change Observed</option>
              <option value="escalated">Escalated</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-400 font-medium mb-1">Settled Within (Seconds)</label>
            <input
              type="number"
              min={0}
              max={3600}
              value={settledWithinSec}
              onChange={(e) => setSettledWithinSec(parseInt(e.target.value) || 0)}
              className="w-full bg-slate-800 border border-slate-700 rounded-md p-2 text-slate-200"
            />
          </div>

          <div>
            <label className="block text-slate-400 font-medium mb-1">Action Status</label>
            <select
              value={performanceStatus}
              onChange={(e) => setPerformanceStatus(e.target.value as PerformanceStatus)}
              className="w-full bg-slate-800 border border-slate-700 rounded-md p-2 text-slate-200"
            >
              <option value="completed">Completed</option>
              <option value="attempted_refused">Attempted / Refused</option>
              <option value="aborted">Aborted</option>
              <option value="not_attempted">Not Attempted</option>
            </select>
          </div>
        </div>

        {/* Child Response & Agency (Invariant 8) */}
        <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg flex flex-col gap-3">
          <span className="font-semibold text-sky-400">Child Agency & Response (Ground Truth):</span>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            <div>
              <label className="block text-slate-400 mb-1">Child Response</label>
              <select
                value={childResponse}
                onChange={(e) => setChildResponse(e.target.value as ChildResponse)}
                className="w-full bg-slate-800 border border-slate-700 rounded-md p-1.5 text-slate-200 text-xs"
              >
                <option value="none">None</option>
                <option value="reach">Physical Reach</option>
                <option value="gesture">Self-Directed Gesture</option>
                <option value="vocal_signal">Vocal Signal</option>
                <option value="aac_selection">AAC Device Selection</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-400 mb-1">Channel</label>
              <select
                value={responseChannel}
                onChange={(e) => setResponseChannel(e.target.value as ResponseChannel)}
                className="w-full bg-slate-800 border border-slate-700 rounded-md p-1.5 text-slate-200 text-xs"
              >
                <option value="none">None</option>
                <option value="motor">Motor</option>
                <option value="vocal">Vocal</option>
                <option value="aac">AAC Speech Device</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-400 mb-1">Independence</label>
              <select
                value={responseIndependence}
                onChange={(e) => setResponseIndependence(e.target.value as ResponseIndependence)}
                className="w-full bg-slate-800 border border-slate-700 rounded-md p-1.5 text-slate-200 text-xs"
              >
                <option value="independent">Independent</option>
                <option value="prompted">Prompted</option>
                <option value="passive">Passive</option>
                <option value="refusal">Dissent / Refusal</option>
              </select>
            </div>
          </div>
        </div>

        {/* Buttons */}
        <div className="flex items-center justify-end gap-2 pt-2 border-t border-slate-800">
          <Button variant="ghost" size="sm" type="button" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" size="sm" type="submit" loading={submitting}>
            Save & Index to Episodic Memory
          </Button>
        </div>
      </form>
    </Modal>
  );
};
