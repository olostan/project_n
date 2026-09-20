/**
 * Project N: Stage Personal Fact or Clinical Technique Modal.
 */

import React, { useState } from "react";
import { createFact } from "../../api/factsApi";
import { FactCategory, FactSourceType } from "../../types/facts";
import { Button } from "../common/Button";
import { Modal } from "../common/Modal";

interface CreateFactModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreated: () => void;
}

export const CreateFactModal: React.FC<CreateFactModalProps> = ({
  isOpen,
  onClose,
  onCreated,
}) => {
  const [category, setCategory] = useState<FactCategory>("therapist_technique");
  const [sourceType, setSourceType] = useState<FactSourceType>("ot_session");
  const [factTitle, setFactTitle] = useState<string>("");
  const [description, setDescription] = useState<string>("");
  const [clinicianRole, setClinicianRole] = useState<string>("Occupational Therapist");
  const [clinicianId, setClinicianId] = useState<string>("");

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!factTitle.trim() || !description.trim()) {
      setError("Title and description are required.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      await createFact({
        category,
        source_type: sourceType,
        fact_title: factTitle,
        description,
        clinician_role: clinicianRole || undefined,
        clinician_id: clinicianId || undefined,
      });
      onCreated();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to stage fact.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Stage Clinician Technique or Personal Fact">
      <form onSubmit={handleSubmit} className="flex flex-col gap-3 text-xs">
        {error && (
          <div className="p-2 rounded bg-rose-950/70 border border-rose-800 text-rose-300">
            {error}
          </div>
        )}

        <div>
          <label className="block text-slate-400 font-medium mb-1">Title</label>
          <input
            type="text"
            placeholder="e.g. Firm Hand Pressure Before Transition"
            value={factTitle}
            onChange={(e) => setFactTitle(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-md p-2 text-slate-200"
            required
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-slate-400 font-medium mb-1">Category</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value as FactCategory)}
              className="w-full bg-slate-800 border border-slate-700 rounded-md p-2 text-slate-200"
            >
              <option value="therapist_technique">Therapist Technique</option>
              <option value="calming_cue">Calming Cue</option>
              <option value="comfort_object">Comfort Object</option>
              <option value="sensory_trigger">Sensory Trigger</option>
              <option value="communication_routine">Communication Routine</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-400 font-medium mb-1">Source Context</label>
            <select
              value={sourceType}
              onChange={(e) => setSourceType(e.target.value as FactSourceType)}
              className="w-full bg-slate-800 border border-slate-700 rounded-md p-2 text-slate-200"
            >
              <option value="ot_session">OT Session</option>
              <option value="slp_session">SLP Session</option>
              <option value="home_observation">Home Observation</option>
              <option value="school">School</option>
            </select>
          </div>
        </div>

        <div>
          <label className="block text-slate-400 font-medium mb-1">Detailed Description & Guidance</label>
          <textarea
            rows={3}
            placeholder="Describe the sensory or behavioral cue and how to apply it..."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-md p-2 text-slate-200"
            required
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-slate-400 mb-1">Clinician Role (Optional)</label>
            <input
              type="text"
              placeholder="e.g. Lead OT"
              value={clinicianRole}
              onChange={(e) => setClinicianRole(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-md p-2 text-slate-200"
            />
          </div>

          <div>
            <label className="block text-slate-400 mb-1">Clinician ID / Name</label>
            <input
              type="text"
              placeholder="e.g. Dr. Sarah"
              value={clinicianId}
              onChange={(e) => setClinicianId(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-md p-2 text-slate-200"
            />
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 pt-3 border-t border-slate-800 mt-2">
          <Button variant="ghost" size="sm" type="button" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" size="sm" type="submit" loading={loading}>
            Stage for Review
          </Button>
        </div>
      </form>
    </Modal>
  );
};
