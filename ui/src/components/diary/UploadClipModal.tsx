/**
 * Project N: Upload Video Clip Modal.
 * Supports drag-and-drop / file selection, chunked vault upload, and pipeline triggering.
 */

import React, { useState, useRef } from "react";
import { Upload, Film, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import { Modal } from "../common/Modal";
import { Button } from "../common/Button";
import { uploadAndAnalyzeClip, UploadProgress } from "../../api/clipsApi";

interface UploadClipModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess: (clipId: string, taskId: string) => void;
}

const ANTECEDENTS = [
  { id: "mealtime", label: "Mealtime / Dining" },
  { id: "post_school_transition", label: "Post-School Transition" },
  { id: "sensory_seeking", label: "Sensory Seeking / Movement" },
  { id: "fatigue_evening", label: "Evening Fatigue / Bedtime" },
  { id: "loud_environment", label: "Loud / High-Stimulus Environment" },
  { id: "motor_restlessness", label: "Motor Restlessness" },
  { id: "unknown", label: "Unknown / Unspecified" },
];

export const UploadClipModal: React.FC<UploadClipModalProps> = ({
  isOpen,
  onClose,
  onUploadSuccess,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [antecedentId, setAntecedentId] = useState("mealtime");
  const [setting, setSetting] = useState("living_room");
  const [notes, setNotes] = useState("");
  const [hypothesis, setHypothesis] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [successInfo, setSuccessInfo] = useState<{ clipId: string; taskId: string } | null>(null);

  const resetState = () => {
    setSelectedFile(null);
    setNotes("");
    setHypothesis("");
    setProgress(null);
    setError(null);
    setIsUploading(false);
    setSuccessInfo(null);
  };

  const handleClose = () => {
    if (isUploading) return;
    resetState();
    onClose();
  };

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("video/") || file.name.endsWith(".mp4") || file.name.endsWith(".mov")) {
        setSelectedFile(file);
        setError(null);
      } else {
        setError("Please select a valid video file (.mp4 or .mov).");
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleStartUpload = async () => {
    if (!selectedFile) {
      setError("Please select a video file first.");
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const result = await uploadAndAnalyzeClip(
        selectedFile,
        {
          antecedent_id: antecedentId,
          antecedent_notes: notes || undefined,
          caregiver_hypothesis: hypothesis || undefined,
          setting: setting || undefined,
          observer: "caregiver",
          captured_at: new Date().toISOString(),
        },
        (p) => setProgress(p)
      );

      setSuccessInfo(result);
      onUploadSuccess(result.clipId, result.taskId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload video clip.");
      setIsUploading(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Upload Video Clip for Ingestion" maxWidth="lg">
      <div className="space-y-4 p-5">
        {successInfo ? (
          <div className="text-center py-6 space-y-4">
            <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto" />
            <h4 className="text-lg font-semibold text-slate-100">Upload & Vaulting Complete!</h4>
            <p className="text-xs text-slate-400 max-w-sm mx-auto">
              Clip <code className="text-slate-200">{successInfo.clipId.slice(0, 8)}</code> is encrypted in the vault and background multimodal analysis is running on Apple Silicon Metal.
            </p>
            <div className="pt-3 flex justify-center gap-3">
              <Button variant="primary" onClick={handleClose}>Back to Diary</Button>
            </div>
          </div>
        ) : (
          <>
            {/* File Dropzone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleFileDrop}
              onClick={() => !isUploading && fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
                isDragging ? "border-sky-500 bg-sky-950/20" : "border-slate-700 hover:border-slate-500 bg-slate-900/50"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4,video/quicktime"
                className="hidden"
                onChange={handleFileChange}
                disabled={isUploading}
              />
              {selectedFile ? (
                <div className="flex items-center justify-center gap-3 text-slate-200">
                  <Film className="w-8 h-8 text-sky-400" />
                  <div className="text-left">
                    <p className="text-sm font-medium">{selectedFile.name}</p>
                    <p className="text-xs text-slate-400">{(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</p>
                  </div>
                </div>
              ) : (
                <div className="space-y-2">
                  <Upload className="w-8 h-8 text-slate-400 mx-auto" />
                  <p className="text-sm text-slate-300 font-medium">Drag & drop video clip here, or click to browse</p>
                  <p className="text-xs text-slate-500">Supports .mp4 and .mov (30–120s clips recommended)</p>
                </div>
              )}
            </div>

            {/* Context Inputs */}
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <label className="block text-slate-400 mb-1 font-medium">Antecedent / Context</label>
                <select
                  value={antecedentId}
                  onChange={(e) => setAntecedentId(e.target.value)}
                  disabled={isUploading}
                  aria-label="Antecedent Context"
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200"
                >
                  {ANTECEDENTS.map((a) => (
                    <option key={a.id} value={a.id}>{a.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-slate-400 mb-1 font-medium">Physical Setting</label>
                <input
                  type="text"
                  value={setting}
                  onChange={(e) => setSetting(e.target.value)}
                  disabled={isUploading}
                  aria-label="Physical Setting"
                  placeholder="e.g. living_room, classroom"
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200"
                />
              </div>
            </div>

            <div className="text-xs space-y-1">
              <label className="block text-slate-400 font-medium">Caregiver Notes / Antecedent Trigger</label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                disabled={isUploading}
                rows={2}
                placeholder="Observed environmental events, sound levels, hunger, transitions..."
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-200"
              />
            </div>

            {/* Progress Feedback */}
            {progress && (
              <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-300 capitalize flex items-center gap-1.5 font-medium">
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-sky-400" />
                    {progress.stage === "hashing" && "Computing SHA-256 Checksum..."}
                    {progress.stage === "uploading" && `Uploading Chunks (${progress.currentChunk} / ${progress.totalChunks})...`}
                    {progress.stage === "finalizing" && "Finalizing & AES-256 Vault Encryption..."}
                    {progress.stage === "analyzing" && "Starting MLX Extraction Pipeline..."}
                    {progress.stage === "completed" && "Ingestion Complete!"}
                  </span>
                  <span className="text-sky-400 font-mono font-medium">{progress.percent}%</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div
                    className="bg-sky-500 h-full transition-all duration-300 rounded-full"
                    style={{ width: `${progress.percent}%` }}
                  />
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center gap-2 p-3 bg-rose-950/40 border border-rose-800/60 rounded-lg text-xs text-rose-300">
                <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
                <span>{error}</span>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex justify-end gap-2 pt-2 border-t border-slate-800">
              <Button variant="secondary" onClick={handleClose} disabled={isUploading}>Cancel</Button>
              <Button variant="primary" onClick={handleStartUpload} disabled={isUploading || !selectedFile}>
                {isUploading ? "Processing..." : "Upload & Begin Analysis"}
              </Button>
            </div>
          </>
        )}
      </div>
    </Modal>
  );
};
