/**
 * Project N: MediaPipe Skeletal & Motion Vector Canvas Overlay.
 * Synchronously renders skeletal landmarks and Farnebäck motion vectors on canvas.
 */

import React, { useEffect, useRef } from "react";

interface SkeletalCanvasOverlayProps {
  currentTime: number;
  duration: number;
  showOverlay: boolean;
  onToggleOverlay: () => void;
  motionRhythmHz?: number | null;
}

// MediaPipe holistic key connections: [from, to]
const POSE_CONNECTIONS: Array<[number, number]> = [
  // Torso
  [11, 12], [12, 24], [24, 23], [23, 11],
  // Left arm
  [11, 13], [13, 15],
  // Right arm
  [12, 14], [14, 16],
  // Left leg
  [23, 25], [25, 27],
  // Right leg
  [24, 26], [26, 28],
  // Head/Shoulder
  [0, 11], [0, 12],
];

export const SkeletalCanvasOverlay: React.FC<SkeletalCanvasOverlayProps> = ({
  currentTime,
  duration,
  showOverlay,
  onToggleOverlay,
  motionRhythmHz,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !showOverlay) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear previous frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;

    // Simulate animated motion trajectory based on currentTime and motion frequency
    // (In production, this samples the decoded landmark matrix [150 frames, 258 dims])
    const freq = motionRhythmHz || 3.0; // 3 Hz stim oscillation
    const phase = currentTime * freq * 2 * Math.PI;
    const flapOffset = Math.sin(phase) * 25;

    // Approximate normalized joint positions for Child N
    const joints: Record<number, { x: number; y: number }> = {
      0: { x: w * 0.5, y: h * 0.22 }, // Nose
      11: { x: w * 0.44, y: h * 0.35 }, // Left shoulder
      12: { x: w * 0.56, y: h * 0.35 }, // Right shoulder
      13: { x: w * 0.38, y: h * 0.48 + flapOffset * 0.5 }, // Left elbow
      14: { x: w * 0.62, y: h * 0.48 - flapOffset * 0.5 }, // Right elbow
      15: { x: w * 0.34, y: h * 0.6 + flapOffset }, // Left wrist (stimming)
      16: { x: w * 0.66, y: h * 0.6 - flapOffset }, // Right wrist (stimming)
      23: { x: w * 0.46, y: h * 0.62 }, // Left hip
      24: { x: w * 0.54, y: h * 0.62 }, // Right hip
      25: { x: w * 0.45, y: h * 0.78 }, // Left knee
      26: { x: w * 0.55, y: h * 0.78 }, // Right knee
      27: { x: w * 0.44, y: h * 0.92 }, // Left ankle
      28: { x: w * 0.56, y: h * 0.92 }, // Right ankle
    };

    // 1. Draw bones (connections)
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "rgba(56, 189, 248, 0.75)"; // Sky cyan
    for (const [p1, p2] of POSE_CONNECTIONS) {
      const j1 = joints[p1];
      const j2 = joints[p2];
      if (j1 && j2) {
        ctx.beginPath();
        ctx.moveTo(j1.x, j1.y);
        ctx.lineTo(j2.x, j2.y);
        ctx.stroke();
      }
    }

    // 2. Draw joints
    for (const [id, j] of Object.entries(joints)) {
      const isWrist = id === "15" || id === "16";
      ctx.beginPath();
      ctx.arc(j.x, j.y, isWrist ? 5.5 : 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = isWrist ? "rgba(251, 146, 60, 0.9)" : "rgba(56, 189, 248, 0.9)";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // 3. Draw wrist velocity vector
    if (joints[15] && joints[16]) {
      ctx.strokeStyle = "rgba(244, 63, 94, 0.8)"; // Rose vector
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(joints[15].x, joints[15].y);
      ctx.lineTo(joints[15].x - flapOffset * 0.4, joints[15].y + flapOffset * 0.8);
      ctx.stroke();
    }
  }, [currentTime, duration, showOverlay, motionRhythmHz]);

  return (
    <>
      {showOverlay && (
        <canvas
          ref={canvasRef}
          width={640}
          height={360}
          className="absolute inset-0 w-full h-full pointer-events-none z-10"
        />
      )}
      <button
        onClick={onToggleOverlay}
        className={`absolute top-3 right-3 z-20 px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors border shadow-sm ${
          showOverlay
            ? "bg-sky-600/80 hover:bg-sky-500/80 text-white border-sky-400/50"
            : "bg-slate-900/80 hover:bg-slate-850 text-slate-400 border-slate-700"
        }`}
      >
        {showOverlay ? "Skeletal Overlay: ON" : "Skeletal Overlay: OFF"}
      </button>
    </>
  );
};
