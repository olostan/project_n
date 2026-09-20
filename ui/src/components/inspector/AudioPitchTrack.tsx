/**
 * Project N: Synchronized Fundamental Frequency (F0) Pitch Canvas Track.
 * Displays continuous F0 pitch contour (Hz) with synchronized scrub cursor.
 */

import React, { useEffect, useRef } from "react";
import { Activity } from "lucide-react";

interface AudioPitchTrackProps {
  currentTime: number;
  duration: number;
  f0MeanHz?: number | null;
  onSeek: (timeSec: number) => void;
}

export const AudioPitchTrack: React.FC<AudioPitchTrackProps> = ({
  currentTime,
  duration,
  f0MeanHz,
  onSeek,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Generate synthetic pitch contour representing Child N's vocal trajectory
  const meanPitch = f0MeanHz || 260.0; // 260 Hz baseline
  const minPitch = 100;
  const maxPitch = 500;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    // Clear
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, w, h);

    // Draw horizontal grid lines (100 Hz, 200 Hz, 300 Hz, 400 Hz)
    ctx.strokeStyle = "rgba(51, 65, 85, 0.5)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);

    for (let pitch = 150; pitch <= 450; pitch += 100) {
      const y = h - ((pitch - minPitch) / (maxPitch - minPitch)) * h;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();

      ctx.fillStyle = "#64748b";
      ctx.font = "9px JetBrains Mono, monospace";
      ctx.fillText(`${pitch}Hz`, 4, y - 2);
    }
    ctx.setLineDash([]);

    // Draw F0 Contour
    const totalFrames = 300;
    ctx.strokeStyle = "#38bdf8"; // Sky cyan
    ctx.lineWidth = 2;
    ctx.beginPath();

    let started = false;
    for (let i = 0; i < totalFrames; i++) {
      const tNorm = i / totalFrames;
      // Voicing envelope (simulated vocal burst at 30%-70% of clip)
      const isVoiced = tNorm > 0.25 && tNorm < 0.75;
      if (!isVoiced) continue;

      // Simulated F0 inflection
      const pitchHz = meanPitch + Math.sin(tNorm * 18) * 45 + Math.cos(tNorm * 32) * 20;
      const x = tNorm * w;
      const y = h - ((pitchHz - minPitch) / (maxPitch - minPitch)) * h;

      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw Current Time Cursor
    const cursorX = duration > 0 ? (currentTime / duration) * w : 0;
    ctx.strokeStyle = "#f43f5e"; // Rose scrub cursor
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cursorX, 0);
    ctx.lineTo(cursorX, h);
    ctx.stroke();

    // Small cursor bead
    ctx.fillStyle = "#f43f5e";
    ctx.beginPath();
    ctx.arc(cursorX, 6, 4, 0, 2 * Math.PI);
    ctx.fill();
  }, [currentTime, duration, meanPitch]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || duration <= 0) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const norm = Math.max(0, Math.min(1, clickX / rect.width));
    onSeek(norm * duration);
  };

  return (
    <div className="flex flex-col gap-1.5 p-3 rounded-xl bg-slate-900 border border-slate-800">
      <div className="flex items-center justify-between text-xs text-slate-300">
        <div className="flex items-center gap-1.5 font-medium">
          <Activity className="w-3.5 h-3.5 text-sky-400" />
          <span>Acoustic Pitch Contour (F₀ Trace)</span>
        </div>
        <span className="font-mono text-[11px] text-slate-400">
          Baseline Mean: <strong className="text-sky-300">{Math.round(meanPitch)} Hz</strong>
        </span>
      </div>

      <canvas
        ref={canvasRef}
        width={600}
        height={100}
        onClick={handleCanvasClick}
        className="w-full h-24 rounded-lg cursor-pointer bg-slate-950 border border-slate-800/80"
        title="Click to seek audio and video"
      />
    </div>
  );
};
