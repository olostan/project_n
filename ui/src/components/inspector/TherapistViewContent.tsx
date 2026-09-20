/**
 * Project N: Therapist View Perspective Component.
 * Delivers dense bioacoustic telemetry, SCERTS taxonomy, Ayres SI domain, and academic citations.
 */

import React from "react";
import { BookOpen, ExternalLink, Network, Waves } from "lucide-react";

interface TherapistViewContentProps {
  l1Measured: string;
  l2Historical: string;
  l3Context: string;
  l4Evidence: string;
  f0MeanHz?: number | null;
  motionRhythmHz?: number | null;
  citations?: Array<{ title: string; authors: string; year: number; doi: string; level: string }>;
}

export const TherapistViewContent: React.FC<TherapistViewContentProps> = ({
  l1Measured,
  l2Historical,
  l3Context,
  l4Evidence,
  f0MeanHz,
  motionRhythmHz,
  citations = [],
}) => {
  return (
    <div className="flex flex-col gap-4 text-xs">
      {/* L1: Quantitative Acoustic & Kinematic Telemetry */}
      <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col gap-2">
        <div className="flex items-center justify-between text-xs font-semibold text-sky-400">
          <div className="flex items-center gap-2">
            <Waves className="w-4 h-4" />
            <span>Layer 1: Objective Bioacoustic & Kinematic Telemetry</span>
          </div>
          <span className="text-[10px] font-mono text-slate-500">Unfiltered Sensors</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 font-mono text-[11px]">
          <div className="p-2 rounded bg-slate-800/80 border border-slate-700/60">
            <span className="text-slate-400 block text-[10px]">F₀ Mean</span>
            <strong className="text-slate-100">{f0MeanHz ? `${Math.round(f0MeanHz)} Hz` : "N/A"}</strong>
          </div>
          <div className="p-2 rounded bg-slate-800/80 border border-slate-700/60">
            <span className="text-slate-400 block text-[10px]">Motion Frequency</span>
            <strong className="text-slate-100">{motionRhythmHz ? `${motionRhythmHz.toFixed(1)} Hz` : "N/A"}</strong>
          </div>
          <div className="p-2 rounded bg-slate-800/80 border border-slate-700/60">
            <span className="text-slate-400 block text-[10px]">Harmonic CQT</span>
            <strong className="text-slate-100">84 bins (7 oct)</strong>
          </div>
          <div className="p-2 rounded bg-slate-800/80 border border-slate-700/60">
            <span className="text-slate-400 block text-[10px]">Farnebäck Flow</span>
            <strong className="text-slate-100">8x8 grid (128d)</strong>
          </div>
        </div>

        <p className="text-slate-300 leading-relaxed font-mono text-[11px] mt-1">
          {l1Measured}
        </p>
      </div>

      {/* L2 & L3: Transactional Precedent & Context */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col gap-1">
          <div className="flex items-center gap-1.5 font-semibold text-emerald-400 text-xs">
            <Network className="w-3.5 h-3.5" />
            <span>Layer 2: Episodic Memory Matches (k-NN)</span>
          </div>
          <p className="text-slate-300 leading-relaxed text-[11px]">{l2Historical}</p>
        </div>

        <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col gap-1">
          <div className="flex items-center gap-1.5 font-semibold text-amber-400 text-xs">
            <Network className="w-3.5 h-3.5" />
            <span>Layer 3: Dyadic Antecedent Context</span>
          </div>
          <p className="text-slate-300 leading-relaxed text-[11px]">{l3Context}</p>
        </div>
      </div>

      {/* L4: Clinical Citations & Evidence Base */}
      <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col gap-2">
        <div className="flex items-center gap-2 font-semibold text-purple-400 text-xs">
          <BookOpen className="w-4 h-4" />
          <span>Layer 4: Grounded Clinical Literature & SCERTS Mapping</span>
        </div>
        <p className="text-slate-300 leading-relaxed text-[11px]">{l4Evidence}</p>

        {citations.length > 0 && (
          <div className="flex flex-col gap-1.5 mt-1 border-t border-slate-800 pt-2">
            {citations.map((c, i) => (
              <div key={i} className="flex items-center justify-between text-[11px] text-slate-400">
                <span>
                  <strong>{c.authors}</strong> ({c.year}). <em>{c.title}</em>
                </span>
                {c.doi && (
                  <a
                    href={`https://doi.org/${c.doi}`}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-sky-400 hover:text-sky-300"
                  >
                    <span>DOI</span>
                    <ExternalLink className="w-3 h-3" />
                  </a>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
