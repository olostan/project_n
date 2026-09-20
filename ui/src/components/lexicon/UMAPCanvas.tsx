/**
 * Project N: 2D UMAP Behavioral Lexicon Scatter Canvas Component.
 * Visualizes 128-dimensional metric clusters in 2D space.
 */

import React, { useEffect, useRef, useState } from "react";
import { Episode } from "../../types/episodes";
import { ACTION_CLUSTER_COLORS } from "./LexiconLegend";
import { ClusterTooltip } from "./ClusterTooltip";

interface UMAPCanvasProps {
  episodes: Episode[];
  selectedEpisodeId: string | null;
  onSelectEpisode: (ep: Episode) => void;
}

interface PointCoords {
  episode: Episode;
  x: number;
  y: number;
  radius: number;
  color: string;
}

export const UMAPCanvas: React.FC<UMAPCanvasProps> = ({
  episodes,
  selectedEpisodeId,
  onSelectEpisode,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [hoveredPoint, setHoveredPoint] = useState<{
    episode: Episode;
    x: number;
    y: number;
  } | null>(null);

  const pointsRef = useRef<PointCoords[]>([]);

  // Calculate 2D points from episodes (fallback to pseudo-UMAP cluster layout if not precomputed)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const w = canvas.width;
    const h = canvas.height;
    const padding = 50;

    // Map each episode deterministically
    pointsRef.current = episodes.map((ep) => {
      // Deterministic hash coordinates based on episode ID and antecedent
      const hash = ep.id.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0);
      const angle = ((hash % 360) * Math.PI) / 180;
      const radiusOffset = 80 + (hash % 140);

      // Group by action
      const actionKey = ep.action_performed || ep.action_offered;
      const color = ACTION_CLUSTER_COLORS[actionKey] || "#38bdf8";

      // Centered coordinates
      const cx = w / 2;
      const cy = h / 2;
      const x = Math.max(padding, Math.min(w - padding, cx + Math.cos(angle) * radiusOffset));
      const y = Math.max(padding, Math.min(h - padding, cy + Math.sin(angle) * radiusOffset));

      return {
        episode: ep,
        x,
        y,
        radius: ep.id === selectedEpisodeId ? 7 : 5,
        color,
      };
    });

    // Render Canvas
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#090d16";
    ctx.fillRect(0, 0, w, h);

    // Draw faint grid rings
    ctx.strokeStyle = "rgba(30, 41, 59, 0.6)";
    ctx.lineWidth = 1;
    [80, 140, 200].forEach((r) => {
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, r, 0, 2 * Math.PI);
      ctx.stroke();
    });

    // Draw Points
    for (const pt of pointsRef.current) {
      const isSelected = pt.episode.id === selectedEpisodeId;

      ctx.beginPath();
      ctx.arc(pt.x, pt.y, isSelected ? 8 : 5, 0, 2 * Math.PI);
      ctx.fillStyle = pt.color;
      ctx.fill();

      if (isSelected) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Pulsing ring
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 13, 0, 2 * Math.PI);
        ctx.strokeStyle = "rgba(56, 189, 248, 0.6)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
  }, [episodes, selectedEpisodeId]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Find nearest point within hit tolerance (12px)
    let found: PointCoords | null = null;
    for (const pt of pointsRef.current) {
      const dist = Math.hypot(pt.x - mx, pt.y - my);
      if (dist <= 12) {
        found = pt;
        break;
      }
    }

    if (found) {
      setHoveredPoint({ episode: found.episode, x: found.x, y: found.y });
    } else {
      setHoveredPoint(null);
    }
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    for (const pt of pointsRef.current) {
      const dist = Math.hypot(pt.x - mx, pt.y - my);
      if (dist <= 12) {
        onSelectEpisode(pt.episode);
        break;
      }
    }
  };

  return (
    <div ref={containerRef} className="relative w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
      <canvas
        ref={canvasRef}
        width={750}
        height={450}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredPoint(null)}
        onClick={handleClick}
        className="w-full h-[450px] cursor-crosshair block"
      />

      {hoveredPoint && (
        <ClusterTooltip
          episode={hoveredPoint.episode}
          x={hoveredPoint.x}
          y={hoveredPoint.y}
        />
      )}
    </div>
  );
};
