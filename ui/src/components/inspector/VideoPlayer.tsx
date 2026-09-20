/**
 * Project N: HTML5 Video Player Container Component.
 * Implements synchronized controls, frame stepping (30 fps), and rate switching.
 */

import React from "react";
import {
  ChevronLeft,
  ChevronRight,
  Maximize,
  Pause,
  Play,
  Volume2,
  VolumeX,
} from "lucide-react";

interface VideoPlayerProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  mediaUrl: string;
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  playbackRate: number;
  onTogglePlay: () => void;
  onSeek: (timeSec: number) => void;
  onStepFrame: (forward?: boolean) => void;
  onChangeRate: (rate: number) => void;
  onTimeUpdate: () => void;
  onLoadedMetadata: () => void;
  children?: React.ReactNode; // Overlay canvas
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  videoRef,
  mediaUrl,
  currentTime,
  duration,
  isPlaying,
  playbackRate,
  onTogglePlay,
  onSeek,
  onStepFrame,
  onChangeRate,
  onTimeUpdate,
  onLoadedMetadata,
  children,
}) => {
  const [muted, setMuted] = React.useState<boolean>(false);

  const formatTime = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 10);
    return `${mins}:${secs.toString().padStart(2, "0")}.${ms}`;
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !muted;
      setMuted(!muted);
    }
  };

  const toggleFullscreen = () => {
    if (videoRef.current?.parentElement) {
      if (!document.fullscreenElement) {
        videoRef.current.parentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    }
  };

  return (
    <div className="flex flex-col bg-black rounded-xl overflow-hidden border border-slate-800 shadow-xl">
      {/* Video Viewport & Canvas Overlay Container */}
      <div className="relative w-full aspect-video bg-black flex items-center justify-center overflow-hidden">
        <video
          ref={videoRef}
          src={mediaUrl}
          onTimeUpdate={onTimeUpdate}
          onLoadedMetadata={onLoadedMetadata}
          className="w-full h-full object-contain"
          playsInline
        />
        {/* Render Skeletal Canvas Overlay as child */}
        {children}
      </div>

      {/* Control Bar */}
      <div className="p-3 bg-slate-900 border-t border-slate-800 flex flex-col gap-2">
        {/* Scrubber slider */}
        <input
          type="range"
          min={0}
          max={duration || 1}
          step={0.033}
          value={currentTime}
          onChange={(e) => onSeek(parseFloat(e.target.value))}
          className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-sky-500"
        />

        <div className="flex items-center justify-between text-xs text-slate-300">
          <div className="flex items-center gap-2">
            {/* Play / Pause */}
            <button
              onClick={onTogglePlay}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-100 transition-colors"
              aria-label={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </button>

            {/* Frame step controls (30 fps) */}
            <button
              onClick={() => onStepFrame(false)}
              className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
              title="Step backward 1 frame (33ms)"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => onStepFrame(true)}
              className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
              title="Step forward 1 frame (33ms)"
            >
              <ChevronRight className="w-4 h-4" />
            </button>

            {/* Timestamp */}
            <span className="font-mono text-[11px] text-slate-400 ml-1">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          <div className="flex items-center gap-3">
            {/* Playback speed */}
            <div className="flex items-center gap-1 text-[11px]">
              {[0.5, 1.0, 1.5].map((rate) => (
                <button
                  key={rate}
                  onClick={() => onChangeRate(rate)}
                  className={`px-1.5 py-0.5 rounded ${
                    playbackRate === rate
                      ? "bg-sky-600 text-white font-medium"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                  }`}
                >
                  {rate}x
                </button>
              ))}
            </div>

            {/* Audio Mute */}
            <button
              onClick={toggleMute}
              className="p-1 text-slate-400 hover:text-slate-200"
              title={muted ? "Unmute" : "Mute"}
            >
              {muted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </button>

            {/* Fullscreen */}
            <button
              onClick={toggleFullscreen}
              className="p-1 text-slate-400 hover:text-slate-200"
              title="Fullscreen"
            >
              <Maximize className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
