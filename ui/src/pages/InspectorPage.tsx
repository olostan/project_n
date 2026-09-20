/**
 * Project N: Multimodal Video Inspector & Four-Layer Insight Page.
 */

import React, { useState } from "react";
import { getEpisodeMediaUrl } from "../api/episodesApi";
import { useSynchronizedPlayback } from "../hooks/useSynchronizedPlayback";
import { Episode, FourLayerCardData } from "../types/episodes";
import { VideoPlayer } from "../components/inspector/VideoPlayer";
import { SkeletalCanvasOverlay } from "../components/inspector/SkeletalCanvasOverlay";
import { AudioPitchTrack } from "../components/inspector/AudioPitchTrack";
import { FourLayerCard } from "../components/inspector/FourLayerCard";
import { OutcomeLoggerModal } from "../components/inspector/OutcomeLoggerModal";
import { Button } from "../components/common/Button";
import { ArrowLeft, HeartPulse } from "lucide-react";

interface InspectorPageProps {
  episode: Episode | null;
  cardData?: FourLayerCardData | null;
  onBackToDiary: () => void;
  onNavigateToTriage: (episodeId: string) => void;
  onEpisodeUpdated: () => void;
}

export const InspectorPage: React.FC<InspectorPageProps> = ({
  episode,
  cardData,
  onBackToDiary,
  onNavigateToTriage,
  onEpisodeUpdated,
}) => {
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [isOutcomeModalOpen, setIsOutcomeModalOpen] = useState<boolean>(false);

  const {
    videoRef,
    currentTime,
    duration,
    isPlaying,
    playbackRate,
    togglePlay,
    seek,
    stepFrame,
    changeRate,
    onTimeUpdate,
    onLoadedMetadata,
  } = useSynchronizedPlayback();

  if (!episode) {
    return (
      <div className="max-w-4xl mx-auto p-8 text-center flex flex-col items-center gap-3">
        <p className="text-slate-400 text-sm">No episode selected for inspection.</p>
        <Button variant="primary" size="sm" onClick={onBackToDiary}>
          Return to Episode Diary
        </Button>
      </div>
    );
  }

  const mediaUrl = getEpisodeMediaUrl(episode.id);

  return (
    <div className="flex flex-col gap-5 max-w-6xl mx-auto p-4 sm:p-6 text-slate-100">
      {/* Top action bar */}
      <div className="flex items-center justify-between">
        <button
          onClick={onBackToDiary}
          className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Diary</span>
        </button>

        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => onNavigateToTriage(episode.id)}
            className="gap-1.5 text-amber-300 border-amber-800/60"
          >
            <HeartPulse className="w-3.5 h-3.5 text-amber-400" />
            <span>NCCPC Triage</span>
          </Button>

          <Button
            variant="primary"
            size="sm"
            onClick={() => setIsOutcomeModalOpen(true)}
          >
            Log Outcome
          </Button>
        </div>
      </div>

      {/* Main Grid: Left = Video + Audio Track, Right = Four-Layer Insight Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Multimodal Playback Column (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <VideoPlayer
            videoRef={videoRef}
            mediaUrl={mediaUrl}
            currentTime={currentTime}
            duration={duration}
            isPlaying={isPlaying}
            playbackRate={playbackRate}
            onTogglePlay={togglePlay}
            onSeek={seek}
            onStepFrame={stepFrame}
            onChangeRate={changeRate}
            onTimeUpdate={onTimeUpdate}
            onLoadedMetadata={onLoadedMetadata}
          >
            <SkeletalCanvasOverlay
              currentTime={currentTime}
              duration={duration}
              showOverlay={showOverlay}
              onToggleOverlay={() => setShowOverlay(!showOverlay)}
              motionRhythmHz={episode.observed_motion_rhythm_hz}
            />
          </VideoPlayer>

          {/* Synchronized Pitch Contour Track */}
          <AudioPitchTrack
            currentTime={currentTime}
            duration={duration}
            f0MeanHz={episode.observed_f0_mean}
            onSeek={seek}
          />
        </div>

        {/* Four-Layer Insight Card Column (5 cols) */}
        <div className="lg:col-span-5 flex flex-col">
          <FourLayerCard
            episode={episode}
            cardData={cardData}
            onOpenOutcomeLogger={() => setIsOutcomeModalOpen(true)}
          />
        </div>
      </div>

      {/* Outcome Recording Modal */}
      {isOutcomeModalOpen && (
        <OutcomeLoggerModal
          isOpen={isOutcomeModalOpen}
          onClose={() => setIsOutcomeModalOpen(false)}
          episode={episode}
          onSuccess={onEpisodeUpdated}
        />
      )}
    </div>
  );
};
