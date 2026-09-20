/**
 * Project N: Synchronized Multimodal Playback Hook.
 * Synchronizes HTML5 video playback with canvas skeletal overlays and audio pitch tracks.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export function useSynchronizedPlayback() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackRate, setPlaybackRate] = useState<number>(1.0);

  const animationFrameRef = useRef<number | null>(null);

  const onTimeUpdate = useCallback(() => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  }, []);

  const onLoadedMetadata = useCallback(() => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration || 0);
      setCurrentTime(videoRef.current.currentTime || 0);
    }
  }, []);

  const togglePlay = useCallback(() => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      videoRef.current.play();
      setIsPlaying(true);
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  }, []);

  const seek = useCallback((timeSec: number) => {
    if (!videoRef.current) return;
    const clamped = Math.max(0, Math.min(timeSec, videoRef.current.duration || 0));
    videoRef.current.currentTime = clamped;
    setCurrentTime(clamped);
  }, []);

  const stepFrame = useCallback((forward = true) => {
    if (!videoRef.current) return;
    videoRef.current.pause();
    setIsPlaying(false);
    const frameDuration = 1 / 30; // 30 fps
    const newTime = videoRef.current.currentTime + (forward ? frameDuration : -frameDuration);
    seek(newTime);
  }, [seek]);

  const changeRate = useCallback((rate: number) => {
    if (!videoRef.current) return;
    videoRef.current.playbackRate = rate;
    setPlaybackRate(rate);
  }, []);

  // RAF loop for smooth 60fps tracking during playback
  useEffect(() => {
    function tick() {
      if (videoRef.current && !videoRef.current.paused) {
        setCurrentTime(videoRef.current.currentTime);
      }
      animationFrameRef.current = requestAnimationFrame(tick);
    }
    animationFrameRef.current = requestAnimationFrame(tick);
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, []);

  return {
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
  };
}
