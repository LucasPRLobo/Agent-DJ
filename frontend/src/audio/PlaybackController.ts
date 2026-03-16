/**
 * PlaybackController — Orchestrates deck loading, playback, and transitions
 * based on a set plan received from the backend.
 *
 * Manages the two-deck system: always one deck playing, one preloading.
 * Listens for transition triggers and executes them.
 */

import { AudioEngine, type DeckId } from "./AudioEngine";
import { TransitionExecutor, type TransitionParams } from "./TransitionExecutor";

export interface TrackInfo {
  filePath: string;
  title: string;
  audioUrl: string;
  bpm: number;
  key: string;
  duration: number;
}

export interface TransitionInfo {
  type: "crossfade" | "bass_swap" | "loop_blend" | "hard_cut" | "echo_out" | "breakdown";
  durationSeconds: number;
  mixOutTime: number;     // When to start transition (seconds into current track)
  mixInTime: number;      // Where to start the next track
  targetBpm: number;
  loopStart?: number;
  loopEnd?: number;
}

export interface PlaybackState {
  currentTrack: TrackInfo | null;
  nextTrack: TrackInfo | null;
  activeDeck: DeckId;
  position: number;
  isPlaying: boolean;
  isTransitioning: boolean;
}

type PlaybackCallback = (state: PlaybackState) => void;

export class PlaybackController {
  private engine: AudioEngine;
  private executor: TransitionExecutor;
  private activeDeck: DeckId = "A";
  private currentTrack: TrackInfo | null = null;
  private nextTrack: TrackInfo | null = null;
  private nextTransition: TransitionInfo | null = null;
  private isPlaying = false;
  private tickInterval: number | null = null;
  private onStateChange: PlaybackCallback | null = null;

  constructor() {
    this.engine = new AudioEngine();
    this.executor = new TransitionExecutor(this.engine);
  }

  /** Set callback for state changes. */
  setOnStateChange(callback: PlaybackCallback): void {
    this.onStateChange = callback;
  }

  /** Get current playback state. */
  getState(): PlaybackState {
    return {
      currentTrack: this.currentTrack,
      nextTrack: this.nextTrack,
      activeDeck: this.activeDeck,
      position: this.engine.getPosition(this.activeDeck),
      isPlaying: this.isPlaying,
      isTransitioning: this.executor.isTransitioning,
    };
  }

  /** Load and start playing the first track. */
  async startPlayback(track: TrackInfo): Promise<void> {
    await this.engine.resume();
    await this.engine.loadTrack("A", track.audioUrl);

    this.currentTrack = track;
    this.activeDeck = "A";
    this.engine.setGain("A", 1);
    this.engine.play("A", 0, 1.0);
    this.isPlaying = true;

    this.startTick();
    this.emitState();
  }

  /** Queue the next track with its transition plan. */
  async queueNext(track: TrackInfo, transition: TransitionInfo): Promise<void> {
    const standbyDeck = this.activeDeck === "A" ? "B" : "A";

    // Pre-load into standby deck
    await this.engine.loadTrack(standbyDeck, track.audioUrl);

    this.nextTrack = track;
    this.nextTransition = transition;
    this.emitState();
  }

  /** Force trigger the next transition immediately. */
  async triggerTransition(): Promise<void> {
    if (!this.nextTrack || !this.nextTransition || this.executor.isTransitioning) {
      return;
    }

    const fromDeck = this.activeDeck;
    const toDeck: DeckId = fromDeck === "A" ? "B" : "A";
    const transition = this.nextTransition;

    // Calculate playback rate for BPM matching
    const toPlaybackRate = transition.targetBpm / (this.nextTrack.bpm || 120);

    const params: TransitionParams = {
      type: transition.type,
      durationSeconds: transition.durationSeconds,
      fromDeck,
      toDeck,
      toTrackUrl: this.nextTrack.audioUrl,
      toStartOffset: transition.mixInTime,
      toPlaybackRate,
      loopStart: transition.loopStart,
      loopEnd: transition.loopEnd,
    };

    // Execute transition
    this.emitState();
    await this.executor.execute(params);

    // Swap decks
    this.activeDeck = toDeck;
    this.currentTrack = this.nextTrack;
    this.nextTrack = null;
    this.nextTransition = null;

    this.emitState();
  }

  /** Skip current track (trigger early transition). */
  async skip(): Promise<void> {
    if (this.nextTrack) {
      await this.triggerTransition();
    }
  }

  /** Pause playback. */
  pause(): void {
    this.engine.stopDeck(this.activeDeck);
    this.isPlaying = false;
    this.stopTick();
    this.emitState();
  }

  /** Set master volume (0-1). */
  setVolume(value: number): void {
    this.engine.setMasterGain(value);
  }

  /** Clean up resources. */
  destroy(): void {
    this.stopTick();
    this.engine.destroy();
  }

  /** Start the tick loop that monitors playback position. */
  private startTick(): void {
    if (this.tickInterval) return;

    this.tickInterval = window.setInterval(() => {
      if (!this.isPlaying) return;

      const position = this.engine.getPosition(this.activeDeck);

      // Check if we should trigger the next transition
      if (this.nextTransition && this.nextTrack && !this.executor.isTransitioning) {
        if (position >= this.nextTransition.mixOutTime) {
          this.triggerTransition();
        }
      }

      this.emitState();
    }, 100); // 10 Hz update rate
  }

  private stopTick(): void {
    if (this.tickInterval) {
      clearInterval(this.tickInterval);
      this.tickInterval = null;
    }
  }

  private emitState(): void {
    if (this.onStateChange) {
      this.onStateChange(this.getState());
    }
  }
}
