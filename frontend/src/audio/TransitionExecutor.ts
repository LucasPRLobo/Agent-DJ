/**
 * TransitionExecutor — Executes DJ transitions between two decks.
 *
 * Each transition type implements a specific mixing technique:
 * crossfade, bass swap, loop blend, hard cut, echo out, breakdown blend.
 */

import { AudioEngine, type DeckId } from "./AudioEngine";

export interface TransitionParams {
  type: "crossfade" | "bass_swap" | "loop_blend" | "hard_cut" | "echo_out" | "breakdown";
  durationSeconds: number;
  fromDeck: DeckId;
  toDeck: DeckId;
  toTrackUrl: string;
  toStartOffset: number;       // Where to start playing track B
  toPlaybackRate: number;      // BPM-adjusted playback rate for track B
  // Loop params (for loop_blend)
  loopStart?: number;
  loopEnd?: number;
}

export class TransitionExecutor {
  private engine: AudioEngine;
  private transitioning = false;

  constructor(engine: AudioEngine) {
    this.engine = engine;
  }

  get isTransitioning(): boolean {
    return this.transitioning;
  }

  /**
   * Execute a transition. Loads track B, then performs the transition.
   */
  async execute(params: TransitionParams): Promise<void> {
    if (this.transitioning) {
      console.warn("Transition already in progress");
      return;
    }

    this.transitioning = true;

    try {
      // Pre-load track B
      await this.engine.loadTrack(params.toDeck, params.toTrackUrl);

      switch (params.type) {
        case "crossfade":
          await this.crossfade(params);
          break;
        case "bass_swap":
          await this.bassSwap(params);
          break;
        case "loop_blend":
          await this.loopBlend(params);
          break;
        case "hard_cut":
          this.hardCut(params);
          break;
        case "echo_out":
          await this.echoOut(params);
          break;
        case "breakdown":
          await this.breakdownBlend(params);
          break;
      }
    } finally {
      this.transitioning = false;
    }
  }

  /**
   * Beatmatched crossfade — gradual volume swap over the transition duration.
   * Uses equal-power crossfade curve for smooth energy transfer.
   */
  private async crossfade(params: TransitionParams): Promise<void> {
    const { fromDeck, toDeck, durationSeconds, toStartOffset, toPlaybackRate } = params;

    // Start track B at low volume
    this.engine.setGain(toDeck, 0);
    this.engine.play(toDeck, toStartOffset, toPlaybackRate);

    // Equal-power crossfade
    this.engine.rampGain(fromDeck, 0, durationSeconds);
    this.engine.rampGain(toDeck, 1, durationSeconds);

    // Wait for transition to complete
    await this.wait(durationSeconds);

    // Stop deck A
    this.engine.stopDeck(fromDeck);
  }

  /**
   * Bass swap — transfer low frequencies from A to B while crossfading mids/highs.
   * This is the classic DJ EQ transition.
   */
  private async bassSwap(params: TransitionParams): Promise<void> {
    const { fromDeck, toDeck, durationSeconds, toStartOffset, toPlaybackRate } = params;
    const halfDuration = durationSeconds / 2;

    // Start track B with bass cut
    this.engine.setGain(toDeck, 0.7);
    this.engine.setEQ(toDeck, "low", -24);  // Cut bass on incoming
    this.engine.play(toDeck, toStartOffset, toPlaybackRate);

    // Phase 1: Blend in track B's mids/highs, cut track A's bass
    this.engine.rampEQ(fromDeck, "low", -24, halfDuration);  // Cut A's bass
    this.engine.rampEQ(toDeck, "low", 0, halfDuration);      // Bring in B's bass
    this.engine.rampGain(toDeck, 1, halfDuration);

    await this.wait(halfDuration);

    // Phase 2: Fade out track A
    this.engine.rampGain(fromDeck, 0, halfDuration);

    await this.wait(halfDuration);

    // Clean up
    this.engine.stopDeck(fromDeck);
    this.engine.setEQ(fromDeck, "low", 0);  // Reset EQ
  }

  /**
   * Loop blend — loop a section of track A while blending in track B.
   */
  private async loopBlend(params: TransitionParams): Promise<void> {
    const { fromDeck, toDeck, durationSeconds, toStartOffset, toPlaybackRate } = params;

    // Set up loop on track A
    if (params.loopStart !== undefined && params.loopEnd !== undefined) {
      this.engine.setLoop(fromDeck, params.loopStart, params.loopEnd);
    }

    // Start track B quietly
    this.engine.setGain(toDeck, 0);
    this.engine.play(toDeck, toStartOffset, toPlaybackRate);

    // Blend: fade B in over the loop
    this.engine.rampGain(toDeck, 1, durationSeconds * 0.7);

    await this.wait(durationSeconds * 0.7);

    // Release loop and fade A out
    this.engine.clearLoop(fromDeck);
    this.engine.rampGain(fromDeck, 0, durationSeconds * 0.3);

    await this.wait(durationSeconds * 0.3);

    this.engine.stopDeck(fromDeck);
  }

  /**
   * Hard cut — instant switch on a downbeat.
   */
  private hardCut(params: TransitionParams): void {
    const { fromDeck, toDeck, toStartOffset, toPlaybackRate } = params;

    // Instant switch
    this.engine.setGain(toDeck, 1);
    this.engine.play(toDeck, toStartOffset, toPlaybackRate);
    this.engine.setGain(fromDeck, 0);
    this.engine.stopDeck(fromDeck);
  }

  /**
   * Echo out — apply reverb tail to outgoing track, clean entry for new track.
   */
  private async echoOut(params: TransitionParams): Promise<void> {
    const { fromDeck, toDeck, durationSeconds, toStartOffset, toPlaybackRate } = params;

    // Start fading out track A with high-frequency emphasis (simulates reverb tail)
    this.engine.rampEQ(fromDeck, "high", 6, durationSeconds * 0.3);
    this.engine.rampEQ(fromDeck, "low", -12, durationSeconds * 0.5);
    this.engine.rampGain(fromDeck, 0, durationSeconds);

    // Bring in track B after a brief pause
    await this.wait(durationSeconds * 0.3);

    this.engine.setGain(toDeck, 0);
    this.engine.play(toDeck, toStartOffset, toPlaybackRate);
    this.engine.rampGain(toDeck, 1, durationSeconds * 0.5);

    await this.wait(durationSeconds * 0.7);

    this.engine.stopDeck(fromDeck);
    this.engine.setEQ(fromDeck, "high", 0);
    this.engine.setEQ(fromDeck, "low", 0);
  }

  /**
   * Breakdown blend — use a low-energy section as the transition point.
   * Similar to crossfade but timed to structural boundaries.
   */
  private async breakdownBlend(params: TransitionParams): Promise<void> {
    // Functionally same as crossfade, the timing is handled by the planner
    await this.crossfade(params);
  }

  private wait(seconds: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, seconds * 1000));
  }
}
