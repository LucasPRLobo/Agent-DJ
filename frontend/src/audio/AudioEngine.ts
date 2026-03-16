/**
 * AudioEngine — Two-deck Web Audio API mixing engine.
 *
 * Manages two decks (A and B) for DJ-style playback with real-time
 * transitions, EQ control, and beat-synchronized mixing.
 */

export interface DeckState {
  id: "A" | "B";
  trackUrl: string | null;
  buffer: AudioBuffer | null;
  source: AudioBufferSourceNode | null;
  gainNode: GainNode;
  eqLow: BiquadFilterNode;
  eqMid: BiquadFilterNode;
  eqHigh: BiquadFilterNode;
  isPlaying: boolean;
  startTime: number;      // AudioContext time when playback started
  offset: number;         // Offset into the buffer (seconds)
  playbackRate: number;
  looping: boolean;
  loopStart: number;
  loopEnd: number;
}

export type DeckId = "A" | "B";

export class AudioEngine {
  private ctx: AudioContext;
  private masterGain: GainNode;
  private decks: Record<DeckId, DeckState>;

  constructor() {
    this.ctx = new AudioContext();
    this.masterGain = this.ctx.createGain();
    this.masterGain.connect(this.ctx.destination);

    this.decks = {
      A: this.createDeck("A"),
      B: this.createDeck("B"),
    };
  }

  private createDeck(id: DeckId): DeckState {
    const gainNode = this.ctx.createGain();

    // 3-band EQ
    const eqLow = this.ctx.createBiquadFilter();
    eqLow.type = "lowshelf";
    eqLow.frequency.value = 250;
    eqLow.gain.value = 0;

    const eqMid = this.ctx.createBiquadFilter();
    eqMid.type = "peaking";
    eqMid.frequency.value = 1000;
    eqMid.Q.value = 1;
    eqMid.gain.value = 0;

    const eqHigh = this.ctx.createBiquadFilter();
    eqHigh.type = "highshelf";
    eqHigh.frequency.value = 4000;
    eqHigh.gain.value = 0;

    // Chain: source → eqLow → eqMid → eqHigh → gain → master
    eqLow.connect(eqMid);
    eqMid.connect(eqHigh);
    eqHigh.connect(gainNode);
    gainNode.connect(this.masterGain);

    return {
      id,
      trackUrl: null,
      buffer: null,
      source: null,
      gainNode,
      eqLow,
      eqMid,
      eqHigh,
      isPlaying: false,
      startTime: 0,
      offset: 0,
      playbackRate: 1.0,
      looping: false,
      loopStart: 0,
      loopEnd: 0,
    };
  }

  /** Resume audio context (required after user interaction). */
  async resume(): Promise<void> {
    if (this.ctx.state === "suspended") {
      await this.ctx.resume();
    }
  }

  /** Get current AudioContext time. */
  get currentTime(): number {
    return this.ctx.currentTime;
  }

  /** Load an audio file into a deck. */
  async loadTrack(deckId: DeckId, url: string): Promise<void> {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await this.ctx.decodeAudioData(arrayBuffer);

    const deck = this.decks[deckId];
    deck.buffer = audioBuffer;
    deck.trackUrl = url;
  }

  /** Start playback on a deck. */
  play(deckId: DeckId, offset: number = 0, playbackRate: number = 1.0): void {
    const deck = this.decks[deckId];
    if (!deck.buffer) return;

    // Stop any existing source
    this.stopDeck(deckId);

    const source = this.ctx.createBufferSource();
    source.buffer = deck.buffer;
    source.playbackRate.value = playbackRate;

    if (deck.looping) {
      source.loop = true;
      source.loopStart = deck.loopStart;
      source.loopEnd = deck.loopEnd;
    }

    source.connect(deck.eqLow);
    source.start(0, offset);

    deck.source = source;
    deck.isPlaying = true;
    deck.startTime = this.ctx.currentTime;
    deck.offset = offset;
    deck.playbackRate = playbackRate;

    source.onended = () => {
      if (deck.source === source) {
        deck.isPlaying = false;
      }
    };
  }

  /** Stop playback on a deck. */
  stopDeck(deckId: DeckId): void {
    const deck = this.decks[deckId];
    if (deck.source) {
      try {
        deck.source.stop();
      } catch {
        // Already stopped
      }
      deck.source.disconnect();
      deck.source = null;
    }
    deck.isPlaying = false;
  }

  /** Get current playback position of a deck in seconds. */
  getPosition(deckId: DeckId): number {
    const deck = this.decks[deckId];
    if (!deck.isPlaying) return deck.offset;
    const elapsed = (this.ctx.currentTime - deck.startTime) * deck.playbackRate;
    return deck.offset + elapsed;
  }

  /** Set deck volume (0-1). */
  setGain(deckId: DeckId, value: number): void {
    this.decks[deckId].gainNode.gain.value = value;
  }

  /** Ramp deck volume over time. */
  rampGain(deckId: DeckId, target: number, duration: number): void {
    const gain = this.decks[deckId].gainNode.gain;
    gain.cancelScheduledValues(this.ctx.currentTime);
    gain.setValueAtTime(gain.value, this.ctx.currentTime);
    gain.linearRampToValueAtTime(target, this.ctx.currentTime + duration);
  }

  /** Set EQ band gain in dB (-24 to +24). */
  setEQ(deckId: DeckId, band: "low" | "mid" | "high", gainDb: number): void {
    const deck = this.decks[deckId];
    const node = band === "low" ? deck.eqLow : band === "mid" ? deck.eqMid : deck.eqHigh;
    node.gain.value = gainDb;
  }

  /** Ramp EQ band over time. */
  rampEQ(deckId: DeckId, band: "low" | "mid" | "high", targetDb: number, duration: number): void {
    const deck = this.decks[deckId];
    const node = band === "low" ? deck.eqLow : band === "mid" ? deck.eqMid : deck.eqHigh;
    node.gain.cancelScheduledValues(this.ctx.currentTime);
    node.gain.setValueAtTime(node.gain.value, this.ctx.currentTime);
    node.gain.linearRampToValueAtTime(targetDb, this.ctx.currentTime + duration);
  }

  /** Set playback rate (speed/pitch). */
  setPlaybackRate(deckId: DeckId, rate: number): void {
    const deck = this.decks[deckId];
    deck.playbackRate = rate;
    if (deck.source) {
      deck.source.playbackRate.value = rate;
    }
  }

  /** Enable looping on a deck. */
  setLoop(deckId: DeckId, start: number, end: number): void {
    const deck = this.decks[deckId];
    deck.looping = true;
    deck.loopStart = start;
    deck.loopEnd = end;
    if (deck.source) {
      deck.source.loop = true;
      deck.source.loopStart = start;
      deck.source.loopEnd = end;
    }
  }

  /** Disable looping. */
  clearLoop(deckId: DeckId): void {
    const deck = this.decks[deckId];
    deck.looping = false;
    if (deck.source) {
      deck.source.loop = false;
    }
  }

  /** Set master volume (0-1). */
  setMasterGain(value: number): void {
    this.masterGain.gain.value = value;
  }

  /** Get deck state. */
  getDeckState(deckId: DeckId): DeckState {
    return this.decks[deckId];
  }

  /** Check if a deck has a track loaded. */
  isLoaded(deckId: DeckId): boolean {
    return this.decks[deckId].buffer !== null;
  }

  /** Get the duration of the loaded track. */
  getTrackDuration(deckId: DeckId): number {
    return this.decks[deckId].buffer?.duration ?? 0;
  }

  /** Clean up. */
  destroy(): void {
    this.stopDeck("A");
    this.stopDeck("B");
    this.ctx.close();
  }
}
