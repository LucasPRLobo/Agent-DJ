# AI DJ Agent — MVP Plan

## Phase 1: Track Analyzer (Week 1)
Build the module that ingests audio files and extracts DJ-relevant features.

- [ ] Set up project structure (pyproject.toml, src layout)
- [ ] Audio feature extraction pipeline:
  - BPM detection
  - Key detection (Camelot wheel mapping)
  - Energy/loudness curve per track
  - Beat grid detection (downbeat positions)
  - Genre/mood tagging (via metadata or audio features)
- [ ] Track library: store analyzed tracks with their features (simple JSON/SQLite)
- [ ] Test with Lucas's sample tracks

## Phase 2: Vibe Parser + Set Planner (Week 1-2)
The brain — turn a natural language prompt into an ordered set.

- [ ] Vibe parser: LLM call that converts prompt → structured profile
  - Target genres, tempo range, energy arc shape, mood keywords
- [ ] Set planning algorithm:
  - Energy arc modeling (intro → build → peak → cool down → outro)
  - Track scoring against vibe profile
  - Ordering by BPM proximity + key compatibility (Camelot)
  - Transition point detection (best moment to mix in/out per track)
- [ ] Output: ordered set plan with transition metadata (JSON)

## Phase 3: Mix Engine (Week 2-3)
Turn the set plan into actual audio.

- [ ] Beat-aligned crossfades between tracks
- [ ] BPM matching via time stretching (pyrubberband)
- [ ] Basic transition types:
  - Smooth crossfade (default)
  - Cut transition (for energy drops)
  - Echo/fade out
- [ ] Export mixed set as WAV/MP3

## Phase 4: Interface + Polish (Week 3-4)
Make it usable.

- [ ] CLI interface for demo
- [ ] Web UI (simple — input prompt, see set plan, play/download mix)
- [ ] Spotify metadata enrichment (optional, for track discovery)
- [ ] Demo video / landing page for validation

## Success Criteria for MVP
- Input: "chill house party, RnB and jazz vibes, 45 minutes"
- Output: a mixed audio file with smooth transitions, good flow, energy arc
- Someone who doesn't know it's AI-generated thinks it sounds like a decent DJ set
