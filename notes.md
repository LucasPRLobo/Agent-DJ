# AI DJ Agent — Brainstorm Notes

## Key Decisions (2026-03-16)

- **Approach: Hybrid (Option C)** — Spotify API for discovery/metadata, set planning as core product, real audio mixing as premium feature later
- **Demo mode** — Lucas provides sample audio files for hands-on demo with actual mixing
- **Interface TBD** — Likely web app, but could start with CLI/notebook for fastest iteration

## MVP Scope — "The DJ Brain"

### What it does
1. Takes a natural language prompt describing the vibe/event
2. Analyzes available tracks (metadata: BPM, key, energy, genre, mood)
3. Plans an intelligent set: song order, transition points, energy arc
4. For demo: actually mixes the provided sample tracks with crossfades, BPM matching

### What it does NOT do (yet)
- No real-time "reading the room"
- No streaming service integration in v1
- No mobile app
- No user accounts or payments

## Architecture Thinking

### Core Components
1. **Vibe Parser** — LLM takes natural language → structured vibe profile (genres, energy curve, mood tags, tempo range)
2. **Track Analyzer** — Extracts audio features from files (BPM, key, energy, segments) using librosa/essentia
3. **Set Planner** — The brain. Takes vibe profile + analyzed tracks → ordered set with transition metadata
   - Energy arc modeling (build up, peak, cool down)
   - Key compatibility (Camelot wheel)
   - BPM proximity for smooth transitions
   - Genre clustering and blending
4. **Mix Engine** — Takes the set plan + audio files → mixed output
   - Beat-aligned crossfades
   - BPM adjustment (time stretching)
   - Basic EQ transitions
5. **Output** — Export as audio file or stream playback

### Tech Stack
- Python 3.11+
- `librosa` / `essentia` — audio analysis (BPM, key, beat tracking)
- `pydub` / `soundfile` — audio manipulation
- `pyrubberband` — time stretching for BPM matching
- LLM (Claude API) — vibe parsing, natural language interface
- FastAPI — API layer when we go web
- Spotify API — metadata enrichment, track discovery (phase 2)

## Open Questions
- What audio format for demo files? (MP3/WAV/FLAC)
- How many sample tracks to start with? (suggest 20-30 for a meaningful demo)
- Target set length? (suggest 30-60 min for demo)
- Should the energy arc be auto-generated from the vibe description or user-configurable?
