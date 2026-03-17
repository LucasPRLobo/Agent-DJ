# Agent DJ — Detailed Implementation Plan

## Vision

A live, interactive AI DJ that runs in the browser. The host describes their party via chat, the DJ starts playing and adapts in real-time. Guests join via QR code to request songs and influence the vibe. Mixing happens client-side via Web Audio API for real-time control.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      BROWSER (Client)                    │
│                                                          │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Chat UI  │  │ DJ Dashboard │  │  Web Audio Engine  │  │
│  │(onboard, │  │(now playing, │  │(2-deck playback,  │  │
│  │ commands) │  │ queue, arc)  │  │ crossfade, EQ,    │  │
│  └────┬─────┘  └──────┬───────┘  │ BPM sync, loops)  │  │
│       │               │          └────────┬──────────┘  │
│       └───────┬───────┘                   │             │
│               │ WebSocket                 │ Audio files  │
│               ▼                           │ via HTTP     │
├───────────────┼───────────────────────────┼─────────────┤
│               │        BACKEND            │             │
│  ┌────────────▼────────────┐   ┌─────────▼──────────┐  │
│  │     Session Manager     │   │   File Server       │  │
│  │  (rooms, permissions,   │   │   (serves audio     │  │
│  │   host/guest state)     │   │    to browser)      │  │
│  └────────────┬────────────┘   └────────────────────┘  │
│               │                                         │
│  ┌────────────▼────────────┐                            │
│  │      DJ Brain           │                            │
│  │  ┌──────────────────┐   │                            │
│  │  │ Vibe Parser (LLM)│   │                            │
│  │  └────────┬─────────┘   │                            │
│  │  ┌────────▼─────────┐   │                            │
│  │  │ Track Selector   │   │                            │
│  │  │ (data science)   │   │                            │
│  │  └────────┬─────────┘   │                            │
│  │  ┌────────▼─────────┐   │                            │
│  │  │Transition Planner│   │                            │
│  │  └──────────────────┘   │                            │
│  └─────────────────────────┘                            │
│                                                          │
│  ┌─────────────────────────┐                            │
│  │    Track Analyzer       │                            │
│  │  (offline, pre-process) │                            │
│  └─────────────────────────┘                            │
└─────────────────────────────────────────────────────────┘
```

**Key architectural decisions:**
- Mixing happens in the browser (Web Audio API) — enables real-time transitions
- Backend is the brain — decides what to play and how to transition
- LLM is only used for chat (onboarding, mid-session commands) — not for track selection
- Track selection uses data science approaches on audio features
- Track analysis is an offline pre-processing step

---

## Phase 1: Track Analyzer (Offline Pipeline)

**Goal:** Build a reliable pipeline that extracts DJ-relevant features from audio files. This runs once per track and stores results. Everything downstream depends on accurate analysis.

### 1.1 Audio Feature Extraction

**Input:** Audio file (MP3/WAV/FLAC)
**Output:** TrackProfile (stored in SQLite)

Features to extract:

| Feature | Library | Method | Notes |
|---|---|---|---|
| BPM | librosa | `beat.beat_track` | Also try `essentia.RhythmExtractor2013` and compare accuracy |
| Beat grid | librosa | `beat.beat_track` → frame-to-time | Array of beat timestamps in seconds |
| Downbeats | librosa | Every 4th beat (4/4 assumption) | Bar start positions |
| Key + scale | essentia | `KeyExtractor` | Map to Camelot wheel notation |
| Energy curve | librosa | RMS energy per 2-bar segment, normalized 0-1 | Critical for set planning |
| Loudness (LUFS) | pyloudnorm | `Meter.integrated_loudness` | For volume normalization across tracks |
| Spectral centroid | librosa | `feature.spectral_centroid` | Brightness — helps cluster similar-sounding tracks |
| Spectral bandwidth | librosa | `feature.spectral_bandwidth` | Frequency spread |
| MFCCs | librosa | `feature.mfcc` (mean of 13 coefficients) | Timbral fingerprint — useful for similarity |
| Onset strength | librosa | `onset.onset_strength` | Rhythmic density / percussiveness |

### 1.2 Structure Detection

Detect the structural sections of a track (intro, verse, chorus, breakdown, outro). This is essential for finding intelligent mix points.

**Approach:** Use librosa's self-similarity matrix + novelty detection.

1. Compute chroma or MFCC features over time
2. Build a self-similarity matrix (recurrence plot)
3. Detect novelty peaks (structural boundaries) using a checkerboard kernel
4. Label sections based on:
   - Position in track (first section → intro, last → outro)
   - Energy level relative to average (high → chorus/peak, low → breakdown)
   - Repetition in self-similarity matrix (repeated sections → chorus)

**Fallback:** If novelty detection is noisy, fall back to energy-based segmentation (split into 8-bar chunks, label by energy relative to mean).

### 1.3 Mix Point Detection

For each track, identify:

- **Mix-in points:** Where another DJ would start blending this track in
  - Start of intro (safest)
  - After a breakdown (energy reset)
  - Any low-energy section with a clear beat grid

- **Mix-out points:** Where to start transitioning away from this track
  - Start of outro
  - Before/during a breakdown in the second half
  - Any energy drop after the main peak

- **Loop candidates:** Sections suitable for looping to extend a transition
  - Low-energy rhythmic sections (breakdowns, minimal intros)
  - Sections with consistent beat grid and low harmonic movement
  - Minimum 4 bars, maximum 8 bars

Each point gets a confidence score (0-1) and a duration.

### 1.4 ML Classification & Audio Embeddings

Use pre-trained models out of the box for genre, mood, and deep audio embeddings. No manual tagging needed.

**Layer 1: Essentia Pre-Trained Classifiers**

Essentia ships with TensorFlow models trained on large labeled music datasets. We use these for:

| Model | Output | Use Case |
|---|---|---|
| `genre_discogs400` | 400 genre/style labels with probabilities | Genre classification (top-N labels per track) |
| `mood_aggressive`, `mood_happy`, `mood_relaxed`, `mood_sad` | Binary mood classifiers (probability) | Mood tagging |
| `danceability` | Danceability score 0-1 | Filter for party contexts |
| `voice_instrumental` | Vocal presence score | Useful for transition planning (instrumental sections mix easier) |
| `tonal_atonal` | Tonality score | Harmonic content detection |

Pipeline:
1. Load audio → compute mel-spectrogram (Essentia's `TensorflowInputMusiCNN`)
2. Feed spectrogram into each classifier
3. Store top-5 genre predictions with confidence, mood scores, danceability score

**Layer 2: Audio Embeddings (MusiCNN / VGGish)**

Instead of hand-picking features for similarity, extract a dense embedding vector that captures the full "character" of a track.

- Use Essentia's `TensorflowPredictMusiCNN` to extract a 200-dim embedding from the penultimate layer
- This vector captures timbral quality, rhythm patterns, instrumentation, and overall feel
- Two tracks with similar embeddings will "sound similar" in ways that BPM+key+genre tags can't capture
- Store the mean embedding per track (averaged over time frames)

**These embeddings become the backbone of Phase 2 track selection** — cosine similarity between embeddings gives us a powerful measure of "these tracks belong together."

**Layer 3: Evaluation (leads into Phase 1.5)**

- Run all classifiers on sample tracks
- Compare genre predictions against known genres (manually verify 20-30 tracks)
- Visualize the embedding space (UMAP/t-SNE) — do similar tracks cluster together?
- Identify failure modes: which genres/moods are poorly classified?

### 1.5 Track Store

- SQLite database for fast querying
- Indexed columns: BPM, key (Camelot), loudness, duration
- Full profile stored as JSON blob
- CLI commands: `analyze <path>`, `library`, `inspect <track>`

### 1.6 Deliverables

- `agent_dj/analyzer/audio.py` — Low-level feature extraction (BPM, key, energy, beats)
- `agent_dj/analyzer/structure.py` — Structure detection (self-similarity + novelty)
- `agent_dj/analyzer/classifier.py` — ML classification pipeline (genre, mood, danceability, embeddings)
- `agent_dj/analyzer/camelot.py` — Key ↔ Camelot mapping and compatibility
- `agent_dj/analyzer/track_store.py` — SQLite store
- CLI commands: `analyze`, `library`, `inspect`
- Test with sample tracks, validate BPM/key accuracy against known values

---

## Phase 1.5: Model Evaluation & Fine-Tuning

**Goal:** Evaluate the pre-trained classifiers on our target music (RnB, jazz, electronic, etc.), identify weaknesses, and fine-tune or build a custom model if needed.

### 1.5.1 Evaluation Dataset

We need labeled data to measure how well the pre-trained models work:

**Option A: Existing datasets (free, research-licensed)**
- **MTG-Jamendo Dataset** — 55,000 tracks, 195 tags (genre, mood, instrument). CC-licensed audio. Best option for our use case.
- **FMA (Free Music Archive)** — 106,000 tracks, genre labels, audio features pre-computed. Good for genre classification benchmarking.
- **GTZAN** — 1,000 tracks, 10 genres. Small but standard benchmark.
- **Million Song Dataset** — Metadata only (no audio), but useful for tag co-occurrence patterns.

**Option B: Build our own evaluation set**
- Download 100-200 tracks across target genres (RnB, jazz, neo-soul, house, electronic, hip-hop, bossa nova)
- Manually label: genre (primary + secondary), mood (3-5 tags), energy level, danceability
- This becomes our ground truth for evaluation
- Small but highly relevant to our actual use case

**Recommendation:** Use both — MTG-Jamendo for broad evaluation, custom set for target-genre accuracy.

### 1.5.2 Evaluation Protocol

Run in Jupyter notebooks (`experiments/`):

**Notebook 1: Genre Classification Accuracy**
- Run Essentia genre classifier on evaluation set
- Metrics: top-1 accuracy, top-5 accuracy, confusion matrix
- Identify: which genres are well-classified? Which are confused with each other?
- Visualize: prediction confidence distribution per genre

**Notebook 2: Mood/Danceability Accuracy**
- Compare mood predictions against manual labels
- Correlation analysis: do mood scores align with human perception?
- Identify systematic biases (e.g., "does it always rate jazz as 'relaxed'?")

**Notebook 3: Embedding Space Quality**
- Extract embeddings for all tracks
- UMAP/t-SNE visualization — do similar genres/moods cluster together?
- Nearest-neighbor test: for each track, are the 5 nearest tracks musically similar? (manual check)
- This is the most important test — if embeddings cluster well, track selection will work

### 1.5.3 Fine-Tuning Strategy (If Needed)

Based on evaluation results, we have several paths:

**Path A: Embeddings are good, classifiers are weak**
- Keep the embedding model as-is (it's the backbone for similarity)
- Train a lightweight classifier (logistic regression, small MLP) on top of embeddings
- Training data: MTG-Jamendo labels mapped to our target genre taxonomy
- Fast to train, doesn't require modifying the base model

**Path B: Embeddings are weak for our genres**
- Fine-tune the MusiCNN model on MTG-Jamendo with focus on our target genres
- Use transfer learning: freeze early layers, retrain last 2-3 layers
- Requires more compute but gives us better representations

**Path C: Everything works well enough**
- Move on to Phase 2 with the pre-trained models
- Revisit fine-tuning later when we have more data from real usage

### 1.5.4 Custom Genre Taxonomy

The pre-trained model outputs 400 Discogs labels. We need to map these to a simpler taxonomy relevant to DJing:

```
DJ Genre Taxonomy (example):
├── Electronic
│   ├── House (deep, tech, progressive, afro)
│   ├── Techno (minimal, industrial, melodic)
│   ├── Drum & Bass
│   ├── Disco / Nu-Disco
│   └── Ambient / Downtempo
├── Urban
│   ├── RnB / Neo-Soul
│   ├── Hip-Hop / Rap
│   ├── Afrobeats
│   └── Reggaeton
├── Jazz
│   ├── Jazz (traditional, modal, bebop)
│   ├── Jazz Fusion
│   ├── Bossa Nova / Latin Jazz
│   └── Acid Jazz
├── Soul / Funk
│   ├── Soul (classic, northern)
│   ├── Funk
│   └── Motown
├── Rock / Indie
│   ├── Indie / Alternative
│   ├── Psychedelic
│   └── Post-Punk
└── World
    ├── Afrobeat
    ├── Brazilian
    ├── Reggae / Dub
    └── Other
```

Map Discogs-400 labels → our taxonomy. A track can have multiple genre labels with weights.

### 1.5.5 Deliverables

- `experiments/01_genre_eval.ipynb` — Genre classification evaluation
- `experiments/02_mood_eval.ipynb` — Mood/danceability evaluation
- `experiments/03_embedding_space.ipynb` — Embedding quality analysis
- `experiments/04_finetune.ipynb` — Fine-tuning (if needed)
- `agent_dj/analyzer/genre_taxonomy.py` — Discogs-400 → DJ genre mapping
- Decision document: which models to keep, what to fine-tune, what accuracy we achieved

---

## Phase 2: Track Selection & Set Planning (Data Science)

**Goal:** Given a VibeProfile and a library of analyzed tracks, select and order tracks for the set. This is where we experiment with different strategies.

### 2.1 VibeProfile Schema

The LLM produces this from the onboarding chat (Phase 4 implements the chat, but we define the schema here and can create profiles manually for testing):

```python
@dataclass
class VibeProfile:
    genres: list[str]           # ["rnb", "jazz", "neo-soul"]
    bpm_range: tuple[int, int]  # (85, 115)
    energy_arc: list[float]     # normalized curve [0.3, 0.5, 0.7, 0.9, 0.7, 0.4]
    mood_tags: list[str]        # ["smooth", "groovy", "warm"]
    duration_minutes: int       # 45
    avoid: list[str]            # genres/artists to skip
    transition_style: str       # "smooth", "energetic", "mixed"
    example_tracks: list[str]   # optional reference tracks from user
```

### 2.2 Track Selection Strategies (Experiments)

We implement multiple strategies, test them, and record results. Each strategy takes a VibeProfile + track library → ordered list of tracks.

**Strategy A: Feature-Distance Scoring (Baseline)**

For each position in the set:
1. Calculate target energy from the arc curve
2. Score every candidate track:
   - Energy match: `1 - |track_energy - target_energy|` (weight: 0.30)
   - BPM fit: `1 - |track_bpm - target_bpm| / bpm_range_width` (weight: 0.25)
   - Key compatibility with previous track: Camelot distance → score (weight: 0.20)
   - Genre/mood match to VibeProfile: tag overlap score (weight: 0.15)
   - Transition feasibility: mix-out quality of prev + mix-in quality of next (weight: 0.10)
3. Select highest-scoring track, remove from candidates, advance position
4. Greedy — picks locally optimal track each step

**Strategy B: Similarity Graph + Path Optimization**

1. Build a weighted graph where tracks are nodes
2. Edge weights = transition quality score between two tracks:
   - BPM proximity (penalize >5 BPM difference)
   - Key compatibility (Camelot distance)
   - Timbral similarity (cosine distance of MFCC vectors)
   - Energy continuity (smooth changes preferred)
3. Find the best path through the graph that:
   - Follows the target energy arc
   - Visits ~N tracks (based on duration)
   - Maximizes total edge quality
4. Solve with: greedy path, beam search, or simulated annealing

**Strategy C: Embedding-Based Clustering**

1. Build a feature vector per track: [BPM_norm, key_numeric, energy_mean, spectral_centroid_mean, MFCCs...]
2. Reduce to 2D with UMAP/t-SNE for visualization
3. Cluster tracks (KMeans or DBSCAN) — clusters represent natural "pockets" of compatible tracks
4. Plan the set as a journey through clusters:
   - Start in the cluster closest to the opening vibe
   - Transition between clusters based on the energy arc
   - Within a cluster, order by BPM/key compatibility

**Strategy D: Hybrid (likely the winner)**

- Use Strategy C clusters to pre-filter candidates into compatible groups
- Use Strategy B graph within and between clusters for ordering
- Use Strategy A scoring as a tiebreaker

### 2.3 Experiment Framework

For each strategy, we record:
- Track order produced
- Total transition quality score (sum of pairwise scores)
- Energy arc adherence (MSE between target and actual energy curve)
- BPM smoothness (max BPM jump in the set)
- Key compatibility rate (% of transitions that are Camelot-compatible)
- Subjective quality (manual listening test)

Results stored in `experiments/` as JSON + plots.

### 2.4 Transition Planning

Once track order is decided, plan each transition:

1. Look at mix-out points of track A, mix-in points of track B
2. Pick the best pair (highest combined confidence)
3. Determine transition type based on:
   - Energy context: dropping energy → echo out or breakdown blend; rising → bass swap or crossfade
   - BPM difference: small (<3 BPM) → crossfade; medium (3-8) → loop + tempo adjust; large (>8) → hard cut or breakdown
   - Genre: electronic → bass swaps, EQ transitions; acoustic/jazz → longer crossfades
4. Calculate BPM adjustment: which track stretches, by how much
5. Output a TransitionPlan per pair

### 2.5 Deliverables

- `agent_dj/planner/vibe_profile.py` — VibeProfile dataclass + manual creation for testing
- `agent_dj/planner/strategies/` — One module per strategy (scoring, graph, clustering, hybrid)
- `agent_dj/planner/transition_planner.py` — Pairwise transition planning
- `agent_dj/planner/set_planner.py` — Orchestrator: vibe + library → SetPlan (track order + transitions)
- `experiments/` — Notebook or scripts comparing strategies
- Visualization: energy arc plot, BPM flow, key compatibility heatmap

---

## Phase 3: Mix Engine (Web Audio API — Client-Side)

**Goal:** Build the browser-based audio engine that plays tracks and executes transitions in real-time based on the transition plan from the backend.

### 3.1 Web Audio Architecture

```
AudioContext
├── Deck A: AudioBufferSourceNode → GainNode → BiquadFilterNode (EQ) → MasterGain → Destination
├── Deck B: AudioBufferSourceNode → GainNode → BiquadFilterNode (EQ) → MasterGain → Destination
└── Effects: ConvolverNode (reverb), DelayNode (echo)
```

Two-deck system, like a real DJ setup:
- Always one deck playing, one on standby or pre-loading
- Transitions blend between decks
- Each deck has independent gain and EQ control

### 3.2 Playback Controller

The client receives a `SetPlan` from the backend containing:
- Ordered list of tracks with URLs (served by backend)
- Per-track: play-from timestamp, play-to timestamp, target BPM, gain adjustment
- Per-transition: type, duration, start time, parameters

The playback controller:
1. Pre-loads the next track into the standby deck while current track plays
2. At the transition start time, begins executing the transition
3. Manages timing: all operations aligned to the beat grid
4. Reports current state back to backend via WebSocket (position, deck status)

### 3.3 Transition Implementations (Client-Side)

**Beatmatched Crossfade:**
- Both decks playing simultaneously for 16-32 beats
- Deck A gain: linear or equal-power fade out
- Deck B gain: linear or equal-power fade in
- Beat grids aligned by adjusting Deck B start time to nearest beat

**Bass Swap (EQ Transition):**
- Both decks playing
- Deck A: gradually apply high-pass filter (cut bass over 8-16 beats)
- Deck B: start with high-pass filter, gradually remove it
- Crossfade the mids/highs simultaneously
- Result: bass transfers from A to B cleanly

**Loop & Blend:**
- Identify a loop region in Deck A (from transition plan)
- Deck A enters the loop (AudioBufferSourceNode with `loop=true`, `loopStart`, `loopEnd`)
- Deck B fades in over the looped section
- When Deck B is fully in, release the loop and fade Deck A out

**Hard Cut on Downbeat:**
- Deck A playing, Deck B pre-loaded and cued to its first downbeat
- On the target downbeat: Deck A gain → 0 instantly, Deck B gain → 1 instantly
- No overlap, clean switch

**Echo/Reverb Tail:**
- Route Deck A through a ConvolverNode (reverb) or DelayNode
- Increase wet signal while decreasing dry
- Fade Deck A out (including reverb tail)
- Deck B fades in clean underneath

### 3.4 BPM Synchronization

- Web Audio API's `playbackRate` property can adjust speed (and pitch)
- For small adjustments (<5%): use playbackRate directly (pitch shift is minimal)
- For larger adjustments: we may need to pre-process with rubberband server-side (time-stretch without pitch shift) and serve the adjusted file
- Decision: for MVP, use playbackRate for ±5% and pre-process anything beyond that

### 3.5 Deliverables

- `frontend/src/audio/AudioEngine.js` — Web Audio API setup, deck management
- `frontend/src/audio/TransitionExecutor.js` — Implements each transition type
- `frontend/src/audio/BeatSync.js` — Beat grid alignment and timing
- `frontend/src/audio/PlaybackController.js` — Orchestrates deck loading, transition timing, state reporting
- Test page: load 2 tracks, execute each transition type, verify quality

---

## Phase 4: Chat Interface + Vibe Parser (LLM)

**Goal:** The conversational layer where the host talks to the DJ. LLM handles natural language only — converts chat to structured commands.

### 4.1 Onboarding Flow

The DJ chat guides the host through setup:

1. **Greeting + occasion** — "Hey! What's the event tonight?"
   → Extracts: event type, audience size/description
2. **Music taste** — "What genres or artists set the vibe?"
   → Extracts: genre list, mood keywords, optional example tracks
3. **Duration** — "How long should I play for?"
   → Extracts: duration in minutes
4. **Energy shape** — "Should I start chill and build up, keep it steady, or go hard from the start?"
   → Extracts: energy arc type or custom description
5. **Boundaries** — "Anything I should avoid?"
   → Extracts: genre/artist blocklist, explicit content preference
6. **Confirmation** — Summarize the vibe profile back, ask to confirm or adjust

LLM outputs a structured VibeProfile JSON at the end.

### 4.2 Mid-Session Commands

During playback, the host (or guests with permission) can chat:

- "Make it more upbeat" → Adjust energy arc upward from current position
- "Play some bossa nova" → Inject genre preference, planner adapts over next 2-3 tracks
- "Skip this track" → Trigger early transition to next queued track
- "Wind it down" → Adjust arc to descend toward end
- "Someone requested [song name]" → Add to request queue (if in library)

LLM parses these into structured commands:
```python
@dataclass
class DJCommand:
    type: str          # "adjust_energy", "adjust_genre", "skip", "request", "end"
    parameters: dict   # depends on type
```

### 4.3 Implementation

- Claude API with a system prompt defining the DJ persona and output schemas
- Conversation history maintained per session
- LLM never sees audio data — only works with text
- Backend validates LLM output before applying to the planner

### 4.4 Deliverables

- `agent_dj/chat/vibe_parser.py` — Onboarding conversation → VibeProfile
- `agent_dj/chat/command_parser.py` — Mid-session chat → DJCommand
- `agent_dj/chat/prompts.py` — System prompts for the LLM
- Test: run onboarding with various descriptions, verify VibeProfile output

---

## Phase 5: Backend + Session System

**Goal:** FastAPI backend that ties everything together — manages sessions, runs the DJ brain, serves audio, handles WebSocket communication.

### 5.1 API Endpoints

**REST:**
- `POST /session` — Create a new DJ session (returns session ID + host token)
- `GET /session/{id}` — Get session state (current track, queue, vibe profile)
- `POST /session/{id}/library` — Upload audio files for the session
- `GET /audio/{track_id}` — Serve audio file to the browser for Web Audio playback
- `GET /session/{id}/qr` — Generate QR code image for guest invite link
- `POST /session/{id}/join` — Guest joins session (returns guest token + permission level)

**WebSocket:** `ws /session/{id}/ws?token=xxx`

Messages from server:
- `now_playing` — Current track info + position
- `queue_update` — Next 2-3 tracks
- `transition_plan` — Upcoming transition details (type, timing, parameters)
- `load_track` — Tell client to pre-fetch an audio file
- `execute_transition` — Trigger transition at specific beat
- `vibe_update` — Current vibe profile changed
- `chat_message` — DJ response in chat

Messages from client:
- `chat` — Host/guest message to the DJ
- `request_song` — Guest song request
- `vote` — Upvote/downvote a request
- `skip` — Host requests skip
- `playback_state` — Client reports current playback position (for sync)

### 5.2 Session Manager

Each session holds:
- Session ID, creation time, state (onboarding / playing / paused / ended)
- Host token + guest tokens with permission levels
- VibeProfile (evolves over time)
- Track library (analyzed tracks for this session)
- Current SetPlan (rolling, 2-3 tracks ahead)
- Playback state (which track, position, which deck)
- Request queue (guest requests with votes)
- Chat history

### 5.3 DJ Brain Loop

The core loop running per session while state = playing:

```
while session.state == "playing":
    1. Check if current track is approaching mix-out point
    2. If queue is short (< 2 tracks ahead):
       a. Run track selector with current VibeProfile + position in arc
       b. Plan transition from last queued track to new track
       c. Push queue_update + transition_plan to client via WebSocket
    3. Check for new commands (chat, requests, skips):
       a. If vibe change → update VibeProfile, re-plan queue
       b. If song request → score it, insert if compatible, notify
       c. If skip → trigger early transition
    4. Push load_track for next track so client can pre-fetch
    5. At transition time → push execute_transition
    6. Sleep briefly, repeat
```

### 5.4 Guest Permission System

Host sets a default permission level for the session. Individual guests can be promoted/demoted.

| Level | See queue | Request songs | Vote | Change vibe | Skip | Manage guests |
|---|---|---|---|---|---|---|
| Listen | Yes | No | No | No | No | No |
| Request | Yes | Yes | No | No | No | No |
| Vote | Yes | Yes | Yes | No | No | No |
| Co-host | Yes | Yes | Yes | Yes | Yes | Yes |

### 5.5 Deliverables

- `agent_dj/api/app.py` — FastAPI app setup, CORS, static files
- `agent_dj/api/routes.py` — REST endpoints
- `agent_dj/api/websocket.py` — WebSocket handler
- `agent_dj/api/session.py` — Session manager + state
- `agent_dj/api/dj_loop.py` — The DJ brain loop (async)
- `agent_dj/api/auth.py` — Simple token-based auth for host/guests

---

## Phase 6: Frontend (Web UI)

**Goal:** Two views — host dashboard and guest mobile view. Minimal but polished.

### 6.1 Host View

```
┌──────────────────────────────────────────────┐
│  Agent DJ                        [QR Code]   │
├──────────────┬───────────────────────────────┤
│              │                               │
│   Chat       │   Now Playing                 │
│              │   ▶ Track Name - Artist        │
│  DJ: Hey!    │   ████████░░ 2:34 / 4:12      │
│  What's the  │                               │
│  party       │   Up Next                     │
│  tonight?    │   1. Track B                  │
│              │   2. Track C                  │
│  You: House  │                               │
│  party, rnb  │   Energy Arc                  │
│  and jazz    │   ▁▂▃▅▇█▇▅▃▁                  │
│              │        ↑ you are here          │
│  [input]     │                               │
│              │   Guest Requests              │
│              │   ♪ Song X  ▲3               │
│              │   ♪ Song Y  ▲1               │
└──────────────┴───────────────────────────────┘
```

### 6.2 Guest View (Mobile-First)

```
┌─────────────────────┐
│  Agent DJ 🎧         │
│                     │
│  Now Playing        │
│  Track Name         │
│  Artist             │
│                     │
│  ┌───────────────┐  │
│  │ Request Song  │  │
│  └───────────────┘  │
│                     │
│  ┌───────────────┐  │
│  │ More Energy ▲ │  │
│  │ Chill Out   ▼ │  │
│  └───────────────┘  │
│                     │
│  Queue              │
│  1. Track B         │
│  2. Track C         │
│                     │
│  Requests           │
│  ♪ Song X  [▲]     │
│  ♪ Song Y  [▲]     │
└─────────────────────┘
```

### 6.3 Tech

- **React** — Needed for managing WebSocket state, two-deck audio state, and real-time UI updates
- Vite for build tooling
- Tailwind or minimal CSS — keep it clean, not overdesigned

### 6.4 Deliverables

- `frontend/` — React app (Vite)
- `frontend/src/pages/HostView.tsx` — Host dashboard
- `frontend/src/pages/GuestView.tsx` — Guest mobile view
- `frontend/src/audio/` — Web Audio engine (from Phase 3)
- `frontend/src/hooks/useWebSocket.ts` — WebSocket client hook
- `frontend/src/components/` — Chat, Queue, EnergyArc, QRCode, NowPlaying

---

## Phase 7: Integration + End-to-End Testing

**Goal:** Wire everything together and test the full flow.

### 7.1 End-to-End Flow Test

1. Start the backend server
2. Open host view in browser
3. Chat with the DJ to set up the vibe
4. DJ starts playing through the browser
5. Open guest view on phone (via QR code / local network)
6. Guest requests a song
7. DJ adapts the set
8. Transitions play smoothly between tracks
9. Host sends "wind it down" → DJ gradually ends the set

### 7.2 Quality Checks

- [ ] BPM detection accuracy (compare against manual BPM for 10+ tracks)
- [ ] Key detection accuracy (compare against known key for 10+ tracks)
- [ ] Transition quality (no audible glitches, beat-aligned, energy-appropriate)
- [ ] Latency: time from command → DJ response < 2 seconds
- [ ] Track pre-loading: next track loaded before transition starts
- [ ] Guest join flow works on mobile browsers
- [ ] WebSocket reconnection if connection drops

### 7.3 Deliverables

- Integration test scripts
- Demo recording (screen capture of full session)
- Bug fixes from testing

---

## Phase 8: Deployment

**Goal:** Make it hostable so someone can actually use it at a party.

### 8.1 Local Network Deployment (Primary — MVP)

The most realistic party scenario: host runs it on their laptop, everyone connects via local WiFi.

- Package as a single `docker-compose` setup:
  - Backend container (FastAPI + DJ brain + audio files)
  - Frontend served by the backend (static files)
- Host runs: `docker-compose up`
- Host opens `localhost:8000` for the dashboard
- QR code points to `http://<host-local-ip>:8000/guest?session=xxx`
- Guests connect via WiFi to the same network
- Audio plays from the host's browser → connected to speakers via Bluetooth/aux

### 8.2 Cloud Deployment (V2 — Broader Access)

For remote sessions or if we want to offer it as a service:

- Deploy backend to a VPS or cloud service (Railway, Fly.io, etc.)
- Audio files uploaded and stored server-side
- Users access via public URL
- Considerations: audio file storage costs, bandwidth, latency
- Could use a CDN for audio file serving

### 8.3 Packaging

- `Dockerfile` for the backend
- `docker-compose.yml` for one-command local setup
- `.env.example` with required config (Claude API key, audio directory)
- Setup script that: installs system deps, analyzes tracks, starts server

### 8.4 Deliverables

- `Dockerfile`
- `docker-compose.yml`
- `.env.example`
- Setup/run documentation in README.md
- Local network discovery (auto-detect host IP for QR code)

---

## Implementation Order (Recommended)

| Step | Phase | What | Why This Order |
|---|---|---|---|
| 1 | Phase 1 | Track Analyzer + ML classifiers | Foundation — features, structure, genre, embeddings |
| 2 | Phase 1.5 | Model evaluation + fine-tuning | Validate ML accuracy before building on top of it |
| 3 | Phase 2 | Track Selection experiments | Core IP — prove the DJ brain works (notebooks) |
| 4 | Phase 3 | Web Audio mix engine | Prove transitions sound good in the browser |
| 5 | Phase 4 | Chat + vibe parser | Connect LLM to the planning pipeline |
| 6 | Phase 5 | Backend + sessions | Wire the brain to the client |
| 7 | Phase 6 | Frontend UI (React) | Make it usable |
| 8 | Phase 7 | Integration testing | End-to-end validation |
| 9 | Phase 8 | Deployment | Make it runnable at an actual party |

**Milestone checkpoints:**
- After Phase 1: "Can we accurately extract features and classify tracks?" (CLI + inspection)
- After Phase 1.5: "Are genre/mood predictions reliable? Do embeddings cluster well?" (notebooks)
- After Phase 2: "Can the brain select and order tracks intelligently?" (CLI demo)
- After Phase 3: "Do transitions sound like a real DJ?" (browser test page)
- After Phase 5+6: "Can I run a full session?" (end-to-end demo)
- After Phase 8: "Can I use this at a party this weekend?" (deployment)

---

## Phase 9: From Playlist to DJ — Short-Term Upgrades

**Goal:** Transform Agent DJ from a "smart playlist with fades" into something that feels like a real DJ performing. Four key upgrades that are all buildable now.

---

### 9.1 Rolling Planner + Auto-Fetch

**The Problem:** Currently we plan the entire set upfront. A real DJ plans 1-2 tracks ahead and adapts constantly.

**The Solution:** Replace the full-set planner with a rolling planner that decides the next track while the current one is playing, and auto-fetches from YouTube when the local library doesn't have a good match.

#### Backend: Rolling DJ Loop

Replace the current "plan full set → send to client" flow with an async loop:

```python
async def dj_loop(session):
    while session.state == "playing":
        current = get_current_track(session)
        position = get_playback_position(session)  # from client heartbeat

        # If we're approaching the mix-out point and don't have a next track queued
        if needs_next_track(current, position, session.next_track):
            # 1. Score all library tracks against current vibe + energy arc position
            candidates = score_candidates(session.vibe_profile, session.track_store,
                                          current_track=current,
                                          played=session.played_tracks,
                                          position_in_set=session.set_position)

            # 2. If best candidate score is too low, auto-fetch from YouTube
            if candidates[0].score < QUALITY_THRESHOLD:
                search_query = generate_search_query(session.vibe_profile, current)
                new_track = await fetch_and_analyze(search_query)
                if new_track:
                    candidates.insert(0, new_track)

            # 3. Plan transition and send to client
            next_track = candidates[0].track
            transition = plan_transition(current, next_track, energy_direction)

            session.next_track = next_track
            session.played_tracks.append(current)

            await ws_mgr.send_to_session(session.id, {
                "type": "queue_next",
                "track": next_track.to_dict(),
                "transition": transition.to_dict(),
                "audio_url": get_audio_url(next_track),
            })

        # 4. Check for vibe changes from chat/guests
        if session.vibe_changed:
            # Re-score — the next planned track might no longer fit
            session.next_track = None  # force re-plan
            session.vibe_changed = False

        await asyncio.sleep(1)
```

#### Auto-Fetch Search Query Generation

When the library doesn't have a good match, generate a YouTube search query:

```python
def generate_search_query(vibe, current_track):
    """Generate a search query for YouTube based on what the DJ needs next."""
    # Use genre + mood + BPM context
    genres = [g for g, w in sorted(vibe.genres.items(), key=lambda x: -x[1])[:2]]

    # If we have a current track, find something in the same vein
    if current_track.classification:
        top_genre = current_track.classification.genres[0][0]
        return f"{top_genre} {' '.join(genres)} similar to {current_track.title}"

    return f"{' '.join(genres)} music"
```

#### Frontend: Receive Rolling Updates

Instead of receiving a full set plan, the client receives `queue_next` messages with one track at a time:

```typescript
case "queue_next": {
    const track = msg.track as TrackInfo;
    const transition = msg.transition as TransitionInfo;
    // Load into standby deck and set up transition trigger
    await playbackController.queueNext(track, transition);
    setQueue(prev => [...prev, track]);
    break;
}
```

#### Deliverables
- `agent_dj/api/dj_loop.py` — Async DJ brain loop
- `agent_dj/sources/auto_fetch.py` — Search query generation + on-demand YouTube fetch
- Update `api/app.py` — Start DJ loop when session enters playing state
- Update frontend `HostView.tsx` — Handle rolling `queue_next` messages
- Update `PlaybackController.ts` — Support receiving individual track + transition pairs

---

### 9.2 Structure-Aware Transitions

**The Problem:** Current transitions are just timed volume crossfades. A real DJ uses the song structure — looping the outro, layering the intro, swapping bass at the right moment.

**The Solution:** Use the structure detection and mix points we already extract to execute multi-phase transitions.

#### Transition Phases

A real DJ transition has phases, not just a fade:

```
Track A playing normally
    ↓
Phase 1: CUE POINT — Track A reaches its mix-out point (outro/breakdown)
    ↓
Phase 2: INTRO LAYER — Start Track B's intro quietly underneath
    ↓
Phase 3: EQ BLEND — Over 16-32 beats:
    - Cut Track A's bass (highpass filter ramp)
    - Bring in Track B's bass
    - Crossfade mids/highs
    ↓
Phase 4: DROP — Track B is now dominant
    - Kill Track A completely or fade its reverb tail
    ↓
Track B playing normally
```

#### Implementation: TransitionExecutor v2

Rewrite transition execution to be multi-phase:

```typescript
interface TransitionPhase {
    startBeat: number;       // relative to transition start
    duration_beats: number;
    actions: PhaseAction[];
}

interface PhaseAction {
    target: "deckA" | "deckB";
    type: "gain" | "eq_low" | "eq_mid" | "eq_high" | "filter" | "loop_start" | "loop_end";
    from: number;
    to: number;
    curve: "linear" | "exponential" | "step";
}
```

**Transition recipes** (pre-built multi-phase plans):

| Recipe | Phases | Best For |
|---|---|---|
| **Bass Swap** | Layer intro → Cut A bass → Bring B bass → Fade A out | Same BPM, compatible keys |
| **Loop & Drop** | Loop A's outro 4 bars → Layer B's intro → Kill A on B's drop | Any BPM, strong drop in B |
| **Filter Sweep** | Low-pass filter A → Layer B underneath → Open B's filter → Close A's filter | Electronic, house |
| **Echo Release** | Echo/reverb A → Fade A with tail → Clean B entry | Genre changes, energy drops |
| **Breakdown Bridge** | A plays into breakdown → B enters during breakdown → Build together → Drop B | Tracks with clear breakdowns |

#### Beat-Aligned Execution

All phase transitions snap to the beat grid:

```typescript
class BeatAlignedScheduler {
    // Given a beat grid and current position, schedule an action on the next downbeat
    scheduleOnNextDownbeat(beatGrid: number[], currentTime: number, action: () => void) {
        const nextDownbeat = this.findNextDownbeat(beatGrid, currentTime);
        const delay = nextDownbeat - currentTime;
        setTimeout(action, delay * 1000);
    }

    // Schedule a ramp that starts and ends on beat boundaries
    scheduleRamp(param: AudioParam, from: number, to: number,
                 startBeat: number, endBeat: number, beatGrid: number[]) {
        const startTime = beatGrid[startBeat];
        const endTime = beatGrid[endBeat];
        param.setValueAtTime(from, startTime);
        param.linearRampToValueAtTime(to, endTime);
    }
}
```

#### Deliverables
- `frontend/src/audio/TransitionRecipes.ts` — Pre-built multi-phase transition recipes
- `frontend/src/audio/BeatScheduler.ts` — Beat-grid-aligned scheduling
- Update `TransitionExecutor.ts` — Execute multi-phase recipes
- Update `transition_planner.py` — Choose recipe based on track analysis + energy context
- Backend sends detailed phase plan, not just "crossfade for 8 seconds"

---

### 9.3 Auto-Expand Library (On-Demand Fetch)

**The Problem:** The DJ only has access to pre-downloaded tracks. A real DJ has thousands of records and can pull anything.

**The Solution:** When the rolling planner can't find a good match in the local library, it searches YouTube, downloads, analyzes, and queues the track — all while the current track is still playing.

#### Pipeline

```
Rolling planner needs a track
    ↓
Score local library → best score < threshold?
    ↓ yes
Generate search query from vibe + current context
    ↓
YouTube search → pick best result (filter by duration, channel quality)
    ↓
Download audio (yt-dlp, ~5-10 seconds)
    ↓
Quick analysis (BPM + key only, skip ML classification for speed)
    ↓
Verify BPM/key compatibility with current track
    ↓
Full analysis in background (ML classification, structure, mix points)
    ↓
Queue for playback
```

#### Quick Analysis Mode

For on-demand fetching, we need a fast analysis path (~2-3 seconds instead of ~30):

```python
def quick_analyze(file_path: str) -> TrackProfile:
    """Fast analysis: BPM + key only. Enough to decide if we should queue it."""
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)  # only first 30s
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    key, scale, _ = essentia.KeyExtractor()(audio)
    # Return minimal profile — full analysis runs async later
```

#### Smart Search Queries

Don't just search for a genre — be specific:

```python
def generate_search_queries(vibe, current_track, played_tracks):
    """Generate multiple search queries ranked by specificity."""
    queries = []

    # 1. Similar artist
    if current_track.artist:
        queries.append(f"songs similar to {current_track.artist}")

    # 2. Genre + BPM range hint
    top_genre = vibe.genres[0] if vibe.genres else "chill"
    queries.append(f"{top_genre} {current_track.bpm:.0f} BPM")

    # 3. Mood-based
    if vibe.mood_relaxed > 0.7:
        queries.append(f"chill {top_genre} smooth vibes")
    elif vibe.mood_happy > 0.7:
        queries.append(f"upbeat {top_genre} feel good")

    # 4. From LLM example tracks (if provided during onboarding)
    for example in vibe.example_tracks:
        if example not in [t.title for t in played_tracks]:
            queries.append(example)

    return queries
```

#### Fetch Timeout & Fallback

If fetching takes too long, fall back to best available local track:

```python
async def fetch_or_fallback(queries, local_candidates, timeout=15):
    try:
        result = await asyncio.wait_for(
            fetch_and_analyze(queries[0]),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        return local_candidates[0]  # best local match
```

#### Deliverables
- `agent_dj/sources/auto_fetch.py` — Search query generation, fetch pipeline, quick analysis
- `agent_dj/analyzer/audio.py` — Add `quick_analyze()` function
- Update `dj_loop.py` — Integrate auto-fetch with quality threshold
- Configurable: `auto_fetch_enabled`, `quality_threshold`, `fetch_timeout`

---

### 9.4 Energy Monitoring & Adaptation

**The Problem:** We plan an energy arc but never check if we're following it. The DJ should notice "we're too chill for where we should be in the set" and course-correct.

**The Solution:** Track actual energy vs target energy and adjust track selection accordingly.

#### Energy Tracker

```python
class EnergyTracker:
    def __init__(self, vibe_profile: VibeProfile):
        self.vibe = vibe_profile
        self.played_energies: list[float] = []  # actual energy of each played track
        self.target_energies: list[float] = []  # what the arc says it should be

    def record(self, track: TrackProfile, position_in_set: float):
        actual = float(np.mean(track.energy_curve))
        target = self.vibe.get_target_energy(position_in_set)
        self.played_energies.append(actual)
        self.target_energies.append(target)

    @property
    def energy_deficit(self) -> float:
        """Positive = we're behind the arc (too chill). Negative = ahead (too intense)."""
        if not self.played_energies:
            return 0.0
        recent_actual = np.mean(self.played_energies[-3:])  # last 3 tracks
        recent_target = np.mean(self.target_energies[-3:])
        return recent_target - recent_actual

    def adjust_next_target(self, base_target: float) -> float:
        """Adjust the next track's target energy to compensate for drift."""
        deficit = self.energy_deficit
        # If we're behind, boost the target. If ahead, ease off.
        return np.clip(base_target + deficit * 0.5, 0.1, 1.0)
```

#### Integration with Rolling Planner

```python
# In the DJ loop:
energy_tracker.record(current_track, set_position)

# When scoring candidates:
raw_target = vibe.get_target_energy(next_position)
adjusted_target = energy_tracker.adjust_next_target(raw_target)
# Use adjusted_target for scoring, not raw_target
```

#### Guest Energy Feedback

When guests press "More Energy" / "Chill Out":

```python
def apply_energy_feedback(vibe: VibeProfile, direction: str):
    """Shift the remaining energy arc based on crowd feedback."""
    shift = 0.15 if direction == "up" else -0.15
    current_pos = get_current_position()

    # Only shift the remaining arc, not what's already played
    for i in range(len(vibe.energy_arc)):
        pos = i / len(vibe.energy_arc)
        if pos > current_pos:
            vibe.energy_arc[i] = np.clip(vibe.energy_arc[i] + shift, 0.1, 1.0)
```

#### Frontend: Live Energy Display

Update the EnergyArc component to show both target and actual:

```
Target: ▁▂▃▅▇█▇▅▃▁  (planned)
Actual: ▁▁▂▃▅▇      (what's been played)
              ↑ you are here
```

#### Deliverables
- `agent_dj/planner/energy_tracker.py` — Energy monitoring + deficit compensation
- Update `dj_loop.py` — Feed energy tracker, use adjusted targets
- Update `api/app.py` — Apply guest energy feedback to live arc
- Update `frontend/src/components/EnergyArc.tsx` — Show target vs actual curves
- WebSocket message: `energy_update` with current actual vs target

---

### Implementation Order

| Step | What | Why First |
|---|---|---|
| 1 | **9.1 Rolling Planner** | Core architecture change — everything else builds on top |
| 2 | **9.3 Auto-Fetch** | Tightly coupled with rolling planner — DJ needs infinite music |
| 3 | **9.4 Energy Monitoring** | Quick win once rolling planner exists — just add tracking |
| 4 | **9.2 Structure-Aware Transitions** | Most complex, but biggest quality jump — do after the brain works |

**Milestone:** After all four, the test is: "Can I start a session with 0 pre-loaded tracks, describe a vibe, and have the DJ play for 2 hours — adapting to energy feedback and never repeating a track?"

---

## Open Research Questions

- How accurate are Essentia's pre-trained classifiers on our target genres? (Phase 1.5)
- Do we need to fine-tune or is pre-trained good enough? (Phase 1.5)
- Do audio embeddings cluster musically similar tracks? (Phase 1.5)
- What track selection strategy performs best? (Phase 2 experiments)
- How much BPM adjustment is acceptable before it sounds unnatural? (Phase 3 testing)
- What transition type sounds best for which genre/energy combination? (Phase 3 testing)
- How far ahead should the rolling planner look? 2 tracks? 5? (Phase 2 testing)
