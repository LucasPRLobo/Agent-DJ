# Agent DJ

An AI-powered live DJ that runs in the browser. Describe the vibe of your party, and Agent DJ builds and plays a set with real transitions — beatmatched crossfades, bass swaps, loop blends, and more. Guests join via QR code to request songs and influence the vibe in real-time.

## How It Works

1. **Chat with the DJ** — Describe your party (genre, mood, energy, duration)
2. **DJ builds a set** — Tracks are selected using audio embeddings, BPM/key compatibility, and energy arc matching
3. **Live mixing** — Browser-based two-deck audio engine handles transitions in real-time
4. **Guests interact** — Scan the QR code to request songs, vote, and change the vibe

## Architecture

```
Browser (Web Audio API)          Backend (FastAPI)
├── Two-deck audio engine        ├── Track analyzer (librosa, essentia)
├── Transition executor          ├── ML classifier (genre, mood, embeddings)
├── Chat UI                      ├── Set planner (4 strategies)
└── Guest mobile view            ├── Transition planner
                                 ├── Chat/LLM (Claude API)
                                 └── Session manager (WebSocket)
```

## Quick Start

### Prerequisites

- Python 3.11+, Node.js 20+, conda, ffmpeg
- Anthropic API key (for the chat interface)

### Setup

```bash
# Clone
git clone https://github.com/LucasPRLobo/Agent-DJ.git
cd Agent-DJ

# Python environment
conda create -n agent-dj python=3.11 -y
conda activate agent-dj
pip install -e .
pip install fastapi uvicorn websockets python-multipart qrcode[pil]

# Download ML models
agent-dj download-models

# Frontend
cd frontend && npm install && cd ..

# Add your music
cp /path/to/your/music/*.mp3 samples/

# Analyze tracks
agent-dj analyze samples/

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...
```

### Run

```bash
# Terminal 1: Backend
agent-dj serve

# Terminal 2: Frontend
cd frontend && npm run dev
```

Open `http://localhost:5173` — chat with the DJ to start your set.

### Docker

```bash
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
# Put music files in samples/

docker-compose up
```

## CLI Commands

```bash
agent-dj analyze <path>          # Analyze audio files
agent-dj library                 # List analyzed tracks
agent-dj inspect <track>         # Detailed track analysis
agent-dj similar <track>         # Find similar tracks by embedding
agent-dj plan --preset house_party --strategy graph_beam
agent-dj download-models         # Download ML models
agent-dj serve                   # Start the backend server
```

## Track Selection Strategies

| Strategy | Description |
|---|---|
| `greedy_scoring` | Score each track by energy/BPM/key/genre match, pick best greedily |
| `graph_greedy` | Build transition-quality graph, find best path |
| `graph_beam` | Beam search over the transition graph (best quality, slower) |
| `clustering` | Cluster tracks by embedding similarity, navigate clusters along energy arc |

## Transition Types

| Type | When Used |
|---|---|
| Beatmatched crossfade | Default — smooth volume swap aligned to beats |
| Bass swap | Compatible keys, both tracks have energy — EQ transition |
| Loop blend | Outgoing track has good loop candidates |
| Hard cut | Large BPM difference — instant switch on downbeat |
| Echo out | Energy dropping — reverb tail on outgoing, clean entry |
| Breakdown blend | Incoming track starts with low-energy section |

## Tech Stack

- **Audio analysis**: librosa, essentia, pyloudnorm
- **ML classification**: Essentia pre-trained models (Discogs-EffNet, MusiCNN)
- **DJ brain**: Python, scikit-learn, numpy
- **Chat**: Claude API (Anthropic)
- **Backend**: FastAPI, WebSockets
- **Frontend**: React, TypeScript, Vite, Web Audio API
- **Deployment**: Docker
