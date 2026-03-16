"""ML classification pipeline using Essentia pre-trained models.

Extracts genre, mood, danceability, voice/instrumental, and deep audio
embeddings from tracks using Essentia's TensorFlow-based models.

Architecture:
  Audio → Discogs-EffNet (feature extractor) → embeddings
                                                  ↓
                                    Classification heads (genre, mood, etc.)

  Audio → MusiCNN (feature extractor) → embeddings (for similarity search)

Models are downloaded on first use from Essentia's model repository.
"""

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Base URL for Essentia model downloads
MODEL_BASE_URL = "https://essentia.upf.edu/models"

# Model definitions: key → (filename, url)
MODELS = {
    # Feature extractors
    "musicnn": {
        "file": "msd-musicnn-1.pb",
        "url": f"{MODEL_BASE_URL}/feature-extractors/musicnn/msd-musicnn-1.pb",
    },
    "discogs_effnet": {
        "file": "discogs-effnet-bs64-1.pb",
        "url": f"{MODEL_BASE_URL}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
    },
    # Classification heads (all use discogs-effnet embeddings)
    "genre_discogs400": {
        "file": "genre_discogs400-discogs-effnet-1.pb",
        "url": f"{MODEL_BASE_URL}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
        "metadata_url": f"{MODEL_BASE_URL}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
    },
    "mood_happy": {
        "file": "mood_happy-discogs-effnet-1.pb",
        "url": f"{MODEL_BASE_URL}/classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb",
    },
    "mood_aggressive": {
        "file": "mood_aggressive-discogs-effnet-1.pb",
        "url": f"{MODEL_BASE_URL}/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb",
    },
    "mood_relaxed": {
        "file": "mood_relaxed-discogs-effnet-1.pb",
        "url": f"{MODEL_BASE_URL}/classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb",
    },
    "mood_sad": {
        "file": "mood_sad-discogs-effnet-1.pb",
        "url": f"{MODEL_BASE_URL}/classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb",
    },
    "danceability": {
        "file": "danceability-discogs-effnet-1.pb",
        "url": f"{MODEL_BASE_URL}/classification-heads/danceability/danceability-discogs-effnet-1.pb",
    },
    "voice_instrumental": {
        "file": "voice_instrumental-discogs-effnet-1.pb",
        "url": f"{MODEL_BASE_URL}/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb",
    },
}

# Default models directory
DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models"


@dataclass
class ClassificationResult:
    """Results from ML classification pipeline."""
    # Genre: top-N genre predictions with confidence
    genres: list[tuple[str, float]] = field(default_factory=list)

    # Mood scores (0-1)
    mood_happy: float = 0.0
    mood_aggressive: float = 0.0
    mood_relaxed: float = 0.0
    mood_sad: float = 0.0

    # Other scores
    danceability: float = 0.0
    voice_instrumental: float = 0.0  # 0 = instrumental, 1 = voice

    # Deep audio embedding (for similarity search)
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "genres": self.genres,
            "mood_happy": self.mood_happy,
            "mood_aggressive": self.mood_aggressive,
            "mood_relaxed": self.mood_relaxed,
            "mood_sad": self.mood_sad,
            "danceability": self.danceability,
            "voice_instrumental": self.voice_instrumental,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClassificationResult":
        data["genres"] = [tuple(g) for g in data.get("genres", [])]
        return cls(**data)


def download_model(model_key: str, model_dir: Path | None = None) -> Path:
    """Download a model file if not already present."""
    model_dir = model_dir or DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    model_info = MODELS[model_key]
    model_path = model_dir / model_info["file"]

    if model_path.exists():
        return model_path

    print(f"  Downloading model: {model_key}...")
    urllib.request.urlretrieve(model_info["url"], str(model_path))
    return model_path


def download_all_models(model_dir: Path | None = None) -> dict[str, Path]:
    """Download all required models."""
    paths = {}
    for key in MODELS:
        paths[key] = download_model(key, model_dir)
    return paths


def _get_genre_labels(model_dir: Path | None = None) -> list[str]:
    """Get the 400 Discogs genre/style labels in model output order."""
    model_dir = model_dir or DEFAULT_MODEL_DIR
    labels_path = model_dir / "genre_discogs400_labels.json"

    if labels_path.exists():
        with open(labels_path) as f:
            data = json.load(f)
            return data if isinstance(data, list) else data.get("classes", [])

    # Download metadata JSON
    try:
        metadata_url = MODELS["genre_discogs400"]["metadata_url"]
        response = urllib.request.urlopen(metadata_url)
        metadata = json.loads(response.read())
        labels = metadata.get("classes", [])
        if labels:
            with open(labels_path, "w") as f:
                json.dump(labels, f)
            return labels
    except Exception as e:
        print(f"  Warning: Could not download genre labels: {e}")

    return [f"genre_{i}" for i in range(400)]


def classify_track(
    audio_path: str | Path,
    model_dir: Path | None = None,
    top_n_genres: int = 10,
) -> ClassificationResult:
    """Run full ML classification pipeline on a track.

    Pipeline:
    1. Load audio at 16kHz
    2. Extract Discogs-EffNet embeddings (shared by all classification heads)
    3. Run genre, mood, danceability, voice/instrumental heads
    4. Extract MusiCNN embeddings (for similarity search)
    """
    from essentia.standard import (
        MonoLoader,
        TensorflowPredict2D,
        TensorflowPredictEffnetDiscogs,
        TensorflowPredictMusiCNN,
    )

    model_dir = model_dir or DEFAULT_MODEL_DIR
    audio_path = str(audio_path)

    # Load audio at 16kHz (required by models)
    audio_16k = MonoLoader(filename=audio_path, sampleRate=16000)()

    result = ClassificationResult()

    # --- Step 1: Discogs-EffNet embeddings (shared input for all heads) ---
    effnet_model = download_model("discogs_effnet", model_dir)
    effnet_embeddings = TensorflowPredictEffnetDiscogs(
        graphFilename=str(effnet_model),
        output="PartitionedCall:1",
    )(audio_16k)

    # --- Step 2: Genre classification ---
    try:
        genre_model = download_model("genre_discogs400", model_dir)
        genre_preds = TensorflowPredict2D(
            graphFilename=str(genre_model),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )(effnet_embeddings)

        mean_preds = np.mean(genre_preds, axis=0)
        genre_labels = _get_genre_labels(model_dir)

        if len(genre_labels) == len(mean_preds):
            top_indices = np.argsort(mean_preds)[::-1][:top_n_genres]
            result.genres = [
                (genre_labels[i], round(float(mean_preds[i]), 4))
                for i in top_indices
            ]
    except Exception as e:
        print(f"  Warning: Genre classification failed: {e}")

    # --- Step 3: Mood, danceability, voice/instrumental (all use effnet embeddings) ---
    head_configs = [
        ("mood_happy", "mood_happy"),
        ("mood_aggressive", "mood_aggressive"),
        ("mood_relaxed", "mood_relaxed"),
        ("mood_sad", "mood_sad"),
        ("danceability", "danceability"),
        ("voice_instrumental", "voice_instrumental"),
    ]

    for model_key, attr_name in head_configs:
        try:
            head_model = download_model(model_key, model_dir)
            preds = TensorflowPredict2D(
                graphFilename=str(head_model),
                input="model/Placeholder",
                output="model/Softmax",
            )(effnet_embeddings)
            # Second column is the "positive" class
            score = float(np.mean(preds[:, 1])) if preds.shape[1] > 1 else float(np.mean(preds))
            setattr(result, attr_name, round(score, 4))
        except Exception as e:
            print(f"  Warning: {model_key} classification failed: {e}")

    # --- Step 4: MusiCNN embeddings (for similarity search) ---
    try:
        musicnn_model = download_model("musicnn", model_dir)
        embeddings = TensorflowPredictMusiCNN(
            graphFilename=str(musicnn_model),
            output="model/dense/BiasAdd",
        )(audio_16k)
        mean_embedding = np.mean(embeddings, axis=0)
        result.embedding = [round(float(x), 6) for x in mean_embedding]
    except Exception as e:
        print(f"  Warning: MusiCNN embedding extraction failed: {e}")

    return result
