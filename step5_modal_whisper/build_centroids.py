import re
import warnings
from collections import defaultdict
from pathlib import Path
import json

# Suppress upstream library noise
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*ESPnet is not installed.*")

import numpy as np
import torch
import wespeaker

# Path to the directory containing your <speaker>_sample_*.wav files
SAMPLES_DIR = Path("audio_cache/voice_samples")
OUTPUT_NPZ = Path("speaker_centroids.npz")
OUTPUT_JSON = Path("speaker_centroids.json")


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def compute_centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    """Computes arithmetic mean vector and projects back to unit hypersphere."""
    mean_vec = np.mean(embeddings, axis=0)
    return l2_normalize(mean_vec)


def main():
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading WeSpeaker model (device: {device})...")
    model = wespeaker.load_model("english")
    model.set_device(device)

    # Collect embeddings grouped by speaker name
    speaker_embeddings = defaultdict(list)
    wav_files = sorted(SAMPLES_DIR.glob("*_sample_*.wav"))

    if not wav_files:
        raise FileNotFoundError(f"No matching *_sample_*.wav files found in {SAMPLES_DIR.resolve()}")

    print(f"Found {len(wav_files)} snippet files. Extracting embeddings...")

    for wav in wav_files:
        # Extract speaker name before '_sample_' (e.g., 'alex_sample_02.wav' -> 'alex')
        match = re.match(r"^([a-zA-Z0-9]+)_sample_\d+", wav.stem)
        if not match:
            print(f"  [Skip] Filename does not match expected format: {wav.name}")
            continue
        speaker = match.group(1).lower()

        # Extract 256-d embedding vector
        emb = model.extract_embedding(str(wav))
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()

        emb_flat = l2_normalize(emb.flatten())
        speaker_embeddings[speaker].append(emb_flat)
        print(f"  Processed {speaker} <- {wav.name}")

    # Compute centroids
    centroids = {}
    print("\n--- Centroid Computation ---")
    for speaker, embs in speaker_embeddings.items():
        centroids[speaker] = compute_centroid(embs)
        print(f"Speaker '{speaker}': aggregated {len(embs)} samples into 256-d centroid.")

    # 1. Save binary .npz (fastest and exact precision)
    np.savez_compressed(OUTPUT_NPZ, **centroids)
    print(f"\nSaved compressed binary: {OUTPUT_NPZ.resolve()}")

    # 2. Save JSON (for human inspection/debugging)
    json_payload = {spk: vec.tolist() for spk, vec in centroids.items()}
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Saved readable JSON: {OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    main()