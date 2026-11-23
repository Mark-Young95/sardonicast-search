import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import wespeaker

def compute_l2_normalized_centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average snippet vectors and L2-normalize the resulting centroid."""
    stack = np.vstack(embeddings)  # Shape: (N, 256)
    mean_vec = np.mean(stack, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    return mean_vec

def main():
    # 1. Initialize WeSpeaker pretrained model (e.g. ResNet34 VoxCeleb 256-d)
    model = wespeaker.load_model("english")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.set_device(device)

    # 2. Map audio files to speakers
    # Example directory structure:
    # voice_samples/
    #   ├── adam/
    #   │     ├── sample_01.wav
    #   │     └── sample_02.wav
    #   └── alex/
    #         ├── sample_01.wav
    #         └── sample_02.wav
    audio_dir = Path("audio_cache/voice_samples")
    
    speaker_embeddings = defaultdict(list)

    # Walk through directory: folder name = speaker label
    for wav_file in sorted(audio_dir.rglob("*.wav")):
        speaker_name = wav_file.parent.name
        
        # Or parse from filename if flat: speaker_name = wav_file.stem.split('_')[0]
        
        print(f"Extracting embedding for {speaker_name}: {wav_file.name}")
        emb = model.extract_embedding(str(wav_file))  # Returns 256-d PyTorch tensor or numpy array
        
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
            
        speaker_embeddings[speaker_name].append(emb.flatten())

    # 3. Calculate normalized centroids per speaker
    centroids = {}
    for speaker, embs in speaker_embeddings.items():
        centroid = compute_l2_normalized_centroid(embs)
        centroids[speaker] = centroid
        print(f"Computed centroid for '{speaker}': shape {centroid.shape}")

    # 4. Save to .npz (Recommended: binary, fast, preserves float32 precision)
    npz_path = "speaker_centroids.npz"
    np.savez_compressed(npz_path, **centroids)
    print(f"Saved centroids to {npz_path}")

    # 5. Save to .json (Optional: human-readable)
    json_path = "speaker_centroids.json"
    json_data = {spk: vec.tolist() for spk, vec in centroids.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved centroids to {json_path}")

if __name__ == "__main__":
    main()