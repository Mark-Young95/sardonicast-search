import collections
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import wespeaker

HOST_CHANNEL_MARKERS = {
    "adum": ["yourmoviesucks", "yms", "adum", "adam"],
    "alex": ["iheofficial", "i hate everything", "alex"],
    "ralph": ["ralphthemoviemaker", "ralphsepe", "ralph"],
    "jake": ["mancarryingthing", "mancarrying", "womancarryingman", "jake"],
}

def detect_expected_hosts(meta: dict, available_hosts: list) -> list:
    text = f"{meta.get('title', '')}\n{meta.get('description', '')}".lower()
    found = [h for h in available_hosts if any(m in text for m in HOST_CHANNEL_MARKERS.get(h.lower(), [h.lower()]))]
    return found if len(found) >= 2 else available_hosts

def extract_strict_embedding(model, audio_np: np.ndarray, s: float, e: float, sr: int = 16000):
    """Slices strictly within [s, e] (no neighbor bleed) and tiles short clips to 1.2s."""
    dur = e - s
    if dur < 0.35:
        return None
    if dur > 8.0:
        mid = 0.5 * (s + e)
        s, e = mid - 4.0, mid + 4.0

    chunk = audio_np[max(0, int(s * sr)) : min(len(audio_np), int(e * sr))]
    if len(chunk) < int(0.35 * sr):
        return None

    # Tile short utterances (<1.2s) with their own audio instead of bleeding into neighbors
    min_samples = int(1.2 * sr)
    if len(chunk) < min_samples:
        reps = int(np.ceil(min_samples / len(chunk)))
        chunk = np.tile(chunk, reps)[:min_samples]

    pcm_int16 = np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16)
    with torch.no_grad():
        emb = model.extract_embedding_from_pcm(torch.from_numpy(pcm_int16).unsqueeze(0), sr)
    if emb is None:
        return None
    v = emb.detach().cpu().numpy().flatten().astype(np.float32)
    norm = np.linalg.norm(v)
    return (v / norm) if norm > 1e-12 else None

def repair_episode(video_id: str, centroids_file: str = "speaker_centroids.json"):
    json_path = Path(f"diarized_transcripts/{video_id}.json")
    wav_path = Path(f"audio_cache/{video_id}.wav")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(centroids_file, "r", encoding="utf-8") as f:
        c_data = json.load(f)

    raw_centroids = c_data.get("centroids", c_data)
    global_centroids = {
        k: np.asarray(v, dtype=np.float32) / (np.linalg.norm(v) + 1e-12)
        for k, v in raw_centroids.items()
    }
    expected_hosts = detect_expected_hosts(data.get("meta", {}), list(global_centroids.keys()))
    print(f"Expected hosts for {video_id}: {expected_hosts}")

    snippets = data["snippets"]
    print(f"Loading {wav_path}...")
    audio_np, sr = sf.read(str(wav_path), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    model = wespeaker.load_model("english")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device)
    model.model.eval()
    model.model.register_forward_pre_hook(
        lambda mod, inp: (inp[0].to(next(mod.parameters()).device),)
    )

    # 1. Extract strict-boundary embeddings (zero neighbor bleed)
    seg_vecs = [
        extract_strict_embedding(model, audio_np, seg["start_seconds"], seg["end_seconds"], sr)
        for seg in snippets
    ]

    # 2. Build episode-adapted centroids from high-confidence long segments (>= 2.5s)
    confident_vecs = {h: [] for h in expected_hosts}
    for i, seg in enumerate(snippets):
        v = seg_vecs[i]
        if v is not None and seg["duration"] >= 2.5:
            scores = {h: float(np.dot(v, global_centroids[h])) for h in expected_hosts}
            sorted_h = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_h, best_s = sorted_h[0]
            runner_up_s = sorted_h[1][1] if len(sorted_h) > 1 else 0.0
            if best_s >= 0.55 and (best_s - runner_up_s) >= 0.12:
                confident_vecs[best_h].append(v)

    adapted_centroids = {}
    for h in expected_hosts:
        if len(confident_vecs[h]) >= 10:
            ep_mean = np.mean(confident_vecs[h], axis=0)
            ep_mean = ep_mean / (np.linalg.norm(ep_mean) + 1e-12)
            # Blend 65% episode-specific mic profile + 35% global reference profile
            blended = 0.65 * ep_mean + 0.35 * global_centroids[h]
            adapted_centroids[h] = blended / (np.linalg.norm(blended) + 1e-12)
            print(f"  Calibrated '{h}' to episode mic using {len(confident_vecs[h])} clean turns.")
        else:
            adapted_centroids[h] = global_centroids[h]

    # 3. Score every segment without merged-cluster neighbor poisoning
    prev_speaker = expected_hosts[0]
    for i, seg in enumerate(snippets):
        cluster = seg.get("speaker_cluster", "")
        cluster_diag = data.get("speaker_diagnostics", {}).get(cluster, {})
        if cluster_diag.get("best_cosine_sim", 1.0) < 0.25:
            continue

        v = seg_vecs[i]
        if v is not None:
            scores = {h: float(np.dot(v, adapted_centroids[h])) for h in expected_hosts}
            best_h = max(scores, key=scores.get)
            if scores[best_h] >= 0.40:
                seg["speaker"] = best_h
                prev_speaker = best_h
            else:
                seg["speaker"] = prev_speaker
        else:
            seg["speaker"] = prev_speaker

        for w in seg.get("words", []):
            w["speaker"] = seg["speaker"]

    after_counts = collections.Counter(s["speaker"] for s in snippets)
    sec_by_speaker = collections.defaultdict(float)
    for s in snippets:
        sec_by_speaker[s["speaker"]] += s["duration"]

    data["speaker_summary"] = {
        spk: {"segments": after_counts[spk], "minutes": round(sec_by_speaker[spk] / 60.0, 1)}
        for spk in sorted(after_counts.keys())
    }
    print("Updated speaker_summary:", json.dumps(data["speaker_summary"]))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved updated {json_path}")

if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "YO9wkuvTsJE"
    repair_episode(vid)