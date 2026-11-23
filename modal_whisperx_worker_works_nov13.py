# modal_whisperx_worker.py
import io
import os
import re
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import modal
import numpy as np

# ---------------------------------------------------------------------------
# 1. Modal Container Image & Persistent Model Cache Volume
# ---------------------------------------------------------------------------
MODEL_CACHE_DIR = "/cache"
model_volume = modal.Volume.from_name("sardonicast-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1", "build-essential", "sox", "libsox-fmt-all")
    .run_commands(
        # Step 1: Install CUDA 12.1 PyTorch trilogy (bundles cuDNN 8.9)
        "pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121",
        # Step 2: Lock versions for packages that could break CUDA 12.1, cuDNN 8, or s3prl
        "printf 'torch==2.3.1\\ntorchaudio==2.3.1\\ntorchvision==0.18.1\\nctranslate2==4.4.0\\nfaster-whisper==1.1.0\\ntransformers==4.44.2\\nnumpy<2.0\\n' > /tmp/constraints.txt",
        # Step 3: Pip automatically installs pyannote, s3prl, silero-vad, kaldiio, hdbscan, etc. from whisperx & wespeaker's setup.py
        "pip install -c /tmp/constraints.txt whisperx==3.3.1 git+https://github.com/wenet-e2e/wespeaker.git soundfile scipy onnxruntime-gpu",
    )
    # Step 4 (Appended layer — reuses cached Steps 1-3): Clear executable stack flag on libctranslate2
    .apt_install("patchelf")
    .run_commands(
        "find /usr/local/lib/python3.11/site-packages -name 'libctranslate2*.so*' -exec patchelf --clear-execstack {} +"
    )
    .env({
        "HF_HOME": f"{MODEL_CACHE_DIR}/huggingface",
        "TORCH_HOME": f"{MODEL_CACHE_DIR}/torch",
        "LD_LIBRARY_PATH": (
            "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:"
            "/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:"
            "/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
        ),
    })
)

app = modal.App("sardonicast-whisperx-diarizer", image=image)

# Place this right below `app = modal.App("sardonicast-whisperx-diarizer", image=image)`
with image.imports():
    import torch  # type: ignore
    import whisperx  # type: ignore
    import wespeaker  # type: ignore
    import soundfile as sf  # type: ignore
    try:
        from whisperx.diarize import DiarizationPipeline  # type: ignore
    except ImportError:
        from whisperx import DiarizationPipeline  # type: ignore

# ---------------------------------------------------------------------------
# 2. Metadata Prompt Builder & Host Prior Detector
# ---------------------------------------------------------------------------
HOST_CHANNEL_MARKERS = {
    #"Adam": ["yourmoviesucks", "yms", "2gay2lift"],
    "Adum": ["yourmoviesucks", "yms", "2gay2lift"],
    "Alex": ["iheofficial", "i hate everything", "ihe_official"],
    "Ralph": ["ralphthemoviemaker", "ralph sepe"],
    "Jake": ["mancarryingthing", "womancarryingman", "mancarrying"],
}

def build_initial_prompt_and_priors(meta: Dict[str, Any]) -> Dict[str, Any]:
    title = meta.get("title", "Sardonicast")
    desc = meta.get("description", "")
    desc_lower = desc.lower()

    # Detect which permanent hosts are listed in the episode description
    expected_hosts = [
        host for host, markers in HOST_CHANNEL_MARKERS.items()
        if any(m in desc_lower for m in markers)
    ]
    if not expected_hosts:
        expected_hosts = ["Adum", "Alex", "Ralph", "Jake"]

    # Extract episode topics / guests after the colon in "Sardonicast 186: Adolescence, Unforgiven"
    topics_part = title.split(":", 1)[1].strip() if ":" in title else title

    initial_prompt = (
        "Sardonicast film podcast hosted by Adum Johnston (YMS, YourMovieSucks), "
        "Alex (I Hate Everything, IHE), Ralph Sepe (Ralphthemoviemaker), and "
        "Jake (ManCarryingThing, WomanCarryingMan). "
        f"Discussing {topics_part}. Includes r/Sardonicast subreddit Q&A."
    )
    return {
        "initial_prompt": initial_prompt[:850],
        "expected_hosts": expected_hosts,
        "topics_part": topics_part,
    }

# ---------------------------------------------------------------------------
# 3. Serverless GPU Worker Class (NVIDIA L4 GPU)
# ---------------------------------------------------------------------------
@app.cls(
    gpu="L4",
    timeout=1800,  # 30-min max timeout (handles 4-7 hr milestone livestreams)
    volumes={MODEL_CACHE_DIR: model_volume},
)
class SardonicastDiarizer:
    @modal.enter()
    def load_models(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing models on {self.device}...")

        # 1. WhisperX (faster-whisper large-v3-turbo)
        self.asr_model = whisperx.load_model(
            "large-v3-turbo",
            device=self.device,
            compute_type="float16",
            language="en",
            asr_options={"initial_prompt": "Sardonicast film podcast."},
            download_root=f"{MODEL_CACHE_DIR}/whisperx",
        )

        # 2. Wav2Vec2 Phoneme Forced Alignment Model
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en",
            device=self.device,
            model_dir=f"{MODEL_CACHE_DIR}/wav2vec2",
        )

        # 3. WeSpeaker 256-d Embedding Model (keep on CPU so feats & weights match as torch.FloatTensor)
        self.wespeaker_model = wespeaker.load_model("english")
        if hasattr(self.wespeaker_model, "set_device"):
            self.wespeaker_model.set_device("cpu")

        model_volume.commit()
        print("✅ Core ASR, Alignment & WeSpeaker models ready.")

    def _match_clusters_to_hosts(
        self,
        diarize_df,
        audio_np: np.ndarray,
        host_centroids: Dict[str, List[float]],
        expected_hosts: List[str],
        sr: int = 16000,
        sim_threshold: float = 0.65,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:

        # Normalize saved host centroids
        norm_centroids = {}
        for h_name, vec in host_centroids.items():
            arr = np.asarray(vec, dtype=np.float32).flatten()
            norm_centroids[h_name] = arr / (np.linalg.norm(arr) + 1e-12)

        df = diarize_df.copy()
        df["duration"] = df["end"] - df["start"]
        total_ep_speech = float(df["duration"].sum()) or 1.0

        speaker_map: Dict[str, str] = {}
        diagnostics: Dict[str, Any] = {}

        for spk_id, group in df.groupby("speaker"):
            spk_total_sec = float(group["duration"].sum())

            # Filter for clean solo turns (between 4s and 20s) to extract embeddings
            clean_turns = group[(group["duration"] >= 4.0) & (group["duration"] <= 20.0)]
            if clean_turns.empty:
                clean_turns = group[group["duration"] >= 1.5]
            top_turns = clean_turns.sort_values("duration", ascending=False).head(8)

            embeddings = []
            with tempfile.TemporaryDirectory() as tmpdir:
                for idx, row in top_turns.iterrows():
                    s_sample = max(0, int(float(row["start"]) * sr))
                    e_sample = min(len(audio_np), int(float(row["end"]) * sr))
                    if (e_sample - s_sample) < sr:  # Skip < 1s
                        continue
                    tmp_wav = os.path.join(tmpdir, f"turn_{idx}.wav")
                    sf.write(tmp_wav, audio_np[s_sample:e_sample], sr)
                    emb = self.wespeaker_model.extract_embedding(tmp_wav)
                    if emb is not None:
                        raw_emb = emb.detach().cpu().numpy() if hasattr(emb, "detach") else np.asarray(emb)
                        v = raw_emb.flatten().astype(np.float32)
                        v = v / (np.linalg.norm(v) + 1e-12)
                        embeddings.append(v)

            if not embeddings:
                speaker_map[spk_id] = "Clip / Unknown"
                diagnostics[spk_id] = {"assigned": "Clip / Unknown", "total_sec": round(spk_total_sec, 1), "scores": {}}
                continue

            # Compute L2-normalized cluster centroid
            cluster_centroid = np.mean(np.stack(embeddings, axis=0), axis=0)
            cluster_centroid = cluster_centroid / (np.linalg.norm(cluster_centroid) + 1e-12)

            # Cosine similarity against each host centroid (+0.02 tie-breaker prior if host is in description)
            raw_scores = {
                h: float(np.dot(cluster_centroid, h_vec))
                for h, h_vec in norm_centroids.items()
            }
            prior_scores = {
                h: score + (0.02 if h in expected_hosts else 0.0)
                for h, score in raw_scores.items()
            }
            best_host = max(prior_scores, key=prior_scores.get)
            best_raw_sim = raw_scores[best_host]

            if best_raw_sim >= sim_threshold:
                assigned_label = best_host
            elif spk_total_sec >= max(90.0, total_ep_speech * 0.03):
                assigned_label = "Guest"
            else:
                assigned_label = "Clip / Third-Party"

            speaker_map[spk_id] = assigned_label
            diagnostics[spk_id] = {
                "assigned": assigned_label,
                "best_host_candidate": best_host,
                "best_cosine_sim": round(best_raw_sim, 4),
                "total_sec": round(spk_total_sec, 1),
                "scores": {k: round(v, 4) for k, v in raw_scores.items()},
            }

        return speaker_map, diagnostics

    @modal.method()
    def process_episode(
        self,
        video_id: str,
        audio_bytes: bytes,
        meta: Dict[str, Any],
        host_centroids: Dict[str, List[float]],
        hf_token: str,
        batch_size: int = 16,
    ) -> Dict[str, Any]:

        t0 = time.time()
        prompt_info = build_initial_prompt_and_priors(meta)
        initial_prompt = prompt_info["initial_prompt"]
        expected_hosts = prompt_info["expected_hosts"]

        print(f"🎙️ [{video_id}] Initial Prompt: {initial_prompt}")
        print(f"👥 [{video_id}] Expected Hosts from Metadata: {expected_hosts}")

        # Write uploaded audio bytes to temp file & load 16kHz waveform
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_f:
            tmp_f.write(audio_bytes)
            tmp_audio_path = tmp_f.name

        try:
            audio_np = whisperx.load_audio(tmp_audio_path)
        finally:
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)

        # Step 1: WhisperX ASR with dynamic episode initial_prompt
        from dataclasses import replace, is_dataclass
        if hasattr(self.asr_model.options, "_replace"):
            self.asr_model.options = self.asr_model.options._replace(initial_prompt=initial_prompt)
        elif is_dataclass(self.asr_model.options):
            self.asr_model.options = replace(self.asr_model.options, initial_prompt=initial_prompt)
        asr_result = self.asr_model.transcribe(audio_np, batch_size=batch_size, language="en")

        # Step 2: Wav2Vec2 Phoneme-Level Forced Alignment (splits into exact sentences)
        t1 = time.time()
        aligned_result = whisperx.align(
            asr_result["segments"],
            self.align_model,
            self.align_metadata,
            audio_np,
            self.device,
            return_char_alignments=False,
        )
        print(f"  ├─ Step 2 (Wav2Vec2 Alignment) complete in {time.time() - t1:.1f}s")

        # Step 3: Pyannote 3.1 Speaker Diarization (sliced into 45m windows if > 3.5 hrs)
        t2 = time.time()
        diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=self.device)
        diarize_df = diarize_model(audio_np, min_speakers=2, max_speakers=6)
        print(f"  ├─ Step 3 (Pyannote 3.1 Diarization) complete in {time.time() - t2:.1f}s")

        # Step 4: Match anonymous SPEAKER_XX clusters to hosts via WeSpeaker Centroids
        t3 = time.time()
        speaker_map, speaker_diagnostics = self._match_clusters_to_hosts(
            diarize_df, audio_np, host_centroids, expected_hosts
        )
        
        # Let WhisperX assign the raw SPEAKER_XX labels to words BEFORE we overwrite them
        final_result = whisperx.assign_word_speakers(diarize_df, aligned_result)
        print(f"  ├─ Step 4 (WeSpeaker Centroid Matching) complete in {time.time() - t3:.1f}s")
        print(f"  └─ Cluster Mapping: { json.dumps(speaker_diagnostics, indent=2) }")

        # Format output
        snippets = []
        for seg in final_result.get("segments", []):
            # Extract raw cluster (e.g. SPEAKER_02) and map to host (e.g. Adum)
            raw_cluster = seg.get("speaker", "Unknown")
            assigned_host = speaker_map.get(raw_cluster, raw_cluster)

            start_s = round(float(seg.get("start", 0.0)), 2)
            end_s = round(float(seg.get("end", start_s)), 2)
            dur_s = round(max(0.0, end_s - start_s), 2)
            h, rem = divmod(int(start_s), 3600)
            m, s = divmod(rem, 60)

            snippets.append({
                "speaker": assigned_host,
                "speaker_cluster": raw_cluster,
                "text": seg.get("text", "").strip(),
                "start_seconds": start_s,
                "end_seconds": end_s,
                "duration": dur_s,
                "timestamp": f"{h:02d}:{m:02d}:{s:02d}",
                "youtube_link": f"https://youtu.be/{video_id}?t={int(start_s)}s",
                "words": [
                    {
                        "word": w.get("word", ""),
                        "start": round(float(w["start"]), 2) if "start" in w else start_s,
                        "end": round(float(w["end"]), 2) if "end" in w else end_s,
                        "speaker": speaker_map.get(w.get("speaker", raw_cluster), w.get("speaker", raw_cluster)),
                    }
                    for w in seg.get("words", [])
                    if "word" in w
                ],
            })

        return {
            "meta": meta,
            "speaker_diagnostics": speaker_diagnostics,
            "snippets": snippets,
        }

# ---------------------------------------------------------------------------
# 4. Local Helper Functions & Entrypoint (Runs on Your Windows PC)
# ---------------------------------------------------------------------------
def load_local_centroids(centroid_path: str) -> Dict[str, List[float]]:
    path = Path(centroid_path)
    if not path.exists():
        raise FileNotFoundError(f"Centroid file not found: {centroid_path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Handle either {"Adum": [...]} or {"centroids": {"Adum": [...]}}
        raw = data.get("centroids", data)
        return {k.capitalize(): list(v) for k, v in raw.items() if isinstance(v, (list, np.ndarray))}

    elif path.suffix.lower() == ".npz":
        npz = np.load(path)
        return {k.capitalize(): npz[k].flatten().tolist() for k in npz.files}

    raise ValueError("Centroid file must be .json or .npz")

def get_compressed_flac_bytes(video_id: str, audio_dir: str = "audio_cache") -> bytes:
    """Downsamples audio_cache/{video_id}.wav to 16kHz mono 32k Opus (~22MB/2hrs) via ffmpeg in memory."""
    import subprocess
    import shutil

    wav_path = Path(audio_dir) / f"{video_id}.wav"
    if not wav_path.exists():
        raise FileNotFoundError(f"Local WAV not found at {wav_path}.")

    # Use local bin/ffmpeg.exe if present, otherwise system ffmpeg
    ffmpeg_bin = "bin/ffmpeg.exe" if Path("bin/ffmpeg.exe").exists() else (shutil.which("ffmpeg") or "ffmpeg")

    print(f"📦 Compressing {wav_path} to 16kHz mono Opus for fast upload...")
    cmd = [
        ffmpeg_bin, "-y", "-i", str(wav_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", "64k",
        "-f", "ogg", "pipe:1"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    audio_bytes = proc.stdout
    print(f"  └─ Compressed payload size: {len(audio_bytes) / (1024 * 1024):.1f} MB")
    return audio_bytes

@app.local_entrypoint()
def main(
    video_ids: str = "abYG_qkqhHk,YO9wkuvTsJE,6LHCYUmb-xc,rM6qPSeEvcA",
    centroids_file: str = "speaker_centroids.json",
    hf_token: str = "",
    batch_size: int = 16,
    overwrite: bool = False,
):
    import json
    import os
    from pathlib import Path

    # 1. Resolve HF Token (CLI flag -> environment variable -> .env file)
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not token:
        raise ValueError("HF_TOKEN not found. Set $env:HF_TOKEN or pass --hf-token.")

    # 2. Load host centroids
    if not Path(centroids_file).exists():
        raise FileNotFoundError(f"Centroids file '{centroids_file}' not found.")
    with open(centroids_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    host_centroids = data.get("centroids", data)
    print(f"✅ Loaded {len(host_centroids)} host centroids: {list(host_centroids.keys())}")

    # 3. Prepare inputs locally for each requested video_id
    os.makedirs("diarized_transcripts", exist_ok=True)
    vid_list = [v.strip() for v in video_ids.split(",") if v.strip()]

    jobs = []
    for vid in vid_list:
        out_path = Path(f"diarized_transcripts/{vid}.json")
        if out_path.exists() and not overwrite:
            print(f"⏭️ Skipping {vid} ({out_path} already exists; pass --overwrite to re-run).")
            continue

        # Load episode metadata from transcript_cache/{vid}.json
        meta_path = Path(f"transcript_cache/{vid}.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                raw_cache = json.load(f)
            meta = raw_cache.get("meta", raw_cache)
        else:
            meta = {"video_id": vid, "title": vid, "upload_date": "", "description": ""}

        audio_bytes = get_compressed_flac_bytes(vid)
        jobs.append({
            "video_id": vid,
            "audio_bytes": audio_bytes,
            "meta": meta,
            "host_centroids": host_centroids,
            "hf_token": token,
            "batch_size": batch_size,
        })

    if not jobs:
        print("✅ Nothing to process.")
        return

    # 4. Spawn parallel GPU containers on Modal
    print(f"\n🚀 Spawning {len(jobs)} Modal GPU container(s) in parallel...")
    diarizer = SardonicastDiarizer()
    spawned = [(job["video_id"], diarizer.process_episode.spawn(**job)) for job in jobs]

    # 5. Collect results with per-episode error isolation
    for vid, future in spawned:
        try:
            result = future.get()
            out_path = Path(f"diarized_transcripts/{vid}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"🎉 Saved diarized transcript: {out_path}")
        except Exception as e:
            print(f"❌ Failed on {vid}: {e}")