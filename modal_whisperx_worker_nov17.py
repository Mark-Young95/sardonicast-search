# modal_whisperx_worker.py
import io
import os
import re
import gc
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
    "adum": ["yourmoviesucks", "yms", "adum", "adam", "2gay2lift"],
    "alex": ["iheofficial", "i hate everything", "alex", "ihe_official"],
    "ralph": ["ralphthemoviemaker", "ralphsepe", "ralph", "ralph sepe"],
    "jake": ["mancarryingthing", "mancarrying", "womancarryingman", "jake"],
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
    timeout=3600,  # 60-min max timeout (handles 4-7 hr milestone livestreams)
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

        # 3. WeSpeaker 256-d Embedding Model (on GPU via forward_pre_hook)
        wespeaker_cache = os.path.join(MODEL_CACHE_DIR, "wespeaker")
        os.makedirs(wespeaker_cache, exist_ok=True)
        default_ws_dir = os.path.expanduser("~/.wespeaker")
        if not os.path.exists(default_ws_dir):
            os.symlink(wespeaker_cache, default_ws_dir)

        self.wespeaker_model = wespeaker.load_model("english")
        self.wespeaker_model.model.to(self.device)
        self.wespeaker_model.model.eval()
        # Automatically move CPU fbank features onto GPU before ResNet forward pass
        self.wespeaker_model.model.register_forward_pre_hook(
            lambda mod, inp: (inp[0].to(next(mod.parameters()).device),)
        )
        model_volume.commit()
        print("✅ Core ASR, Alignment & WeSpeaker models ready.")

    def _extract_embedding_fast(self, chunk_np: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
        """In-memory GPU WeSpeaker extraction matching 16-bit PCM WAV scale."""
        if len(chunk_np) < int(0.8 * sr):
            return None
        pcm_int16 = np.clip(chunk_np * 32768.0, -32768, 32767).astype(np.int16)
        pcm_tensor = torch.from_numpy(pcm_int16).unsqueeze(0)
        with torch.no_grad():
            emb = self.wespeaker_model.extract_embedding_from_pcm(pcm_tensor, sr)
        if emb is None:
            return None
        raw_emb = emb.detach().cpu().numpy() if hasattr(emb, "detach") else np.asarray(emb)
        v = raw_emb.flatten().astype(np.float32)
        norm = np.linalg.norm(v)
        return (v / norm) if norm > 1e-12 else None

    def _split_multispeaker_segments(
        self,
        raw_segments: List[Dict[str, Any]],
        audio_np: np.ndarray,
        norm_centroids: Dict[str, np.ndarray],
        expected_hosts: List[str],
        sr: int = 16000,
    ) -> List[Dict[str, Any]]:
        """
        1) Splits segments whenever Pyannote's word-level speaker (w['speaker']) changes.
        2) Splits remaining long unpunctuated segments (>= 4.5s) at clause/turn boundaries
           when the acoustic voiceprint before vs. after the break switches hosts.
        """
        turn_starters = {
            "yeah", "yep", "mm-hmm", "mhm", "well", "do", "did", "what", "how",
            "why", "wait", "no", "oh", "so", "but", "i", "it", "it's", "that",
            "that's", "right", "honestly", "true", "actually", "like",
        }

        # Pass A: Split on Pyannote word-level speaker transitions
        pass_a_segments: List[Dict[str, Any]] = []
        for seg in raw_segments:
            words = [w for w in seg.get("words", []) if "word" in w and "start" in w and "end" in w]
            if len(words) < 2:
                pass_a_segments.append(seg)
                continue

            default_spk = seg.get("speaker", "Unknown")
            spks = [w.get("speaker", default_spk) for w in words]

            # Smooth 1-word speaker blips (< 0.45s) sandwiched between identical speakers
            for k in range(1, len(words) - 1):
                w_dur = float(words[k]["end"]) - float(words[k]["start"])
                if spks[k - 1] == spks[k + 1] and spks[k] != spks[k - 1] and w_dur < 0.45:
                    spks[k] = spks[k - 1]
                    words[k]["speaker"] = spks[k - 1]

            # Group contiguous runs of words with the same Pyannote speaker
            runs: List[List[Dict[str, Any]]] = [[words[0]]]
            for k in range(1, len(words)):
                if spks[k] == spks[k - 1]:
                    runs[-1].append(words[k])
                else:
                    runs.append([words[k]])

            # Merge tiny 1-word orphan runs (< 0.35s) into adjacent run
            merged_runs: List[List[Dict[str, Any]]] = []
            for r in runs:
                r_dur = float(r[-1]["end"]) - float(r[0]["start"])
                if merged_runs and len(r) == 1 and r_dur < 0.35:
                    merged_runs[-1].extend(r)
                else:
                    merged_runs.append(r)

            for r in merged_runs:
                pass_a_segments.append({
                    "start": float(r[0]["start"]),
                    "end": float(r[-1]["end"]),
                    "text": " ".join(w["word"].strip() for w in r).strip(),
                    "speaker": r[0].get("speaker", default_spk),
                    "words": r,
                })

        # Pass B: Acoustic turn-change split on long segments (>= 4.5s) where Pyannote merged speakers
        final_segments: List[Dict[str, Any]] = []
        for seg in pass_a_segments:
            queue = [seg]
            while queue:
                curr = queue.pop(0)
                words = curr.get("words", [])
                s_start = float(curr.get("start", 0.0))
                s_end = float(curr.get("end", s_start))
                dur = s_end - s_start

                if dur < 4.5 or len(words) < 8:
                    final_segments.append(curr)
                    continue

                # Find candidate word indices where a new speaker turn could start
                candidates = []
                for k in range(3, len(words) - 3):
                    left_dur = float(words[k - 1]["end"]) - s_start
                    right_dur = s_end - float(words[k]["start"])
                    if left_dur < 1.2 or right_dur < 1.2:
                        continue
                    gap = float(words[k]["start"]) - float(words[k - 1]["end"])
                    clean_w = words[k]["word"].lower().strip(".,?!\"'")
                    prev_punct = words[k - 1]["word"].strip().endswith((".", "?", "!"))
                    if gap >= 0.22 or prev_punct or (gap >= 0.06 and clean_w in turn_starters):
                        candidates.append((k, gap))

                split_idx = None
                best_divergence = 0.0

                for k, gap in candidates:
                    t_split = float(words[k]["start"])
                    l_chunk = audio_np[max(0, int(max(s_start, t_split - 2.5) * sr)) : int(t_split * sr)]
                    r_chunk = audio_np[int(t_split * sr) : min(len(audio_np), int(min(s_end, t_split + 2.5) * sr))]
                    v_l = self._extract_embedding_fast(l_chunk, sr)
                    v_r = self._extract_embedding_fast(r_chunk, sr)
                    if v_l is None or v_r is None:
                        continue

                    scores_l = {h: float(np.dot(v_l, norm_centroids[h])) for h in expected_hosts}
                    scores_r = {h: float(np.dot(v_r, norm_centroids[h])) for h in expected_hosts}
                    h_l = max(scores_l, key=scores_l.get)
                    h_r = max(scores_r, key=scores_r.get)
                    self_sim = float(np.dot(v_l, v_r))

                    if h_l != h_r and scores_l[h_l] >= 0.45 and scores_r[h_r] >= 0.45 and self_sim < 0.55:
                        divergence = (scores_l[h_l] - scores_l[h_r]) + (scores_r[h_r] - scores_r[h_l])
                        if divergence > best_divergence:
                            best_divergence = divergence
                            split_idx = k

                if split_idx is not None:
                    left_words = words[:split_idx]
                    right_words = words[split_idx:]
                    final_segments.append({
                        "start": float(left_words[0]["start"]),
                        "end": float(left_words[-1]["end"]),
                        "text": " ".join(w["word"].strip() for w in left_words).strip(),
                        "speaker": curr.get("speaker", "Unknown"),
                        "words": left_words,
                    })
                    # Re-queue right half in case there are multiple speaker changes in a 25s block
                    queue.insert(0, {
                        "start": float(right_words[0]["start"]),
                        "end": float(right_words[-1]["end"]),
                        "text": " ".join(w["word"].strip() for w in right_words).strip(),
                        "speaker": curr.get("speaker", "Unknown"),
                        "words": right_words,
                    })
                else:
                    final_segments.append(curr)

        return final_segments

    def _match_clusters_to_hosts(
        self,
        diarize_df,
        audio_np: np.ndarray,
        host_centroids: Dict[str, List[float]],
        expected_hosts: List[str],
        sr: int = 16000,
        sim_threshold: float = 0.62,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        centroids = {
            k: np.asarray(v, dtype=np.float32) / (np.linalg.norm(v) + 1e-12)
            for k, v in host_centroids.items()
        }
        active_hosts = [h for h in expected_hosts if h in centroids] or list(centroids.keys())

        cluster_turns: Dict[str, List[Tuple[float, float]]] = {}
        cluster_total_dur: Dict[str, float] = {}
        all_rows = list(diarize_df.itertuples())

        for i, row in enumerate(all_rows):
            spk = row.speaker
            s, e = float(row.start), float(row.end)
            dur = max(0.0, e - s)
            cluster_total_dur[spk] = cluster_total_dur.get(spk, 0.0) + dur
            if dur < 1.5:
                continue
            overlaps = False
            for j in range(max(0, i - 3), min(len(all_rows), i + 4)):
                if i != j and max(s, float(all_rows[j].start)) < min(e, float(all_rows[j].end)):
                    overlaps = True
                    break
            if not overlaps:
                cluster_turns.setdefault(spk, []).append((s, e))

        speaker_map: Dict[str, str] = {}
        diagnostics: Dict[str, Any] = {}

        for spk, total_sec in sorted(cluster_total_dur.items()):
            turns = cluster_turns.get(spk, [])
            if len(turns) > 40:
                idx = np.linspace(0, len(turns) - 1, 40, dtype=int)
                sampled_turns = [turns[i] for i in idx]
            else:
                sampled_turns = turns

            embeddings = []
            turn_winner_counts: Dict[str, int] = {h: 0 for h in active_hosts}
            for s, e in sampled_turns:
                mid = 0.5 * (s + e)
                half = min(4.0, 0.5 * (e - s))
                chunk = audio_np[int((mid - half) * sr) : int((mid + half) * sr)]
                v = self._extract_embedding_fast(chunk, sr)
                if v is not None:
                    embeddings.append(v)
                    scores_t = {h: float(np.dot(v, centroids[h])) for h in active_hosts}
                    best_h = max(scores_t, key=scores_t.get)
                    if scores_t[best_h] >= 0.48:
                        turn_winner_counts[best_h] += 1

            if not embeddings:
                speaker_map[spk] = "Unknown"
                diagnostics[spk] = {"assigned": "Unknown", "total_sec": round(total_sec, 1)}
                continue

            mean_vec = np.mean(embeddings, axis=0)
            mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)
            all_scores = {h: round(float(np.dot(mean_vec, c)), 4) for h, c in centroids.items()}
            expected_scores = {h: all_scores[h] for h in active_hosts}
            best_host = max(expected_scores, key=expected_scores.get)
            best_sim = expected_scores[best_host]

            significant_hosts = [
                h for h, cnt in turn_winner_counts.items()
                if cnt >= max(3, int(0.15 * len(embeddings)))
            ]
            is_merged = len(significant_hosts) >= 2

            if best_sim >= sim_threshold:
                assigned = best_host
            elif is_merged and best_sim >= 0.52:
                assigned = best_host
            elif total_sec >= 180.0:
                assigned = "Guest"
            else:
                assigned = "Clip / Third-Party"

            speaker_map[spk] = assigned
            diagnostics[spk] = {
                "assigned": assigned,
                "is_merged_cluster": is_merged,
                "merged_candidates": significant_hosts if is_merged else [assigned],
                "turn_host_distribution": turn_winner_counts,
                "best_host_candidate": best_host,
                "best_cosine_sim": best_sim,
                "total_sec": round(total_sec, 1),
                "scores": all_scores,
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
        asr_result = self.asr_model.transcribe(
            audio_np,
            batch_size=batch_size,
            chunk_size=15,
            language="en"
        )
        print(f"  ├─ Step 1 (WhisperX ASR) complete in {time.time() - t0:.1f}s")

        gc.collect()
        torch.cuda.empty_cache()

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

        gc.collect()
        torch.cuda.empty_cache()

        # Normalize expected_hosts against actual keys in host_centroids
        centroid_key_map = {k.lower(): k for k in host_centroids.keys()}
        expected_hosts = [centroid_key_map[h.lower()] for h in expected_hosts if h.lower() in centroid_key_map]
        if not expected_hosts:
            expected_hosts = list(host_centroids.keys())

        # Step 3: Pyannote 3.1 Speaker Diarization
        t2 = time.time()
        if getattr(self, "diarize_model", None) is None:
            DiarizationPipeline = getattr(whisperx, "DiarizationPipeline", None)
            if DiarizationPipeline is None:
                from whisperx.diarize import DiarizationPipeline  # type: ignore
            self.diarize_model = DiarizationPipeline(
                use_auth_token=hf_token,
                device=self.device,
            )

        min_spk = max(2, len(expected_hosts))
        diarize_df = self.diarize_model(
            audio_np,
            min_speakers=min_spk,
            max_speakers=min_spk + 4,
        )
        print(f"  ├─ Step 3 (Pyannote 3.1 Diarization, min_speakers={min_spk}) complete in {time.time() - t2:.1f}s")

        # Step 4: Hybrid Cluster Matching + Multi-Speaker Segment Splitting
        t3 = time.time()
        speaker_map, speaker_diagnostics = self._match_clusters_to_hosts(
            diarize_df, audio_np, host_centroids, expected_hosts
        )
        final_result = whisperx.assign_word_speakers(diarize_df, aligned_result)

        norm_centroids = {
            k: np.asarray(v, dtype=np.float32) / (np.linalg.norm(v) + 1e-12)
            for k, v in host_centroids.items()
        }
        sr = 16000

        # 4a. Split run-on segments at word-level speaker changes & acoustic turn boundaries
        split_segments = self._split_multispeaker_segments(
            final_result.get("segments", []),
            audio_np,
            norm_centroids,
            expected_hosts,
            sr,
        )

        # 4b. Assign speakers using Hybrid Cluster-Prior + Merged-Cluster Verification
        snippets = []
        prev_host = expected_hosts[0]
        seg_counts: Dict[str, int] = {}
        sec_totals: Dict[str, float] = {}

        for seg in split_segments:
            raw_cluster = seg.get("speaker", "Unknown")
            cluster_info = speaker_diagnostics.get(raw_cluster, {})
            cluster_assigned = speaker_map.get(raw_cluster, raw_cluster)
            is_merged = cluster_info.get("is_merged_cluster", False)
            merged_candidates = cluster_info.get("merged_candidates", expected_hosts)

            start_s = round(float(seg.get("start", 0.0)), 2)
            end_s = round(float(seg.get("end", start_s)), 2)
            dur_s = round(max(0.0, end_s - start_s), 2)

            assigned_host = cluster_assigned

            # CASE 1: Guest / Third-Party Cluster -> Keep locked as Guest unless >= 2.0s and >= 0.68 host match
            if cluster_assigned in ("Guest", "Clip / Third-Party"):
                if dur_s >= 2.0:
                    chunk = audio_np[max(0, int(start_s * sr)) : min(len(audio_np), int(end_s * sr))]
                    v = self._extract_embedding_fast(chunk, sr)
                    if v is not None:
                        scores = {h: float(np.dot(v, norm_centroids[h])) for h in expected_hosts}
                        best_h = max(scores, key=scores.get)
                        if scores[best_h] >= 0.68:
                            assigned_host = best_h

            # CASE 2: Pure Host Cluster (not merged) -> Trust cluster label by default;
            # only override if segment >= 1.5s AND another host beats cluster_assigned by >= 0.15
            elif not is_merged and cluster_assigned in expected_hosts:
                if dur_s >= 1.5:
                    chunk = audio_np[max(0, int(start_s * sr)) : min(len(audio_np), int(end_s * sr))]
                    v = self._extract_embedding_fast(chunk, sr)
                    if v is not None:
                        scores = {h: float(np.dot(v, norm_centroids[h])) for h in expected_hosts}
                        best_h = max(scores, key=scores.get)
                        if (
                            best_h != cluster_assigned
                            and scores[best_h] >= 0.56
                            and (scores[best_h] - scores.get(cluster_assigned, 0.0)) >= 0.15
                        ):
                            assigned_host = best_h

            # CASE 3: Merged Host Cluster (e.g. Adum + Jake in Ep 200) -> Score directly between merged hosts
            else:
                candidate_pool = [h for h in merged_candidates if h in expected_hosts] or expected_hosts
                if dur_s >= 0.4:
                    chunk = audio_np[max(0, int(start_s * sr)) : min(len(audio_np), int(end_s * sr))]
                    v = self._extract_embedding_fast(chunk, sr)
                    if v is not None:
                        scores = {h: float(np.dot(v, norm_centroids[h])) for h in candidate_pool}
                        # Slight +0.02 continuity prior to cluster default to prevent noisy flips
                        if cluster_assigned in scores:
                            scores[cluster_assigned] += 0.02
                        best_h = max(scores, key=scores.get)
                        if scores[best_h] >= 0.42:
                            assigned_host = best_h
                            prev_host = best_h
                        else:
                            assigned_host = prev_host
                    else:
                        assigned_host = prev_host
                else:
                    assigned_host = prev_host

            if assigned_host in expected_hosts:
                prev_host = assigned_host

            seg_counts[assigned_host] = seg_counts.get(assigned_host, 0) + 1
            sec_totals[assigned_host] = sec_totals.get(assigned_host, 0.0) + dur_s

            h, rem = divmod(int(start_s), 3600)
            m, s_int = divmod(rem, 60)

            snippets.append({
                "speaker": assigned_host,
                "speaker_cluster": raw_cluster,
                "text": seg.get("text", "").strip(),
                "start_seconds": start_s,
                "end_seconds": end_s,
                "duration": dur_s,
                "timestamp": f"{h:02d}:{m:02d}:{s_int:02d}",
                "youtube_link": f"https://youtu.be/{video_id}?t={int(start_s)}s",
                "words": [
                    {
                        "word": w.get("word", ""),
                        "start": round(float(w["start"]), 2) if "start" in w else start_s,
                        "end": round(float(w["end"]), 2) if "end" in w else end_s,
                        "speaker": assigned_host,
                    }
                    for w in seg.get("words", [])
                    if "word" in w
                ],
            })

        speaker_summary = {
            spk: {
                "segments": seg_counts[spk],
                "minutes": round(sec_totals[spk] / 60.0, 1),
            }
            for spk in sorted(seg_counts.keys())
        }
        print(f"  ├─ Step 4 (Hybrid Split & Verification) complete in {time.time() - t3:.1f}s")
        print(f"  └─ Final Speaker Summary: {json.dumps(speaker_summary)}")

        return {
            "meta": meta,
            "speaker_summary": speaker_summary,
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

def get_compressed_flac_bytes(video_id: str) -> bytes:
    """
    Ensures audio_cache/{video_id}.wav exists (downloading via yt-dlp if missing),
    then compresses it in-memory to 16kHz mono Opus (ogg) bytes for fast Modal upload.
    """
    import shutil
    import subprocess
    from pathlib import Path
    import yt_dlp

    cache_dir = Path("audio_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav_path = cache_dir / f"{video_id}.wav"

    # Locate ffmpeg binary (check ./bin/ffmpeg.exe first, then system PATH)
    local_bin_dir = Path("bin")
    local_ffmpeg = local_bin_dir / ("ffmpeg.exe" if Path("C:/").exists() else "ffmpeg")
    if local_ffmpeg.exists():
        ffmpeg_bin = str(local_ffmpeg.resolve())
        ffmpeg_location = str(local_bin_dir.resolve())
    else:
        ffmpeg_bin = shutil.which("ffmpeg")
        ffmpeg_location = None

    if not ffmpeg_bin:
        raise RuntimeError("FFmpeg not found in ./bin or system PATH.")

    # 1. Download and convert to 16kHz mono WAV if not already cached locally
    if not wav_path.exists() or wav_path.stat().st_size < 1024:
        url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"📥 '{wav_path}' not found. Downloading audio from YouTube ({url})...")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(cache_dir / f"{video_id}.%(ext)s"),
            "extractor_args": {"youtube": {"player_client": ["android", "ios"]}},
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "postprocessor_args": ["-ac", "1", "-ar", "16000"],
            "retries": 10,
            "fragment_retries": 10,
            "socket_timeout": 60,
            "quiet": False,
        }
        if ffmpeg_location:
            ydl_opts["ffmpeg_location"] = ffmpeg_location

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not wav_path.exists():
            raise FileNotFoundError(f"yt-dlp finished, but '{wav_path}' was not created.")
        print(f"✅ Downloaded and saved 16kHz mono WAV: {wav_path} ({wav_path.stat().st_size / (1024**2):.1f} MB)")

    # 2. Compress WAV to Opus-in-Ogg in memory (auto-scale bitrate for 3+ hr marathons)
    wav_size_mb = wav_path.stat().st_size / (1024 * 1024)
    bitrate = "32k" if wav_size_mb > 350 else "64k"
    print(f"🗜️ Compressing {wav_path.name} ({wav_size_mb:.1f} MB) to {bitrate} Opus for upload...")

    cmd = [
        ffmpeg_bin, "-y", "-i", str(wav_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", bitrate,
        "-f", "ogg", "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=1800,  # 30-min local safety ceiling for compressing 5+ hr files
    )
    ogg_bytes = proc.stdout
    print(f"📦 Ready to upload {video_id}: {len(ogg_bytes) / (1024 * 1024):.2f} MB")
    return ogg_bytes

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

            # Delete the ~230MB local .wav file once the transcript JSON is safely saved
            Path(f"audio_cache/{vid}.wav").unlink(missing_ok=True)
        except Exception as e:
            print(f"❌ Failed on {vid}: {e}")