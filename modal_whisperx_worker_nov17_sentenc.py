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
# 2. Metadata Prompt Builder, Host Priors & Linguistic Rules
# ---------------------------------------------------------------------------

HOST_CHANNEL_MARKERS = {
    "adum": ["yourmoviesucks", "yms", "adum", "adam", "2gay2lift"],
    "alex": ["iheofficial", "i hate everything", "alex", "ihe_official"],
    "ralph": ["ralphthemoviemaker", "ralphsepe", "ralph", "ralph sepe"],
    "jake": ["mancarryingthing", "mancarrying", "womancarryingman", "jake"],
}

# Strict mid-phrase function words that cannot end a sentence unless the speaker trails off (pause >= 0.55s)
STRICT_HANGING_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her",
    "its", "our", "their", "some", "any", "every", "no", "of", "in", "to", "for",
    "with", "on", "at", "from", "by", "about", "into", "through", "over", "after",
    "between", "without", "under", "within", "along",
    "have", "has", "had", "is", "was", "were", "are", "am", "been", "being",
    "will", "would", "could", "should", "can", "may", "might", "must",
    "do", "did", "does", "don't", "didn't", "doesn't", "wasn't", "weren't",
    "isn't", "aren't", "haven't", "hasn't", "hadn't", "won't", "wouldn't",
    "couldn't", "shouldn't", "can't", "i", "we", "they", "he", "she", "who",
    "very", "really", "too", "quite", "rather", "almost", "just",
}

# Clause connectors that normally join clauses, EXCEPT when interrupted by a capitalized turn ("because | Yeah, one of...")
INTERRUPTIBLE_CONNECTORS = {
    "because", "and", "or", "but", "so", "if", "when", "while", "although",
    "since", "unless", "until", "whether", "like",
}

BACKCHANNEL_WORDS = {
    "yeah", "yep", "yup", "mm-hmm", "mhm", "uh-huh", "mm", "hmm",
    "right", "sure", "true", "totally", "exactly", "honestly", "oh", "okay", "ok",
}

# General conversational turn-openers (discourse markers, reactions, interjections)
TURN_OPENERS = BACKCHANNEL_WORDS | {
    "well", "no", "nah", "wait", "actually", "literally", "basically",
    "anyway", "look", "listen", "so", "but", "and", "like", "mean", "guess",
}

TURN_STARTER_BIGRAMS = {
    ("do", "you"), ("did", "you"), ("have", "you"), ("are", "you"), ("were", "you"),
    ("would", "you"), ("can", "you"), ("could", "you"), ("what", "do"), ("what", "did"),
    ("what", "was"), ("what", "is"), ("what", "are"), ("what", "else"), ("how", "do"),
    ("how", "did"), ("how", "high"), ("how", "was"), ("why", "do"), ("why", "did"),
    ("why", "would"), ("well", "what"), ("well", "i"), ("well", "you"), ("well", "it"),
    ("well", "that"), ("well", "he"), ("well", "they"), ("well", "yeah"), ("well", "we'll"),
    ("yeah", "i"), ("yeah", "it"), ("yeah", "it's"), ("yeah", "that"), ("yeah", "that's"),
    ("yeah", "we"), ("yeah", "he"), ("yeah", "do"), ("yeah", "one"), ("no", "i"),
    ("no", "it"), ("oh", "i"), ("oh", "my"), ("oh", "yeah"), ("but", "i"), ("but", "what"),
    ("so", "what"), ("i", "think"), ("i", "mean"), ("i", "feel"), ("i", "don't"),
    ("i", "didn't"), ("i", "guess"), ("it", "was"), ("it", "must"), ("it's", "like"),
    ("that", "was"), ("that's", "like"),
}

DISCOURSE_TAILS = {"know", "right", "honestly", "though", "something", "whatever", "even"}


def _clean_token(w: str) -> str:
    return w.lower().strip(".,?!\"'():;-—…")


def _has_terminal_punct(w: str) -> bool:
    s = w.strip()
    return s.endswith((".", "?", "!")) and not s.endswith("...")


def _is_capitalized_turn_word(w: str) -> bool:
    s = w.strip().lstrip("\"'(")
    if not s or not s[0].isupper():
        return False
    tok = _clean_token(s)
    return tok in TURN_OPENERS or tok in {"what", "how", "why", "do", "did", "have", "are", "we'll"}


def _is_backchannel_phrase(words: List[Dict[str, Any]]) -> bool:
    if not words or len(words) > 4:
        return False
    tokens = [_clean_token(w["word"]) for w in words if _clean_token(w["word"])]
    return bool(tokens) and all(t in BACKCHANNEL_WORDS for t in tokens)


def build_initial_prompt_and_priors(meta: Dict[str, Any]) -> Dict[str, Any]:
    title = meta.get("title", "Sardonicast")
    desc = meta.get("description", "")
    desc_lower = f"{title}\n{desc}".lower()

    expected_hosts = [
        host for host, markers in HOST_CHANNEL_MARKERS.items()
        if any(m in desc_lower for m in markers)
    ]
    if len(expected_hosts) < 2:
        expected_hosts = ["adum", "alex", "ralph", "jake"]

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
    timeout=3600,          # 60-min max timeout (handles 4-7 hr milestone livestreams)
    scaledown_window=60,   # Spin down idle GPU container after 60s to minimize billing
    volumes={MODEL_CACHE_DIR: model_volume},
)
class SardonicastDiarizer:
    @modal.enter()
    def load_models(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.diarize_model = None
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
        self.wespeaker_model.model.register_forward_pre_hook(
            lambda mod, inp: (inp[0].to(next(mod.parameters()).device),)
        )
        model_volume.commit()
        print("✅ Core ASR, Alignment & WeSpeaker models ready.")

    def _extract_embedding_fast(self, chunk_np: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
        """Strict-window GPU WeSpeaker extraction; self-tiles short clips (<1.2s) to prevent neighbor bleed."""
        if len(chunk_np) < int(0.35 * sr):
            return None
        if len(chunk_np) > int(8.0 * sr):
            mid = len(chunk_np) // 2
            half = int(4.0 * sr)
            chunk_np = chunk_np[mid - half : mid + half]

        min_samples = int(1.2 * sr)
        if len(chunk_np) < min_samples:
            reps = int(np.ceil(min_samples / len(chunk_np)))
            chunk_np = np.tile(chunk_np, reps)[:min_samples]

        pcm_int16 = np.clip(chunk_np * 32768.0, -32768, 32767).astype(np.int16)
        pcm_tensor = torch.from_numpy(pcm_int16).unsqueeze(0)
        with torch.inference_mode():
            emb = self.wespeaker_model.extract_embedding_from_pcm(pcm_tensor, sr)
        if emb is None:
            return None
        raw_emb = emb.detach().cpu().numpy() if hasattr(emb, "detach") else np.asarray(emb)
        v = raw_emb.flatten().astype(np.float32)
        norm = np.linalg.norm(v)
        return (v / norm) if norm > 1e-12 else None

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

            if best_sim >= sim_threshold or (is_merged and best_sim >= 0.52):
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

    def _validate_and_refine_segments(
        self,
        raw_segments: List[Dict[str, Any]],
        audio_np: np.ndarray,
        speaker_map: Dict[str, str],
        speaker_diagnostics: Dict[str, Any],
        global_centroids: Dict[str, np.ndarray],
        expected_hosts: List[str],
        video_id: str,
        sr: int = 16000,
    ) -> List[Dict[str, Any]]:
        # 1. Group contiguous word runs from Pyannote word-level speakers
        initial_snippets: List[Dict[str, Any]] = []
        for seg in raw_segments:
            words = [w for w in seg.get("words", []) if "word" in w and "start" in w and "end" in w]
            default_cluster = seg.get("speaker", "Unknown")
            if not words:
                s_val = round(float(seg.get("start", 0.0)), 2)
                e_val = round(float(seg.get("end", s_val)), 2)
                initial_snippets.append({
                    "speaker": speaker_map.get(default_cluster, default_cluster),
                    "speaker_cluster": default_cluster,
                    "text": seg.get("text", "").strip(),
                    "start_seconds": s_val,
                    "end_seconds": e_val,
                    "duration": round(max(0.0, e_val - s_val), 2),
                    "words": [],
                })
                continue

            spks = [w.get("speaker", default_cluster) for w in words]
            for k in range(1, len(words) - 1):
                w_dur = float(words[k]["end"]) - float(words[k]["start"])
                if spks[k - 1] == spks[k + 1] and spks[k] != spks[k - 1] and w_dur < 0.45:
                    spks[k] = spks[k - 1]
                    words[k]["speaker"] = spks[k - 1]

            runs: List[List[Dict[str, Any]]] = [[words[0]]]
            for k in range(1, len(words)):
                if spks[k] == spks[k - 1]:
                    runs[-1].append(words[k])
                else:
                    runs.append([words[k]])

            for r in runs:
                r_cluster = r[0].get("speaker", default_cluster)
                r_host = speaker_map.get(r_cluster, r_cluster)
                s_val = round(float(r[0]["start"]), 2)
                e_val = round(float(r[-1]["end"]), 2)
                initial_snippets.append({
                    "speaker": r_host,
                    "speaker_cluster": r_cluster,
                    "text": " ".join(w["word"].strip() for w in r).strip(),
                    "start_seconds": s_val,
                    "end_seconds": e_val,
                    "duration": round(max(0.0, e_val - s_val), 2),
                    "words": [
                        {
                            "word": w["word"],
                            "start": round(float(w["start"]), 2),
                            "end": round(float(w["end"]), 2),
                            "speaker": r_host,
                        }
                        for w in r
                    ],
                })

        # 2. Calibrate episode-specific host centroids + dynamic Guest centroid
        guest_clusters = {
            spk for spk, info in speaker_diagnostics.items()
            if info.get("assigned") == "Guest"
            and float(info.get("total_sec", 0.0)) >= 180.0
            and float(info.get("best_cosine_sim", 0.0)) < 0.56
        }
        active_speakers = list(expected_hosts) + (["Guest"] if guest_clusters else [])
        confident_vecs: Dict[str, List[np.ndarray]] = {s: [] for s in active_speakers}

        for seg in initial_snippets:
            s, e = float(seg["start_seconds"]), float(seg["end_seconds"])
            if (e - s) < 2.5:
                continue
            chunk = audio_np[max(0, int(s * sr)) : min(len(audio_np), int(e * sr))]
            v = self._extract_embedding_fast(chunk, sr)
            if v is None:
                continue
            if seg["speaker_cluster"] in guest_clusters:
                if max(float(np.dot(v, global_centroids[h])) for h in expected_hosts) < 0.52:
                    confident_vecs["Guest"].append(v)
            else:
                scores = {h: float(np.dot(v, global_centroids[h])) for h in expected_hosts}
                sorted_h = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                best_h, best_s = sorted_h[0]
                runner_up_s = sorted_h[1][1] if len(sorted_h) > 1 else 0.0
                if best_s >= 0.54 and (best_s - runner_up_s) >= 0.10:
                    confident_vecs[best_h].append(v)

        adapted_centroids: Dict[str, np.ndarray] = {}
        for h in expected_hosts:
            if len(confident_vecs[h]) >= 8:
                ep_mean = np.mean(confident_vecs[h], axis=0)
                ep_mean = ep_mean / (np.linalg.norm(ep_mean) + 1e-12)
                blended = 0.65 * ep_mean + 0.35 * global_centroids[h]
                adapted_centroids[h] = blended / (np.linalg.norm(blended) + 1e-12)
            else:
                adapted_centroids[h] = global_centroids[h]

        if "Guest" in active_speakers and len(confident_vecs["Guest"]) >= 5:
            g_mean = np.mean(confident_vecs["Guest"], axis=0)
            adapted_centroids["Guest"] = g_mean / (np.linalg.norm(g_mean) + 1e-12)

        # Helper A: Strictly target Uppercase stranded opener ('...time. Well,' or '...because Yeah,')
        # followed by a lowercase continuation ('yeah, we'll...' or 'one of my...'), plus trailing '. Yeah,'
        def fix_boundary_backchannels(segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            for idx in range(len(segs) - 1):
                cw, nw = segs[idx].get("words", []), segs[idx + 1].get("words", [])
                if len(cw) < 2 or not nw:
                    continue
                last_w, prev_w, first_nw = cw[-1], cw[-2], nw[0]
                last_str = last_w["word"].strip().lstrip("\"'(")
                last_tok = _clean_token(last_str)
                first_n_str = first_nw["word"].strip().lstrip("\"'(")
                gap = float(first_nw["start"]) - float(last_w["end"])

                # Strictly require: last word of current seg is Uppercase (non-terminal)
                # AND first word of next seg is lowercase
                is_upper_to_lower = (
                    bool(last_str and last_str[0].isupper())
                    and not _has_terminal_punct(last_w["word"])
                    and bool(first_n_str and first_n_str[0].islower())
                )
                # Must either follow a period ('...time. Well,') or be a capitalized turn-opener ('...because Yeah,')
                is_valid_opener_context = _has_terminal_punct(prev_w["word"]) or (last_tok in TURN_OPENERS)

                if is_upper_to_lower and is_valid_opener_context and gap < 0.50:
                    moved = cw.pop()
                    moved["speaker"] = segs[idx + 1]["speaker"]
                    nw.insert(0, moved)
                    segs[idx]["end_seconds"] = round(float(cw[-1]["end"]), 2)
                    segs[idx]["duration"] = round(segs[idx]["end_seconds"] - float(segs[idx]["start_seconds"]), 2)
                    segs[idx]["text"] = " ".join(w["word"].strip() for w in cw).strip()
                    segs[idx + 1]["start_seconds"] = round(float(nw[0]["start"]), 2)
                    segs[idx + 1]["duration"] = round(float(segs[idx + 1]["end_seconds"]) - segs[idx + 1]["start_seconds"], 2)
                    segs[idx + 1]["text"] = " ".join(w["word"].strip() for w in nw).strip()

            out: List[Dict[str, Any]] = []
            for seg in segs:
                words = seg.get("words", [])
                if len(words) >= 4:
                    d_idx = None
                    if _clean_token(words[-1]["word"]) in BACKCHANNEL_WORDS and _has_terminal_punct(words[-2]["word"]):
                        d_idx = len(words) - 1
                    elif (
                        len(words) >= 5
                        and _clean_token(words[-1]["word"]) in BACKCHANNEL_WORDS
                        and _clean_token(words[-2]["word"]) in TURN_OPENERS
                        and _has_terminal_punct(words[-3]["word"])
                    ):
                        d_idx = len(words) - 2
                    if d_idx is not None:
                        mw, tw = words[:d_idx], words[d_idx:]
                        ms, me = round(float(mw[0]["start"]), 2), round(float(mw[-1]["end"]), 2)
                        ts, te = round(float(tw[0]["start"]), 2), round(float(tw[-1]["end"]), 2)
                        out.append({
                            **seg,
                            "start_seconds": ms,
                            "end_seconds": me,
                            "duration": round(me - ms, 2),
                            "text": " ".join(w["word"].strip() for w in mw).strip(),
                            "words": mw,
                        })
                        out.append({
                            **seg,
                            "start_seconds": ts,
                            "end_seconds": te,
                            "duration": round(te - ts, 2),
                            "text": " ".join(w["word"].strip() for w in tw).strip(),
                            "words": tw,
                            "_detached_tail_backchannel": True,
                        })
                        continue
                out.append(seg)
            return out

        # Helper B: Heal mid-phrase hangers ("A | diverse group") and true A -> B -> A crosstalk hijacks
        def heal_sandwiches(segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            idx = 1
            while idx < len(segs) - 1:
                ps, cs, ns = segs[idx - 1], segs[idx], segs[idx + 1]
                gap_b = float(cs["start_seconds"]) - float(ps["end_seconds"])
                gap_a = float(ns["start_seconds"]) - float(cs["end_seconds"])
                pw, cw, nw = ps.get("words", []), cs.get("words", []), ns.get("words", [])
                last_pw = pw[-1]["word"] if pw else ""
                cs_is_full_turn = bool(cw and _is_capitalized_turn_word(cw[0]["word"]) and _has_terminal_punct(cw[-1]["word"]))

                if (
                    ps["speaker"] == ns["speaker"]
                    and cs["speaker"] != ps["speaker"]
                    and cs["duration"] <= 2.8
                    and gap_b < 0.40
                    and gap_a < 0.40
                    and not _is_backchannel_phrase(cw)
                    and not _is_backchannel_phrase(nw)
                    and not cs_is_full_turn
                    and (not _has_terminal_punct(last_pw) or _clean_token(last_pw) in STRICT_HANGING_WORDS)
                ):
                    mw = pw + cw + nw
                    for w in mw:
                        w["speaker"] = ps["speaker"]
                    ps["words"] = mw
                    ps["end_seconds"] = ns["end_seconds"]
                    ps["duration"] = round(float(ps["end_seconds"]) - float(ps["start_seconds"]), 2)
                    ps["text"] = " ".join(w["word"].strip() for w in mw).strip()
                    segs.pop(idx + 1)
                    segs.pop(idx)
                    continue
                idx += 1

            idx = 0
            while idx < len(segs) - 1:
                cs, ns = segs[idx], segs[idx + 1]
                cw, nw = cs.get("words", []), ns.get("words", [])
                if not cw or not nw:
                    idx += 1
                    continue
                last_w = cw[-1]["word"]
                gap = float(ns["start_seconds"]) - float(cs["end_seconds"])
                if (
                    _clean_token(last_w) in STRICT_HANGING_WORDS
                    and not _has_terminal_punct(last_w)
                    and not last_w.strip().endswith("...")
                    and gap < 0.45
                    and not _is_backchannel_phrase(nw)
                    and not _is_capitalized_turn_word(nw[0]["word"])
                ):
                    dom = cs["speaker"] if cs["duration"] >= ns["duration"] else ns["speaker"]
                    mw = cw + nw
                    for w in mw:
                        w["speaker"] = dom
                    cs["speaker"] = dom
                    cs["words"] = mw
                    cs["end_seconds"] = ns["end_seconds"]
                    cs["duration"] = round(float(cs["end_seconds"]) - float(cs["start_seconds"]), 2)
                    cs["text"] = " ".join(w["word"].strip() for w in mw).strip()
                    segs.pop(idx + 1)
                    continue
                idx += 1
            return segs

        # Execute Unified 5-Step Linguistic + Acoustic Pipeline
        s1 = fix_boundary_backchannels(initial_snippets)
        s2 = heal_sandwiches(s1)

        # Step 3: Split unpunctuated multi-speaker clause run-ons
        speaker_pool = list(adapted_centroids.keys())
        s3: List[Dict[str, Any]] = []
        for seg in s2:
            queue = [seg]
            while queue:
                curr = queue.pop(0)
                words = curr.get("words", [])
                s_start, s_end = float(curr["start_seconds"]), float(curr["end_seconds"])
                if (s_end - s_start) < 2.8 or len(words) < 6:
                    s3.append(curr)
                    continue

                tokens = [_clean_token(w["word"]) for w in words]
                cands = []
                for k in range(2, len(words) - 1):
                    pw, cw = words[k - 1]["word"], words[k]["word"]
                    pt, ct = tokens[k - 1], tokens[k]
                    nt = tokens[k + 1] if k + 1 < len(words) else ""
                    gap = float(words[k]["start"]) - float(words[k - 1]["end"])

                    # Guard 1: Never split right after a STRICT hanging word (a, the, of, to)
                    if pt in STRICT_HANGING_WORDS and gap < 0.55:
                        continue

                    # Guard 2: Never split between an Uppercase turn-opener ('Well,', 'Yeah,')
                    # and its lowercase continuation ('yeah,', 'one'), without blocking proper nouns ('Leo', 'Fantasia')
                    pw_clean_str = pw.strip().lstrip("\"'(")
                    cw_clean_str = cw.strip().lstrip("\"'(")
                    prev_follows_period = k >= 2 and _has_terminal_punct(words[k - 2]["word"])
                    if (
                        pw_clean_str
                        and pw_clean_str[0].isupper()
                        and not _has_terminal_punct(pw)
                        and (pt in TURN_OPENERS or prev_follows_period)
                        and cw_clean_str
                        and cw_clean_str[0].islower()
                        and gap < 0.40
                    ):
                        continue

                    if pt in INTERRUPTIBLE_CONNECTORS and gap < 0.35:
                        if not (_is_capitalized_turn_word(cw) or (ct, nt) in TURN_STARTER_BIGRAMS):
                            continue
                    if (
                        _has_terminal_punct(pw)
                        or gap >= 0.26
                        or (ct, nt) in TURN_STARTER_BIGRAMS
                        or _is_capitalized_turn_word(cw)
                        or (pt in DISCOURSE_TAILS and (ct in TURN_OPENERS or (ct, nt) in TURN_STARTER_BIGRAMS or gap >= 0.05))
                        or (ct in BACKCHANNEL_WORDS and pt not in TURN_OPENERS)
                        or (pt in BACKCHANNEL_WORDS and ct not in TURN_OPENERS and (_has_terminal_punct(pw) or gap >= 0.22 or not pw.strip().endswith(",")))
                    ):
                        cands.append(k)

                best_k, best_gain = None, 0.0
                for k in cands:
                    t_sp = float(words[k]["start"])
                    if (float(words[k - 1]["end"]) - s_start) < 0.8 or (s_end - t_sp) < 0.8:
                        continue
                    l_ch = audio_np[max(0, int(max(s_start, t_sp - 3.0) * sr)) : int(float(words[k - 1]["end"]) * sr)]
                    r_ch = audio_np[int(t_sp * sr) : min(len(audio_np), int(min(s_end, t_sp + 3.0) * sr))]
                    vl, vr = self._extract_embedding_fast(l_ch, sr), self._extract_embedding_fast(r_ch, sr)
                    if vl is None or vr is None:
                        continue
                    sl = {h: float(np.dot(vl, adapted_centroids[h])) for h in speaker_pool}
                    sr_scores = {h: float(np.dot(vr, adapted_centroids[h])) for h in speaker_pool}
                    hl, hr = max(sl, key=sl.get), max(sr_scores, key=sr_scores.get)
                    if hl != hr and sl[hl] >= 0.42 and sr_scores[hr] >= 0.42 and float(np.dot(vl, vr)) < 0.58:
                        gain = (sl[hl] - sl[hr]) + (sr_scores[hr] - sr_scores[hl])
                        if gain >= 0.14 and gain > best_gain:
                            best_gain, best_k = gain, k

                if best_k is not None:
                    lw, rw = words[:best_k], words[best_k:]
                    ls, le = round(float(lw[0]["start"]), 2), round(float(lw[-1]["end"]), 2)
                    rs, re = round(float(rw[0]["start"]), 2), round(float(rw[-1]["end"]), 2)
                    s3.append({
                        **curr,
                        "start_seconds": ls,
                        "end_seconds": le,
                        "duration": round(le - ls, 2),
                        "text": " ".join(w["word"].strip() for w in lw).strip(),
                        "words": lw,
                    })
                    queue.insert(0, {
                        **curr,
                        "start_seconds": rs,
                        "end_seconds": re,
                        "duration": round(re - rs, 2),
                        "text": " ".join(w["word"].strip() for w in rw).strip(),
                        "words": rw,
                    })
                else:
                    s3.append(curr)

        s4 = fix_boundary_backchannels(s3)

        # Step 5: Score whole clauses + final sandwich cleanup
        prev_spk = speaker_pool[0]
        for idx, seg in enumerate(s4):
            s_val, e_val = float(seg["start_seconds"]), float(seg["end_seconds"])
            dur = max(0.0, e_val - s_val)
            raw_c = seg.get("speaker_cluster", "Unknown")
            c_def = "Guest" if raw_c in guest_clusters else speaker_map.get(raw_c, seg.get("speaker", prev_spk))
            is_m = speaker_diagnostics.get(raw_c, {}).get("is_merged_cluster", False)
            is_detached_tail = seg.pop("_detached_tail_backchannel", False)

            chunk = audio_np[max(0, int(s_val * sr)) : min(len(audio_np), int(e_val * sr))]
            v = self._extract_embedding_fast(chunk, sr) if dur >= 0.40 else None

            if is_detached_tail and len(speaker_pool) >= 2:
                # A trailing '. Yeah,' detached from the end of prev_spk's completed sentence belongs to a listener
                listener_pool = [h for h in speaker_pool if h != prev_spk]
                if v is not None:
                    l_scores = {h: float(np.dot(v, adapted_centroids[h])) for h in listener_pool}
                    assigned = max(l_scores, key=l_scores.get)
                else:
                    assigned = listener_pool[0]
            elif v is not None:
                scores = {h: float(np.dot(v, adapted_centroids[h])) for h in speaker_pool}
                if not is_m and c_def in scores and dur >= 1.0:
                    scores[c_def] += 0.05
                best_h = max(scores, key=scores.get)
                assigned = best_h if scores[best_h] >= 0.42 else (c_def if c_def in speaker_pool else prev_spk)
            else:
                if _is_backchannel_phrase(seg.get("words", [])) and idx > 0 and len(speaker_pool) >= 2:
                    others = [h for h in speaker_pool if h != prev_spk]
                    assigned = c_def if c_def in others else others[0]
                else:
                    assigned = c_def if (not is_m and c_def in speaker_pool) else prev_spk

            seg["speaker"] = assigned
            prev_spk = assigned
            for w in seg.get("words", []):
                w["speaker"] = assigned

        final_segs = heal_sandwiches(s4)

        for seg in final_segs:
            s_int = int(seg["start_seconds"])
            h_i, rem = divmod(s_int, 3600)
            m_i, sec_i = divmod(rem, 60)
            seg["timestamp"] = f"{h_i:02d}:{m_i:02d}:{sec_i:02d}"
            seg["youtube_link"] = f"https://youtu.be/{video_id}?t={s_int}s"

        return final_segs

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
            language="en",
        )
        print(f"  ├─ Step 1 (WhisperX ASR) complete in {time.time() - t0:.1f}s")

        gc.collect()
        torch.cuda.empty_cache()

        # Step 2: Wav2Vec2 Phoneme-Level Forced Alignment
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
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  ├─ Step 3 (Pyannote 3.1 Diarization, min_speakers={min_spk}) complete in {time.time() - t2:.1f}s")

        # Step 4: Unified Linguistic + Acoustic Validation
        t3 = time.time()
        speaker_map, speaker_diagnostics = self._match_clusters_to_hosts(
            diarize_df, audio_np, host_centroids, expected_hosts
        )
        final_result = whisperx.assign_word_speakers(diarize_df, aligned_result)

        norm_centroids = {
            k: np.asarray(v, dtype=np.float32) / (np.linalg.norm(v) + 1e-12)
            for k, v in host_centroids.items()
        }
        snippets = self._validate_and_refine_segments(
            final_result.get("segments", []),
            audio_np,
            speaker_map,
            speaker_diagnostics,
            norm_centroids,
            expected_hosts,
            video_id,
            sr=16000,
        )

        seg_counts: Dict[str, int] = {}
        sec_totals: Dict[str, float] = {}
        for s in snippets:
            spk = s["speaker"]
            seg_counts[spk] = seg_counts.get(spk, 0) + 1
            sec_totals[spk] = sec_totals.get(spk, 0.0) + float(s["duration"])

        speaker_summary = {
            spk: {
                "segments": seg_counts[spk],
                "minutes": round(sec_totals[spk] / 60.0, 1),
            }
            for spk in sorted(seg_counts.keys())
        }
        print(f"  ├─ Step 4 (Unified Linguistic + Acoustic Validation) complete in {time.time() - t3:.1f}s")
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

        base_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(cache_dir / f"{video_id}.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "postprocessor_args": ["-ac", "1", "-ar", "16000"],
            "retries": 10,
            "fragment_retries": 10,
            "retry_sleep_functions": {"http": lambda n: min(2 * n, 15)},
            "sleep_interval_requests": 1,
            "socket_timeout": 60,
            "nocheckcertificate": True,
            "quiet": False,
        }
        if ffmpeg_location:
            base_opts["ffmpeg_location"] = ffmpeg_location

        # Try clean mobile/embedded clients first; fall back to default extractor if 403 occurs
        client_profiles = [
            {"youtube": {"player_client": ["android", "web_embedded"]}},
            {"youtube": {"player_client": ["android", "ios"]}},
            {},  # Let yt-dlp use its native default rotation
        ]

        last_err = None
        for profile in client_profiles:
            ydl_opts = dict(base_opts)
            if profile:
                ydl_opts["extractor_args"] = profile
            try:
                # Clean up any partial .part file from a failed attempt before retrying
                for part_file in cache_dir.glob(f"{video_id}.*.part"):
                    part_file.unlink(missing_ok=True)

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                if wav_path.exists() and wav_path.stat().st_size >= 1024:
                    break
            except Exception as e:
                last_err = e
                print(f"⚠️ Download attempt with profile {profile or 'default'} failed ({e}), trying fallback...")

        if not wav_path.exists() or wav_path.stat().st_size < 1024:
            raise RuntimeError(f"All yt-dlp download profiles failed for {video_id}: {last_err}")

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
        timeout=1800,
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
    keep_wav: bool = True,
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

            if not keep_wav:
                Path(f"audio_cache/{vid}.wav").unlink(missing_ok=True)
        except Exception as e:
            print(f"❌ Failed on {vid}: {e}")