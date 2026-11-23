# build_dataset.py
import os
# Prevent OpenMP / MKL deadlock between PyTorch and Pandas/NumPy
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import csv
import torch
import traceback
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from pathlib import Path

# Import existing utils
from attempt_3_ds.utils import (
    fetch_youtube_metadata,
    parse_description_timestamps,
    fetch_and_clean_transcript
)

CACHE_DIR = Path(__file__).resolve().parent / "transcript_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Load lightweight or full embedding model
# Auto-detect CUDA GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"  # or 'all-MiniLM-L6-v2' for faster training runs
# Load BAAI/bge-m3 with FP16 precision on CUDA
model = SentenceTransformer(
    EMBEDDING_MODEL_NAME, 
    device=device,
    model_kwargs={
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "use_safetensors": True
    }
)
model.max_seq_length = 256

OUTRO_PHRASES = ["final thoughts", "rating", "out of 10", "recommend", "next episode", "thanks for listening", "patreon"]

def get_cached_transcript_and_meta(vid_id: str):
    """Loads from local disk cache if available, otherwise fetches via network."""
    cache_path = os.path.join(CACHE_DIR, f"{vid_id}.json")
    
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["meta"], data["snippets"]
            
    # Fetch from network
    meta = fetch_youtube_metadata(vid_id)
    if not meta or not meta.get("description"):
        return None, None
        
    snippets = fetch_and_clean_transcript(vid_id)
    if not snippets:
        return None, None
        
    # Save to cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "snippets": snippets}, f, ensure_ascii=False)
        
    return meta, snippets

def extract_features_for_candidates(
    snippets: List[Dict[str, Any]], 
    ground_truth_chapters: List[Dict[str, Any]],
    window_sec: float = 60.0,
    stride_sec: float = 5.0,
    tolerance_sec: float = 45.0,
    gpu_batch_size: int = 128
) -> Dict[str, List[Any]]: #pd.DataFrame:
    """Generates feature vectors for potential split points in a transcript."""
    if not snippets or not ground_truth_chapters:
        return pd.DataFrame()

    total_duration = snippets[-1]["start_seconds"] + (snippets[-1].get("duration") or 5.0)
    gt_starts = [ch["start_seconds"] for ch in ground_truth_chapters if ch["start_seconds"] > 10.0]

    snippet_texts = [s["text"] for s in snippets]
    print("creating np array")
    snippet_times = np.array([s["start_seconds"] for s in snippets], dtype=np.float64)
    snippet_texts_arr = np.array(snippet_texts)

    print("encoding snippet vecs")
    # 1. GPU Encoding Step
    snippet_vecs = model.encode(
        snippet_texts, 
        batch_size=gpu_batch_size, 
        show_progress_bar=False, 
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    print("creating np array from snippet vecs")
    # FOOLPROOF STEP: Cast vectors to float32 NumPy array & flush CUDA thread handles
    snippet_vecs = np.asarray(snippet_vecs, dtype=np.float32)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # 2. Pure NumPy / CPU Feature Extraction Loop
    candidate_times = np.arange(window_sec, total_duration - window_sec, stride_sec)
    
    # Pre-allocate column lists for direct DataFrame construction
    timestamps, rel_positions, cosine_dists = [], [], []
    has_outros, is_boundaries, gap_durations = [], [], []
    extracted_rows = []

    for curr_time in candidate_times:
        if curr_time % 3600 == 0:
            print(f"processing curr_time: {curr_time}")
        left_mask = (snippet_times >= (curr_time - window_sec)) & (snippet_times < curr_time)
        right_mask = (snippet_times >= curr_time) & (snippet_times < (curr_time + window_sec))

        if not np.any(left_mask) or not np.any(right_mask):
            continue

        # Mean pooling
        vec_left = snippet_vecs[left_mask].mean(axis=0)
        vec_right = snippet_vecs[right_mask].mean(axis=0)
        
        # Unit re-normalization
        vec_left /= (np.linalg.norm(vec_left) + 1e-9)
        vec_right /= (np.linalg.norm(vec_right) + 1e-9)

        # Cosine distance (strictly Python float)
        cosine_dist = float(1.0 - np.dot(vec_left, vec_right))
        rel_pos = float(curr_time / total_duration)

        combined_mask = left_mask | right_mask
        combined_text = " ".join(snippet_texts_arr[combined_mask]).lower()
        has_outro_kw = int(any(phrase in combined_text for phrase in OUTRO_PHRASES))

        is_boundary = int(any(abs(curr_time - gt_t) <= tolerance_sec for gt_t in gt_starts))

        curr_idx = np.searchsorted(snippet_times, curr_time)
        if 0 < curr_idx < len(snippets):
            prev_end = snippets[curr_idx - 1]["start_seconds"] + (snippets[curr_idx - 1].get("duration") or 0.0)
            gap_dur = float(max(0.0, curr_time - prev_end))
        else:
            gap_dur = 0.0

        # timestamps.append(float(curr_time))
        # rel_positions.append(rel_pos)
        # cosine_dists.append(cosine_dist)
        # has_outros.append(has_outro_kw)
        # is_boundaries.append(is_boundary)
        # gap_durations.append(gap_dur)
        # Output pure primitives: [video_id added later, timestamp, rel_pos, cosine_dist, gap_dur, has_outro, is_boundary]
        extracted_rows.append([
            float(curr_time),
            rel_pos,
            cosine_dist,
            gap_dur,
            has_outro_kw,
            is_boundary
        ])

    # if not timestamps:
    #     return pd.DataFrame()

    # Direct column-based DataFrame creation (bypasses dictionary C-extension parsing)
    # print("creating dict...")
    # ret_dict = {
    #     "timestamp_sec": timestamps,
    #     "rel_position": rel_positions,
    #     "cosine_dist": cosine_dists,
    #     "has_outro_keyword": has_outros,
    #     "is_boundary": is_boundaries,
    #     "gap_duration": gap_durations
    # }
    print("returning rows...")
    return extracted_rows
    # return ret_dict

def build_dataset_from_video_list(video_ids: List[str], output_csv: str = "boundary_dataset.csv"):
    fieldnames = ["video_id", "timestamp_sec", "rel_position", "cosine_dist", "gap_duration", "has_outro_keyword", "is_boundary"]
    
    # Check if CSV exists to handle header logic for stacking/appending
    file_exists = os.path.exists(output_csv)
    
    # Open CSV in append mode ('a') - Writes directly to disk without Pandas
    with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write header only if starting a fresh file
        if not file_exists:
            writer.writerow(fieldnames)
            f.flush()

        for i, vid_id in enumerate(video_ids, 1):
            print(f"[{i}/{len(video_ids)}] Processing episode: {vid_id}...")
            
            try:
                meta, snippets = get_cached_transcript_and_meta(vid_id)
                if not meta or not snippets:
                    print(f"  ⚠️ Skipping {vid_id}: Unable to load metadata/transcript.")
                    continue

                gt_chapters = parse_description_timestamps(meta.get("description", ""))
                if not gt_chapters:
                    print(f"  ⚠️ Skipping {vid_id}: No ground-truth timestamps in description.")
                    continue

                rows = extract_features_for_candidates(snippets, gt_chapters)
                if rows:
                    # Prepend video_id to each row and write directly to disk
                    for r in rows:
                        writer.writerow([vid_id] + r)
                    
                    # Force disk write immediately
                    f.flush()
                    print(f"  ✅ Saved {len(rows)} rows for {vid_id} directly to disk.")

            except Exception as e:
                print(f"  ❌ Error processing {vid_id}: {str(e)}")
                continue
                
            finally:
                if device == "cuda":
                    torch.cuda.empty_cache()

    print(f"\n🎉 Incremental run complete! All data written to {output_csv}")

if __name__ == "__main__":
    # Add your training video IDs here
    test_video_ids =  ["8zNF77wosvQ", "WpC75FsTKgI", "XknVaaXJY_k", "djAw3gYe0eA", "MH0_ccLF_wQ", "jhsLkaovRf4", "aVbILMqNxgQ", "Xdrf3vy7Edk", "A0Wy9u6DPj8", "kTElzwepV88", "9dyGSglqflk", "yp8GigxUc8A", "JtEkJ22uDAY", "MCwW-vnG5S8", "GDiZzicN71w", "1PaTzu-auA8", "bAw40bYAbuQ", "0rx44FdnEIY", "dZrdd2MmkNs", "obNj3gaJros", "n8uX0X9eiVg", "cis2raB_Bdo", "Xaal5pQ-Zl0", "Nf2wmGN3mkQ", "QKA13IdBZd4", "Ce3IJL3pPwg", "cTNEIRS50D8", "vxObcVWDpJY", "FGNmMheGw8c", "1s5SWZNR0Z4", "QCv6CJNVJi8", "fExN7E8vtZ0", "Ds_FVmolrrM", "nnC-qeutqFM", "ZEz20RdBlQA", "Q_jR1U82wvc", "f-OmVBjIC5M", "jdfSNUgJxCI", "a9lovmzo94g", "g66tJLCQmlQ", "leN-vjrW1eU", "I7IycKwKZ0c", "VqoBXdHw87g", "QQ7X33bzLBQ", "GkAxbUVqHVQ", "STFAcONCXRM", "W43k4lgx6pM", "ym-hIVRp-4Q", "Wl-4iuEgXoo", "01Fcmcwk15E", "9FqJN-6iq9I", "stCaFJ77Nc0", "TVZBFjs8EAI", "NWRf6vTheB4", "0dRTNWui2qM", "rM6qPSeEvcA", "Tuiv84ZDRyM", "Xuu-qqq25hU", "RFetquVwQiE", "a4YvDSgzUYg", "cS0jXqcBZFg", "dpwbfTta0XE", "6O5bEy7_bSM", "YXdqlGFOmNw", "WZP1ab8i9nk", "stfN4kzaDcw", "PWyWirZiHsg", "VfBes3X1ovk", "LT_rSGNHHtA", "b7OrCQP5EC4", "anKdXQ-kfyc", "bvxlnZ5UFcs", "AuUHVdAKS_Y", "_bvT65wMeoE", "mPWvB0WEwZw", "NP-YRmcWzqE", "bUKJkaLWeGw", "abxgAZ2a-RQ", "ANsRcUCwrbI", "Zb9bB0t4tyU", "lScrhK9Za6k", "e4e0e7APJT0", "tVG-PoH-n80", "AhY_XvQt2KU", "BZLbkzMX57Y", "NiTtWpPrVp4", "kRQHSycmvd0", "1MVRQj2vr9Q", "SYjANODJkzc", "u4_9_elgJyg", "4IP8hfMPzas", "xtSnzvJFyYU", "9MTD8kOpGhk", "nMwkk5jozAw", "6QxmbwpFRF4", "8nP1uXKmwSY", "7nlPtTUtHrw", "leOu1G1faBE", "aQBFixFqv6M", "HPVFbp_CBFA", "AnNI-4ihV9c", "6jGjbEm99CI", "7qxWAC4A0yk", "xVFrlN1aQIE", "egz-Hx4iLDw", "AExOwJa58no", "0OYydixtVA8", "kdlzHIbwQfs", "Ojz0sskxFBo", "5rL7bBwOuio", "FXv2XUshrNE", "WCA8Jt_qrG4", "nmKCUFGq0hA", "9OZAYwBtTHo"]
    
    #qa
    #["cSkEMK0Z3Eo", "jRGf2_A9jxI", "XnRdkcs7Tqo", "tajqOLlRWq0", "w-OlJK-GRyc", "DvgONdaCpe0", "bzblLkTJJmM", "AZuzd3RWxTA", "XwN6xIEf3Es", "jtuSSzFx4bA", "BgPz4MmstQQ", "miwkDPCDO-E"]
    #test_qa_video_ids = ["9KpIgO6y2PQ", "js7B-K6BtJs", "IybB6sCakxI", "h3xPjF4TGWk", "6Mx9pSn-Y0g", "qU36qE6JaG0", "PJIhUkG-bAY", "z0dUUHx7TJE", "LjtFPxabYvY", "yBq71Tc7R8I", "x1cfOLa6_kI", "xAV2jXtKjW8", "rxshEE7NC7E"] 
    test_qa_video_ids = ["NTxV9b4V0Ow", "qR6gD-esmbE", "qzwXZqZqcpI", "zrry2eQp1_8", "gHkeMxKr1xM", "0M15ka97iVA", "czkI09RVRjA", "5SOziaQapMI", "kzsb9SlOqb0", "tuNhA-H1zbs", "_5zw4p257rY", "16LuBAskfOs", "xWxy9I5JGsM"]

    TRAIN_EPISODES = test_video_ids
    build_dataset_from_video_list(TRAIN_EPISODES, "boundary_112_five.csv")