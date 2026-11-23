import collections
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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

# Words that grammatically cannot end a speaker's turn unless they trailed off (pause >= 0.50s)
HANGING_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her",
    "its", "our", "their", "some", "any", "every", "no", "of", "in", "to", "for",
    "with", "on", "at", "from", "by", "about", "into", "through", "over", "after",
    "between", "without", "under", "within", "along", "and", "or", "but", "because",
    "if", "when", "while", "although", "since", "unless", "until", "whether",
    "have", "has", "had", "is", "was", "were", "are", "am", "been", "being",
    "will", "would", "could", "should", "can", "may", "might", "must",
    "do", "did", "does", "don't", "didn't", "doesn't", "wasn't", "weren't",
    "isn't", "aren't", "haven't", "hasn't", "hadn't", "won't", "wouldn't",
    "couldn't", "shouldn't", "can't", "i", "we", "they", "he", "she", "who",
    "very", "really", "so", "too", "quite", "rather", "almost", "just",
}

# Listener backchannels that can appear standalone or embedded inside unpunctuated segments
BACKCHANNEL_WORDS = {
    "yeah", "yep", "yup", "mm-hmm", "mhm", "uh-huh", "mm", "hmm",
    "right", "sure", "true", "totally", "exactly", "honestly", "oh", "okay", "ok",
}

# Multi-word conversational turn starters where unpunctuated speaker switches happen
TURN_STARTER_BIGRAMS = {
    ("do", "you"), ("did", "you"), ("have", "you"), ("are", "you"), ("were", "you"),
    ("would", "you"), ("can", "you"), ("could", "you"), ("what", "do"), ("what", "did"),
    ("what", "was"), ("what", "is"), ("what", "are"), ("what", "else"), ("how", "do"),
    ("how", "did"), ("how", "high"), ("how", "was"), ("why", "do"), ("why", "did"),
    ("why", "would"), ("well", "what"), ("well", "i"), ("well", "you"), ("well", "it"),
    ("well", "that"), ("well", "he"), ("well", "they"), ("yeah", "i"), ("yeah", "it"),
    ("yeah", "it's"), ("yeah", "that"), ("yeah", "that's"), ("yeah", "we"), ("yeah", "he"),
    ("yeah", "do"), ("no", "i"), ("no", "it"), ("oh", "i"), ("oh", "my"), ("oh", "yeah"),
    ("but", "i"), ("but", "what"), ("so", "what"), ("i", "think"), ("i", "mean"),
    ("i", "feel"), ("i", "don't"), ("i", "didn't"), ("i", "guess"), ("it", "was"),
    ("it", "must"), ("it's", "like"), ("that", "was"), ("that's", "like"), ("wait", "what"),
}

DISCOURSE_TAILS = {"know", "right", "honestly", "though", "something", "whatever", "even"}


def clean_token(w: str) -> str:
    return w.lower().strip(".,?!\"'():;-—…")


def has_terminal_punct(w: str) -> bool:
    s = w.strip()
    return s.endswith((".", "?", "!")) and not s.endswith("...")


def is_backchannel_phrase(words: List[Dict[str, Any]]) -> bool:
    if not words or len(words) > 4:
        return False
    tokens = [clean_token(w["word"]) for w in words if clean_token(w["word"])]
    return bool(tokens) and all(t in BACKCHANNEL_WORDS for t in tokens)


def detect_expected_hosts(meta: dict, available_hosts: list) -> list:
    text = f"{meta.get('title', '')}\n{meta.get('description', '')}".lower()
    found = [
        h for h in available_hosts
        if any(m in text for m in HOST_CHANNEL_MARKERS.get(h.lower(), [h.lower()]))
    ]
    return found if len(found) >= 2 else available_hosts


def extract_strict_embedding(
    model, audio_np: np.ndarray, s: float, e: float, sr: int = 16000
) -> Optional[np.ndarray]:
    dur = e - s
    if dur < 0.35:
        return None
    if dur > 8.0:
        mid = 0.5 * (s + e)
        s, e = mid - 4.0, mid + 4.0

    chunk = audio_np[max(0, int(s * sr)) : min(len(audio_np), int(e * sr))]
    if len(chunk) < int(0.35 * sr):
        return None

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


def build_episode_centroids(
    model,
    audio_np: np.ndarray,
    snippets: List[Dict[str, Any]],
    speaker_diagnostics: Dict[str, Any],
    global_centroids: Dict[str, np.ndarray],
    expected_hosts: List[str],
    sr: int = 16000,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Adapts host centroids to current episode mic EQ and builds a dynamic 'Guest'
    centroid whenever a true Guest cluster exists (like Ep 5).
    """
    cluster_to_initial: Dict[str, str] = {}
    guest_clusters = set()

    for spk, info in speaker_diagnostics.items():
        assigned = info.get("assigned", "Unknown")
        best_sim = float(info.get("best_cosine_sim", 0.0))
        total_sec = float(info.get("total_sec", 0.0))
        # True Guest cluster: >3 mins speaking time, low similarity (<0.56) to all hosts
        if assigned == "Guest" and total_sec >= 180.0 and best_sim < 0.56:
            cluster_to_initial[spk] = "Guest"
            guest_clusters.add(spk)
        else:
            cluster_to_initial[spk] = assigned

    active_speakers = list(expected_hosts)
    if guest_clusters:
        active_speakers.append("Guest")

    confident_vecs: Dict[str, List[np.ndarray]] = {s: [] for s in active_speakers}

    for seg in snippets:
        s, e = float(seg["start_seconds"]), float(seg["end_seconds"])
        dur = e - s
        if dur < 2.5:
            continue
        cluster = seg.get("speaker_cluster", "")
        v = extract_strict_embedding(model, audio_np, s, e, sr)
        if v is None:
            continue

        if cluster in guest_clusters:
            # Verify it doesn't strongly match an expected host before adding to Guest voiceprint
            host_scores = [float(np.dot(v, global_centroids[h])) for h in expected_hosts]
            if max(host_scores) < 0.52:
                confident_vecs["Guest"].append(v)
        else:
            scores = {h: float(np.dot(v, global_centroids[h])) for h in expected_hosts}
            sorted_h = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_h, best_s = sorted_h[0]
            runner_up_s = sorted_h[1][1] if len(sorted_h) > 1 else 0.0
            if best_s >= 0.54 and (best_s - runner_up_s) >= 0.10:
                confident_vecs[best_h].append(v)

    adapted: Dict[str, np.ndarray] = {}
    for h in expected_hosts:
        if len(confident_vecs[h]) >= 8:
            ep_mean = np.mean(confident_vecs[h], axis=0)
            ep_mean = ep_mean / (np.linalg.norm(ep_mean) + 1e-12)
            blended = 0.65 * ep_mean + 0.35 * global_centroids[h]
            adapted[h] = blended / (np.linalg.norm(blended) + 1e-12)
            print(f"  ├─ Calibrated '{h}' voiceprint from {len(confident_vecs[h])} episode turns")
        else:
            adapted[h] = global_centroids[h]

    if "Guest" in active_speakers and len(confident_vecs["Guest"]) >= 5:
        g_mean = np.mean(confident_vecs["Guest"], axis=0)
        adapted["Guest"] = g_mean / (np.linalg.norm(g_mean) + 1e-12)
        print(f"  ├─ Built live 'Guest' voiceprint from {len(confident_vecs['Guest'])} guest turns")

    return adapted, cluster_to_initial


def heal_syntactic_splits_and_sandwiches(
    snippets: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Re-joins sentences that were chopped mid-phrase (hanging determiner/preposition)
    or hijacked by a brief crosstalk sandwich (Speaker A -> Speaker B -> Speaker A).
    """
    if not snippets:
        return snippets, 0

    healed_count = 0
    segs = [dict(s, words=[dict(w) for w in s.get("words", [])]) for s in snippets]

    # Pass 1: Heal A -> B -> A mid-sentence crosstalk hijacks
    i = 1
    while i < len(segs) - 1:
        prev_s = segs[i - 1]
        curr_s = segs[i]
        next_s = segs[i + 1]

        spk_prev = prev_s["speaker"]
        spk_curr = curr_s["speaker"]
        spk_next = next_s["speaker"]

        gap_before = float(curr_s["start_seconds"]) - float(prev_s["end_seconds"])
        gap_after = float(next_s["start_seconds"]) - float(curr_s["end_seconds"])
        curr_dur = float(curr_s["end_seconds"]) - float(curr_s["start_seconds"])

        prev_words = prev_s.get("words", [])
        curr_words = curr_s.get("words", [])
        last_prev_word = prev_words[-1]["word"] if prev_words else ""
        last_prev_clean = clean_token(last_prev_word)

        # If A -> B -> A, B is NOT a backchannel ("mm-hmm", "yeah"), gaps are tight,
        # and A did not finish a punctuated sentence before B:
        if (
            spk_prev == spk_next
            and spk_curr != spk_prev
            and curr_dur <= 3.5
            and gap_before < 0.40
            and gap_after < 0.40
            and not is_backchannel_phrase(curr_words)
            and (not has_terminal_punct(last_prev_word) or last_prev_clean in HANGING_WORDS)
        ):
            # Merge prev + curr + next into a single uninterrupted sentence for spk_prev
            merged_words = prev_s.get("words", []) + curr_words + next_s.get("words", [])
            for w in merged_words:
                w["speaker"] = spk_prev
            prev_s["words"] = merged_words
            prev_s["end_seconds"] = next_s["end_seconds"]
            prev_s["duration"] = round(float(prev_s["end_seconds"]) - float(prev_s["start_seconds"]), 2)
            prev_s["text"] = " ".join(w["word"].strip() for w in merged_words).strip()
            segs.pop(i + 1)
            segs.pop(i)
            healed_count += 1
            continue
        i += 1

    # Pass 2: Heal hanging function words ("A" | "diverse group of assholes")
    i = 0
    while i < len(segs) - 1:
        curr_s = segs[i]
        next_s = segs[i + 1]
        curr_words = curr_s.get("words", [])
        next_words = next_s.get("words", [])
        if not curr_words or not next_words:
            i += 1
            continue

        last_w = curr_words[-1]["word"]
        last_clean = clean_token(last_w)
        gap = float(next_s["start_seconds"]) - float(curr_s["end_seconds"])

        # Hard syntactic hanger with no trail-off pause (< 0.45s) and not followed by a backchannel
        if (
            last_clean in HANGING_WORDS
            and not has_terminal_punct(last_w)
            and not last_w.strip().endswith("...")
            and gap < 0.45
            and not is_backchannel_phrase(next_words)
        ):
            dominant_spk = (
                curr_s["speaker"]
                if curr_s["duration"] >= next_s["duration"]
                else next_s["speaker"]
            )
            merged_words = curr_words + next_words
            for w in merged_words:
                w["speaker"] = dominant_spk
            curr_s["speaker"] = dominant_spk
            curr_s["words"] = merged_words
            curr_s["end_seconds"] = next_s["end_seconds"]
            curr_s["duration"] = round(float(curr_s["end_seconds"]) - float(curr_s["start_seconds"]), 2)
            curr_s["text"] = " ".join(w["word"].strip() for w in merged_words).strip()
            segs.pop(i + 1)
            healed_count += 1
            continue
        i += 1

    return segs, healed_count


def find_grammatical_candidate_splits(words: List[Dict[str, Any]]) -> List[int]:
    """
    Finds valid clause/turn boundary indices inside a word list while strictly
    forbidding splits right after hanging determiners, prepositions, or auxiliaries.
    """
    candidates = set()
    n = len(words)
    if n < 6:
        return []

    tokens = [clean_token(w["word"]) for w in words]

    for k in range(2, n - 2):
        prev_w = words[k - 1]["word"]
        prev_tok = tokens[k - 1]
        curr_tok = tokens[k]
        next_tok = tokens[k + 1] if k + 1 < n else ""
        gap = float(words[k]["start"]) - float(words[k - 1]["end"])

        # Never split immediately after a hanging function word unless speaker paused >= 0.55s
        if prev_tok in HANGING_WORDS and gap < 0.55:
            continue

        # 1. Punctuation boundary (. ? !)
        if has_terminal_punct(prev_w):
            candidates.add(k)
            continue

        # 2. Natural silence gap >= 0.26s
        if gap >= 0.26:
            candidates.add(k)
            continue

        # 3. Conversational turn-starter bigram ("do you", "well what", "it was", "that was", "yeah i")
        if (curr_tok, next_tok) in TURN_STARTER_BIGRAMS:
            candidates.add(k)
            continue

        # 4. After a discourse tail ("...you know", "...honestly", "...or something")
        if prev_tok in DISCOURSE_TAILS and (
            curr_tok in BACKCHANNEL_WORDS or (curr_tok, next_tok) in TURN_STARTER_BIGRAMS or gap >= 0.05
        ):
            candidates.add(k)
            continue

        # 5. Embedded backchannel boundary ("...insane | mm-hmm | just like...")
        if curr_tok in BACKCHANNEL_WORDS and prev_tok not in BACKCHANNEL_WORDS:
            candidates.add(k)
        if prev_tok in BACKCHANNEL_WORDS and curr_tok not in BACKCHANNEL_WORDS:
            candidates.add(k)

    return sorted(candidates)


def split_clause_runons(
    snippets: List[Dict[str, Any]],
    model,
    audio_np: np.ndarray,
    adapted_centroids: Dict[str, np.ndarray],
    sr: int = 16000,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Splits unpunctuated multi-speaker segments at grammatical clause boundaries
    when the voiceprint before vs. after switches speakers.
    """
    output_segments: List[Dict[str, Any]] = []
    split_count = 0
    speaker_pool = list(adapted_centroids.keys())

    for seg in snippets:
        queue = [seg]
        while queue:
            curr = queue.pop(0)
            words = curr.get("words", [])
            s_start = float(curr["start_seconds"])
            s_end = float(curr["end_seconds"])
            dur = s_end - s_start

            if dur < 3.2 or len(words) < 7:
                output_segments.append(curr)
                continue

            candidates = find_grammatical_candidate_splits(words)
            best_split_idx = None
            best_gain = 0.0

            for k in candidates:
                t_split = float(words[k]["start"])
                left_dur = float(words[k - 1]["end"]) - s_start
                right_dur = s_end - t_split

                # Allow shorter window (0.65s) if one side starts with a backchannel + turn
                if left_dur < 0.9 or right_dur < 0.9:
                    continue

                v_l = extract_strict_embedding(
                    model, audio_np, max(s_start, t_split - 3.0), float(words[k - 1]["end"]), sr
                )
                v_r = extract_strict_embedding(
                    model, audio_np, t_split, min(s_end, t_split + 3.0), sr
                )
                if v_l is None or v_r is None:
                    continue

                scores_l = {h: float(np.dot(v_l, adapted_centroids[h])) for h in speaker_pool}
                scores_r = {h: float(np.dot(v_r, adapted_centroids[h])) for h in speaker_pool}
                h_l = max(scores_l, key=scores_l.get)
                h_r = max(scores_r, key=scores_r.get)
                lr_sim = float(np.dot(v_l, v_r))

                if h_l != h_r and scores_l[h_l] >= 0.42 and scores_r[h_r] >= 0.42 and lr_sim < 0.58:
                    gain = (scores_l[h_l] - scores_l[h_r]) + (scores_r[h_r] - scores_r[h_l])
                    if gain >= 0.14 and gain > best_gain:
                        best_gain = gain
                        best_split_idx = k

            if best_split_idx is not None:
                left_w = words[:best_split_idx]
                right_w = words[best_split_idx:]
                l_start, l_end = float(left_w[0]["start"]), float(left_w[-1]["end"])
                r_start, r_end = float(right_w[0]["start"]), float(right_w[-1]["end"])

                output_segments.append({
                    **curr,
                    "start_seconds": round(l_start, 2),
                    "end_seconds": round(l_end, 2),
                    "duration": round(l_end - l_start, 2),
                    "text": " ".join(w["word"].strip() for w in left_w).strip(),
                    "words": left_w,
                })
                queue.insert(0, {
                    **curr,
                    "start_seconds": round(r_start, 2),
                    "end_seconds": round(r_end, 2),
                    "duration": round(r_end - r_start, 2),
                    "text": " ".join(w["word"].strip() for w in right_w).strip(),
                    "words": right_w,
                })
                split_count += 1
            else:
                output_segments.append(curr)

    return output_segments, split_count


def score_whole_clauses(
    snippets: List[Dict[str, Any]],
    model,
    audio_np: np.ndarray,
    adapted_centroids: Dict[str, np.ndarray],
    speaker_diagnostics: Dict[str, Any],
    cluster_to_initial: Dict[str, str],
    sr: int = 16000,
) -> List[Dict[str, Any]]:
    speaker_pool = list(adapted_centroids.keys())
    prev_speaker = speaker_pool[0]

    for seg in snippets:
        s = float(seg["start_seconds"])
        e = float(seg["end_seconds"])
        dur = max(0.0, e - s)
        raw_cluster = seg.get("speaker_cluster", "Unknown")
        cluster_default = cluster_to_initial.get(raw_cluster, seg.get("speaker", prev_speaker))
        is_merged = speaker_diagnostics.get(raw_cluster, {}).get("is_merged_cluster", False)

        v = extract_strict_embedding(model, audio_np, s, e, sr) if dur >= 0.45 else None

        if v is not None:
            scores = {h: float(np.dot(v, adapted_centroids[h])) for h in speaker_pool}
            # Give a modest stability prior (+0.05) to the Pyannote cluster default if it's a pure cluster
            if not is_merged and cluster_default in scores:
                scores[cluster_default] += 0.05
            best_h = max(scores, key=scores.get)
            if scores[best_h] >= 0.42:
                assigned = best_h
            else:
                assigned = cluster_default if cluster_default in speaker_pool else prev_speaker
        else:
            assigned = cluster_default if (not is_merged and cluster_default in speaker_pool) else prev_speaker

        seg["speaker"] = assigned
        prev_speaker = assigned

        h_int, rem = divmod(int(s), 3600)
        m_int, s_int = divmod(rem, 60)
        seg["timestamp"] = f"{h_int:02d}:{m_int:02d}:{s_int:02d}"
        for w in seg.get("words", []):
            w["speaker"] = assigned

    return snippets


def validate_episode(video_id: str, centroids_file: str = "speaker_centroids.json"):
    json_path = Path(f"diarized_transcripts/{video_id}.json")
    backup_path = Path(f"diarized_transcripts/{video_id}.pre_validator.json")
    wav_path = Path(f"audio_cache/{video_id}.wav")

    if not json_path.exists():
        print(f"⚠️ {json_path} not found, skipping.")
        return
    if not wav_path.exists():
        print(f"⚠️ {wav_path} not found in audio_cache/, cannot extract voiceprints.")
        return

    # Save backup of original output on first run so re-runs are always clean
    if not backup_path.exists():
        shutil.copy2(json_path, backup_path)
        print(f"📦 Backed up original transcript to {backup_path}")

    with open(backup_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(centroids_file, "r", encoding="utf-8") as f:
        c_data = json.load(f)

    raw_centroids = c_data.get("centroids", c_data)
    global_centroids = {
        k: np.asarray(v, dtype=np.float32) / (np.linalg.norm(v) + 1e-12)
        for k, v in raw_centroids.items()
    }
    expected_hosts = detect_expected_hosts(data.get("meta", {}), list(global_centroids.keys()))
    print(f"\n🔍 Validating {video_id} ({data.get('meta', {}).get('title', video_id)})")
    print(f"  ├─ Expected hosts: {expected_hosts}")

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

    # 1. Build episode-calibrated centroids (including dynamic Guest centroid if present)
    adapted_centroids, cluster_to_initial = build_episode_centroids(
        model,
        audio_np,
        data["snippets"],
        data.get("speaker_diagnostics", {}),
        global_centroids,
        expected_hosts,
        sr,
    )

    # 2. Heal mid-phrase syntactic splits and A -> B -> A crosstalk hijacks
    healed_snippets, heal_count_1 = heal_syntactic_splits_and_sandwiches(data["snippets"])

    # 3. Split unpunctuated multi-speaker run-ons at grammatical clause boundaries
    split_snippets, split_count = split_clause_runons(
        healed_snippets, model, audio_np, adapted_centroids, sr
    )

    # 4. Score whole grammatical clauses against episode centroids
    scored_snippets = score_whole_clauses(
        split_snippets,
        model,
        audio_np,
        adapted_centroids,
        data.get("speaker_diagnostics", {}),
        cluster_to_initial,
        sr,
    )

    # 5. Final pass to merge any newly adjacent identical-speaker crosstalk sandwiches
    final_snippets, heal_count_2 = heal_syntactic_splits_and_sandwiches(scored_snippets)

    # Update summary
    counts = collections.Counter(s["speaker"] for s in final_snippets)
    sec_totals = collections.defaultdict(float)
    for s in final_snippets:
        sec_totals[s["speaker"]] += float(s["duration"])

    data["snippets"] = final_snippets
    data["speaker_summary"] = {
        spk: {"segments": counts[spk], "minutes": round(sec_totals[spk] / 60.0, 1)}
        for spk in sorted(counts.keys())
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  ├─ Healed {heal_count_1 + heal_count_2} mid-phrase / crosstalk splits")
    print(f"  ├─ Split {split_count} unpunctuated multi-speaker clause turns")
    print(f"  └─ Updated speaker_summary: {json.dumps(data['speaker_summary'])}")


if __name__ == "__main__":
    vids = sys.argv[1:] if len(sys.argv) > 1 else ["YO9wkuvTsJE"]
    for v in vids:
        validate_episode(v)