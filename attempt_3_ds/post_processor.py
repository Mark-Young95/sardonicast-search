# post_processor.py
import pandas as pd
import numpy as np

def extract_cluster_timestamp(cluster_df: pd.DataFrame, is_outro: bool = False) -> float:
    """
    Percentile Shoulder Alignment:
    - Outro: 25th percentile (Left shoulder ramp-up)
    - Standard/Q&A Shifts: 75th percentile (Right shoulder transition completion)
    """
    cluster_df = cluster_df.sort_values("timestamp_sec").reset_index(drop=True)
    n = len(cluster_df)
    
    if n == 1:
        return float(cluster_df.iloc[0]["timestamp_sec"])
    
    idx = int(np.floor(n * 0.25)) if is_outro else int(np.floor(n * 0.75))
    return float(cluster_df.iloc[idx]["timestamp_sec"])


def enforce_window_density(candidates: list, max_per_window: int = 3, window_sec: float = 900.0) -> list:
    """
    Ensures there are no more than 3 boundaries in any 15-minute (900s) window.
    Removes the lowest probability candidate inside overflowing windows.
    """
    if len(candidates) <= max_per_window:
        return candidates

    candidates = sorted(candidates, key=lambda x: x["timestamp_sec"])
    keep_mask = [True] * len(candidates)

    for i in range(len(candidates)):
        if not keep_mask[i]:
            continue
            
        t_start = candidates[i]["timestamp_sec"]
        t_end = t_start + window_sec
        
        # Find all active candidates in this 15-minute window
        window_indices = [
            j for j in range(i, len(candidates)) 
            if keep_mask[j] and candidates[j]["timestamp_sec"] <= t_end
        ]

        # If density exceeded, drop lowest probability non-outro item in window
        while len(window_indices) > max_per_window:
            # Sort window candidates by probability ascending, protecting outros
            window_indices_by_prob = sorted(
                window_indices, 
                key=lambda idx: (candidates[idx]["is_outro"], candidates[idx]["boosted_prob"])
            )
            drop_idx = window_indices_by_prob[0]
            keep_mask[drop_idx] = False
            window_indices.remove(drop_idx)

    return [cand for idx, cand in enumerate(candidates) if keep_mask[idx]]


def enforce_min_spacing(candidates: list, min_gap_sec: float = 120.0, ep_duration_sec: float = 0.0) -> list:
    """
    Suppresses candidates closer than min_gap_sec (120s), except terminal outros.
    """
    if not candidates:
        return candidates
        
    filtered = []
    candidates = sorted(candidates, key=lambda x: x["timestamp_sec"])
    
    for cand in candidates:
        if not filtered:
            filtered.append(cand)
            continue
            
        prev_time = filtered[-1]["timestamp_sec"]
        curr_time = cand["timestamp_sec"]
        is_terminal_outro = cand["is_outro"]
        
        if (curr_time - prev_time) >= min_gap_sec or is_terminal_outro:
            filtered.append(cand)
            
    return filtered


def process_episode_predictions(
    df: pd.DataFrame, 
    prob_col: str = "prob", 
    high_threshold: float = 0.60, 
    low_threshold: float = 0.45,
    max_chapters: int = 8,
    max_gap_sec: float = 20.0,
    outro_window_sec: float = 600.0
) -> pd.DataFrame:
    """
    Master Stage 2 Engine with Priority Retention, Keyword Boosts, and Window Density Limits.
    """
    if df.empty or prob_col not in df.columns:
        return pd.DataFrame(columns=["timestamp_sec", "prob", "boosted_prob", "is_outro", "is_qa"])

    ep_duration_sec = df["timestamp_sec"].max()
    active_points = df[df[prob_col] >= low_threshold].sort_values("timestamp_sec").copy()
    
    if active_points.empty:
        return pd.DataFrame(columns=["timestamp_sec", "prob", "boosted_prob", "is_outro", "is_qa"])

    # 1. Spatial Grouping
    clusters = []
    current_cluster = []

    for _, row in active_points.iterrows():
        if not current_cluster:
            current_cluster.append(row)
        else:
            if row["timestamp_sec"] - current_cluster[-1]["timestamp_sec"] <= max_gap_sec:
                current_cluster.append(row)
            else:
                clusters.append(pd.DataFrame(current_cluster))
                current_cluster = [row]
                
    if current_cluster:
        clusters.append(pd.DataFrame(current_cluster))

    # 2. Identify Terminal Outro Cluster
    terminal_outro_cluster_id = None
    for c in reversed(clusters):
        c_max_t = c["timestamp_sec"].max()
        if (ep_duration_sec - c_max_t) <= outro_window_sec:
            terminal_outro_cluster_id = id(c)
            break

    # 3. Candidate Scoring & Keyword Boosts
    candidates = []
    for cluster_df in clusters:
        raw_max_p = cluster_df[prob_col].max()
        boosted_p = raw_max_p
        
        is_outro = (id(cluster_df) == terminal_outro_cluster_id)
        
        qa_hits = cluster_df["has_qa_kw"].sum() if "has_qa_kw" in cluster_df.columns else 0
        transition_hits = cluster_df["has_transition_kw"].sum() if "has_transition_kw" in cluster_df.columns else 0
        
        is_qa = qa_hits >= 2
        
        # Apply Keyword Boosts (+0.15) for Q&A, Transitions, or Terminal Outros
        if is_qa or transition_hits >= 2:
            boosted_p += 0.15
        if is_outro:
            boosted_p += 0.20

        # Include candidate if boosted probability hits high_threshold
        if boosted_p >= high_threshold:
            ts = extract_cluster_timestamp(cluster_df, is_outro=is_outro)
            candidates.append({
                "timestamp_sec": ts, 
                "prob": raw_max_p,
                "boosted_prob": boosted_p,
                "is_outro": is_outro,
                "is_qa": is_qa
            })

    if not candidates:
        return pd.DataFrame(columns=["timestamp_sec", "prob", "boosted_prob", "is_outro", "is_qa"])

    # 4. Enforce Minimum Spacing (120s Rule)
    candidates = enforce_min_spacing(candidates, min_gap_sec=120.0, ep_duration_sec=ep_duration_sec)

    # 5. Enforce Window Density Limit (Max 3 in any 15-minute window)
    candidates = enforce_window_density(candidates, max_per_window=3, window_sec=900.0)

    # 6. Priority Whittling Down to max_chapters (8)
    # Sort key order: (Outro First, Q&A Second, then Highest Boosted Probability)
    candidates = sorted(
        candidates, 
        key=lambda x: (x["is_outro"], x["is_qa"], x["boosted_prob"]), 
        reverse=True
    )[:max_chapters]

    # Re-sort into chronological order for output
    candidates = sorted(candidates, key=lambda x: x["timestamp_sec"])
    
    return pd.DataFrame(candidates)