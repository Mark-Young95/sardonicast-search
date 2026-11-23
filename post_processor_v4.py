# post_processor.py
import pandas as pd
import numpy as np

def extract_cluster_timestamp(cluster_df: pd.DataFrame, is_outro: bool = False) -> float:
    """
    Percentile Shoulder Alignment:
    - Outro: 33th percentile (Left shoulder ramp-up)
    - Standard/Q&A Shifts: 45th percentile (Right shoulder transition completion)
    """
    cluster_df = cluster_df.sort_values("timestamp_sec").reset_index(drop=True)
    n = len(cluster_df)
    if n == 1:
        return float(cluster_df.iloc[0]["timestamp_sec"])
    idx = int(np.floor(n * 0.33)) if is_outro else int(np.floor(n * 0.45))
    return float(cluster_df.iloc[idx]["timestamp_sec"])

def enforce_min_spacing(candidates: list, ep_duration_sec: float, min_gap_sec: float = 80.0) -> list:
    """
    True Non-Maximum Suppression (NMS).
    Exempts candidates if BOTH are in the terminal 10-minute outro zone, 
    letting downstream rules handle them.
    """
    if not candidates:
        return candidates
        
    sorted_cands = sorted(
        candidates, 
        key=lambda x: (x["is_outro"], x["prob"]), 
        reverse=True
    )
    
    filtered = []
    for cand in sorted_cands:
        too_close = False
        for kept in filtered:
            # Check if BOTH the candidate and the kept peak are in the final 10 mins (600s)
            cand_in_outro = (ep_duration_sec - cand["timestamp_sec"]) <= 600
            kept_in_outro = (ep_duration_sec - kept["timestamp_sec"]) <= 600
            
            # If they are both in the outro zone, ignore the 80s moat!
            if cand_in_outro and kept_in_outro:
                continue
                
            # Otherwise, apply standard 80s suppression
            if abs(cand["timestamp_sec"] - kept["timestamp_sec"]) < min_gap_sec:
                too_close = True
                break
        
        if not too_close:
            filtered.append(cand)
            
    return sorted(filtered, key=lambda x: x["timestamp_sec"])

def enforce_window_limits(candidates: list, ep_duration_sec: float) -> list:
    """Enforces max 3 boundaries per 6 mins (360s) and max 5 per 18 mins (1080s)."""
    if not candidates:
        return candidates
        
    candidates = sorted(candidates, key=lambda x: x["timestamp_sec"])
    keep_mask = [True] * len(candidates)
    
    for i in range(len(candidates)):
        if not keep_mask[i]: continue
            
        t_start = candidates[i]["timestamp_sec"]
        
        # Check 6-minute window (Max 3)
        window_6m = [j for j in range(i, len(candidates)) if keep_mask[j] and (candidates[j]["timestamp_sec"] - t_start) <= 360]
        while len(window_6m) > 3 or (len(window_6m) == 3 and ep_duration_sec - candidates[window_6m[-1]]["timestamp_sec"] <= 1200):
            # Identify the lowest probability non-outro candidate to drop
            drop_idx = sorted(window_6m, key=lambda idx: (candidates[idx]["is_outro"], candidates[idx]["prob"]))[0]
            keep_mask[drop_idx] = False
            window_6m.remove(drop_idx)
            
        # Check 18-minute window (Max 5)
        window_18m = [j for j in range(i, len(candidates)) if keep_mask[j] and (candidates[j]["timestamp_sec"] - t_start) <= 1080]
        while len(window_18m) > 5 or (len(window_18m) > 3 and ep_duration_sec - candidates[window_18m[-1]]["timestamp_sec"] <= 1200):
            drop_idx = sorted(window_18m, key=lambda idx: (candidates[idx]["is_outro"], candidates[idx]["prob"]))[0]
            keep_mask[drop_idx] = False
            window_18m.remove(drop_idx)
            
    return [c for idx, c in enumerate(candidates) if keep_mask[idx]]

def resolve_double_outros(candidates: list, ep_duration_sec: float, outro_window_sec: float = 600.0) -> list:
    """
    Evaluates boundaries in the last 10 minutes. 
    Only triggers the showdown logic if there are EXACTLY two predictions in this zone.
    """
    if len(candidates) < 2:
        return candidates
        
    # Find all candidates that fall in the terminal outro window (last 10 mins)
    outro_cands = [c for c in candidates if (ep_duration_sec - c["timestamp_sec"]) <= outro_window_sec]
    
    # Only apply this specific tie-breaker if there are EXACTLY two outro candidates
    if len(outro_cands) == 2:
        c1 = outro_cands[0]
        c2 = outro_cands[1]
        
        print(f"  [DEBUG] Exact double outro detected! c1: {c1['prob']:.4f}, c2: {c2['prob']:.4f}")
        
        # Rule 1: if 2nd is +0.15 prob of 1st, then pick 2nd.
        if c1["prob"] + 0.15 <= c2["prob"]:
            print("  [DEBUG] Keep 2nd, dropping 1st.")
            # Keep 2nd, drop 1st
            candidates = [c for c in candidates if c["timestamp_sec"] != c1["timestamp_sec"]]
        # Rule 2: If 1st is higher probability OR extremely confident (>= 0.700)
        elif c1["prob"] > c2["prob"] or c1["prob"] >= 0.700:
            print("  [DEBUG] Keep 1st, dropping 2nd.")
            # Keep 1st, drop 2nd
            candidates = [c for c in candidates if c["timestamp_sec"] != c2["timestamp_sec"]]
        else:
            print("  [DEBUG] Keep 2nd, dropping 1st.")
            # Keep 2nd, drop 1st
            candidates = [c for c in candidates if c["timestamp_sec"] != c1["timestamp_sec"]]
    else:
        if len(outro_cands) > 2:
            print(f"  [DEBUG] Skipping double-outro logic: Found {len(outro_cands)} boundaries in the outro zone.")
            
    return candidates

def process_episode_predictions(
    df: pd.DataFrame, 
    prob_col: str = "prob", 
    high_threshold: float = 0.57,  
    low_threshold: float = 0.545,
    max_chapters: int = 12,
    max_gap_sec: float = 20.0,
    outro_window_sec: float = 600.0,
    apply_rating_boost: bool = False  # <--- NEW TOGGLE
) -> pd.DataFrame:
    
    if df.empty or prob_col not in df.columns:
        return pd.DataFrame(columns=["timestamp_sec", "prob", "boosted_prob", "is_outro"])

    df = df.copy()

    # Apply the boost ONLY if the toggle is True
    if apply_rating_boost and "rating_rolling_5min" in df.columns:
        df["boosted_prob"] = df.apply(
            lambda row: min(0.99, row[prob_col] + 0.15) if row["rating_rolling_5min"] > 0 else row[prob_col], 
            axis=1
        )
    else:
        df["boosted_prob"] = df[prob_col]

    ep_duration_sec = df["timestamp_sec"].max()
    
    # Notice we now filter and sort using the BOOSTED probability
    active_points = df[df["boosted_prob"] >= low_threshold].sort_values("timestamp_sec")
    
    if active_points.empty:
        return pd.DataFrame(columns=["timestamp_sec", "prob", "boosted_prob", "is_outro"])

    # 1. Spatial Grouping (Hysteresis with Strict Trough Breaking)
    clusters = []
    current_cluster = []
    
    # Iterate through the entire timeline chronologically, NO pre-filtering
    for _, row in df.sort_values("timestamp_sec").iterrows():
        if row[prob_col] >= low_threshold:
            if not current_cluster:
                current_cluster.append(row)
            else:
                # If there's a massive temporal gap (e.g. missing data), break cluster
                if row["timestamp_sec"] - current_cluster[-1]["timestamp_sec"] > max_gap_sec:
                    clusters.append(pd.DataFrame(current_cluster))
                    current_cluster = [row]
                else:
                    current_cluster.append(row)
        else:
            # TROUGH DETECTED (< low_threshold). Instantly close any active cluster.
            if current_cluster:
                clusters.append(pd.DataFrame(current_cluster))
                current_cluster = []
                
    if current_cluster:
        clusters.append(pd.DataFrame(current_cluster))

    # 2. Identify Terminal Outro Cluster
    outro_candidates = [c for c in clusters if (ep_duration_sec - c["timestamp_sec"].max()) <= outro_window_sec]
    terminal_outro_cluster_id = None
    if outro_candidates:
        # Find the HIGHEST probability peak in the outro window, not just the last one
        true_outro_cluster = max(outro_candidates, key=lambda c: c[prob_col].max())
        terminal_outro_cluster_id = id(true_outro_cluster)

    # 3. Candidate Extraction (Using the Boosted Probabilities!)
    candidates = []
    for cluster_df in clusters:
        # Get the peak of the boosted curve
        max_boosted_prob = cluster_df["boosted_prob"].max()
        raw_prob_at_peak = cluster_df.loc[cluster_df["boosted_prob"].idxmax()][prob_col]
        is_outro = (id(cluster_df) == terminal_outro_cluster_id)
        
        if max_boosted_prob >= high_threshold or is_outro:
            ts = extract_cluster_timestamp(cluster_df, is_outro=is_outro)
            candidates.append({
                "timestamp_sec": ts, 
                "prob": raw_prob_at_peak,
                "boosted_prob": max_boosted_prob,
                "is_outro": is_outro
            })

    # 4. Enforce True Minimum Spacing (80s Rule with NMS) - Run Density Limits BEFORE Max Chapters
    candidates = enforce_min_spacing(candidates, ep_duration_sec, min_gap_sec=80.0)
    candidates = enforce_window_limits(candidates, ep_duration_sec) # <-- New multi-tier weeding
    
    # 5. Priority Whittling Down
    candidates = sorted(candidates, key=lambda x: (x["is_outro"], x["boosted_prob"]), reverse=True)[:max_chapters]

    # Re-sort chronologically for the final output
    candidates = sorted(candidates, key=lambda x: x["timestamp_sec"])

    # 6. NEW: Resolve Double Outros (The Final Tie-Breaker)
    candidates = resolve_double_outros(candidates, ep_duration_sec, outro_window_sec)

    pred_df = pd.DataFrame(sorted(candidates, key=lambda x: x["timestamp_sec"]))

    # LABELING QA AND OUTRO BOUNDARIES

    if pred_df.empty:
        return pred_df

    cols_to_merge = ["timestamp_sec", "ep_duration_sec"]
    if "qa_density" in df.columns: cols_to_merge.append("qa_density")
    if "outro_density" in df.columns: cols_to_merge.append("outro_density")
    if "rel_position" in df.columns: cols_to_merge.append("rel_position")
    
    pred_df = pd.merge(pred_df, df[cols_to_merge], on="timestamp_sec", how="left")
    pred_df["label"] = "chapter"
    
    # 3. Label Q&A Boundary (Search 50% to 90% window)
    qa_assigned = False
    if "qa_density" in pred_df.columns:
        
        # Step A: Isolate all predicted boundaries in the 50% - 90% time window
        window_mask = (pred_df["rel_position"] >= 0.50) & (pred_df["rel_position"] <= 0.90)
        window_candidates = pred_df[window_mask]
        print(window_candidates[["timestamp_sec", "prob"]].to_markdown(index=False))
        
        if not window_candidates.empty:
            # Step B: Check which of these actually have Q&A keywords
            keyword_candidates = window_candidates[window_candidates["qa_density"] > 0]
            
            if not keyword_candidates.empty:
                # Primary Route: Highest density in the window
                best_idx = keyword_candidates["qa_density"].idxmax()
                pred_df.at[best_idx, "label"] = "qa_boundary"
                qa_assigned = True
                ts = pred_df.at[best_idx, "timestamp_sec"]
                print(f"  ✅ 'qa_boundary' identified by density at {ts}s (Density: {pred_df.at[best_idx, 'qa_density']:.4f})")
            
            else:
                # Primary Fallback: No keywords, so grab the LATEST boundary inside the 50-90% window
                # (Since pred_df is sorted chronologically, .index[-1] grabs the latest one)
                fallback_idx = window_candidates.index[-1]
                pred_df.at[fallback_idx, "label"] = "qa_boundary"
                qa_assigned = True
                ts = pred_df.at[fallback_idx, "timestamp_sec"]
                print(f"  ⚠️ Warning: No keywords found. Fell back to latest boundary in 50-90% window at {ts}s")
                
        if not qa_assigned:
            # Only fallback to index[-2] if that boundary is actually in the second half (>= 45%)!
            qa_idx = pred_df.index[-2] if len(pred_df) >= 2 else pred_df.index[-1]
            if pred_df.at[qa_idx, "rel_position"] >= 0.45:
                pred_df.at[qa_idx, "label"] = "qa_boundary"
                ts = pred_df.at[qa_idx, "timestamp_sec"]
                print(f"  🚨 Critical: No boundaries exist in 50-90% window! Fell back to position {ts}s")
            else:
                print("  ⚠️ Skipped Q&A fallback because candidate is in the first half (< 45%) of episode.")

    # 4. Label Outro Boundary (Search highest to lowest)
    outro_assigned = False
    if "outro_density" in pred_df.columns:
        # sorted_outro = pred_df["outro_density"].sort_values(ascending=False).index
        
        # for idx in sorted_outro:
        #     if pred_df.at[idx, "outro_density"] == 0:
        #         break
                
        #     # Prevent Outro from stealing the Q&A boundary
        #     if pred_df.at[idx, "label"] == "qa_boundary" and len(pred_df) > 1:
        #         continue
                
        #     ep_dur = pred_df.at[idx, "ep_duration_sec"]
        #     ts = pred_df.at[idx, "timestamp_sec"]
            
        #     # Validation: Must be in the final 10 minutes (600s)
        #     if (ep_dur - ts) <= 600:
        #         pred_df.at[idx, "label"] = "outro_boundary"
        #         outro_assigned = True
        #         print(f"'outro_boundary' identified at {ts}s ")
        #         break

        sorted_outro = pred_df["rel_position"].sort_values(ascending=False).index
        for idx in sorted_outro:
            if pred_df.at[idx, "rel_position"] < 0.92:
                break
                
            # Prevent Outro from stealing the Q&A boundary
            if pred_df.at[idx, "label"] == "qa_boundary" and len(pred_df) > 1:
                break
                
            ep_dur = pred_df.at[idx, "ep_duration_sec"]
            ts = pred_df.at[idx, "timestamp_sec"]
            
            # Validation: Must be in the final 10 minutes (600s)
            if (ep_dur - ts) <= 600:
                pred_df.at[idx, "label"] = "outro_boundary"
                outro_assigned = True
                print(f"'outro_boundary' identified at {ts}s ")
                break
                
        if not outro_assigned:
            # Fallback to final chapter
            outro_idx = pred_df.index[-1]
            if pred_df.at[outro_idx, "label"] == "qa_boundary" and len(pred_df) > 1:
                outro_idx = pred_df.index[-2] # Shift if collision
                
            pred_df.at[outro_idx, "label"] = "outro_boundary"
            ep_dur = pred_df.at[outro_idx, "ep_duration_sec"]
            ts = pred_df.at[outro_idx, "timestamp_sec"]
            print(f"  ⚠️ Warning: No valid 'outro_boundary' density peaks in final 10 mins. Fell back to position {ts}s")

    # Cleanup temporary merge columns
    pred_df = pred_df.drop(columns=[c for c in cols_to_merge if c != "timestamp_sec"])
    return pred_df