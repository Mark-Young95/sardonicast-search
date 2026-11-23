import pandas as pd

def enforce_min_spacing(candidates: list, min_gap_sec: float = 45.0) -> list:
    """True NMS tuned for rapid Q&A questions."""
    if not candidates:
        return candidates
        
    # Sort strictly by highest probability
    sorted_cands = sorted(candidates, key=lambda x: x["prob"], reverse=True)
    
    filtered = []
    for cand in sorted_cands:
        too_close = False
        for kept in filtered:
            if abs(cand["timestamp_sec"] - kept["timestamp_sec"]) < min_gap_sec:
                too_close = True
                break
        if not too_close:
            filtered.append(cand)
            
    # Return chronologically
    return sorted(filtered, key=lambda x: x["timestamp_sec"])

def process_qa_predictions(
    df: pd.DataFrame, 
    prob_col: str = "prob", 
    high_threshold: float = 0.60,  
    low_threshold: float = 0.55,
    max_questions: int = 20,
    max_gap_sec: float = 15.0 # Tighter cluster grouping for Q&A
) -> pd.DataFrame:
    
    if df.empty or prob_col not in df.columns:
        return pd.DataFrame(columns=["timestamp_sec", "prob"])

    df = df.copy()
    
    # 1. Spatial Grouping (Trough Breaking)
    clusters = []
    current_cluster = []
    
    for _, row in df.sort_values("timestamp_sec").iterrows():
        if row[prob_col] >= low_threshold:
            if not current_cluster:
                current_cluster.append(row)
            else:
                if row["timestamp_sec"] - current_cluster[-1]["timestamp_sec"] > max_gap_sec:
                    clusters.append(pd.DataFrame(current_cluster))
                    current_cluster = [row]
                else:
                    current_cluster.append(row)
        else:
            if current_cluster:
                clusters.append(pd.DataFrame(current_cluster))
                current_cluster = []
                
    if current_cluster:
        clusters.append(pd.DataFrame(current_cluster))

    # 2. Candidate Extraction (40th Percentile of Probability Mass)
    candidates = []
    for cluster_df in clusters:
        max_prob = cluster_df[prob_col].max()
        if max_prob >= high_threshold:
            # Normalize probabilities so they sum to 1.0
            weights = cluster_df[prob_col]
            norm_weights = weights / weights.sum()
            
            # Calculate cumulative sum (CDF)
            cdf = norm_weights.cumsum()
            
            # Find the exact row where we cross the 40% threshold
            idx_40th = cdf[cdf >= 0.40].index[0]
            final_ts = cluster_df.loc[idx_40th, "timestamp_sec"]
            
            candidates.append({
                "timestamp_sec": round(final_ts, 2), 
                "prob": max_prob
            })

    # 3. Apply 40-Second Moat
    candidates = enforce_min_spacing(candidates, min_gap_sec=40.0)

    # 4. Cap at max questions (just in case)
    candidates = sorted(candidates, key=lambda x: x["prob"], reverse=True)[:max_questions]
    return pd.DataFrame(sorted(candidates, key=lambda x: x["timestamp_sec"]))