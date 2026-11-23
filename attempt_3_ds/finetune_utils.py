import pandas as pd
import numpy as np

def coalesce_clusters_to_midpoint(
    video_df: pd.DataFrame, 
    prob_col: str = "prob", 
    threshold: float = 0.6,
    min_cluster_size: int = 3,
    max_gap_sec: float = 15.0
) -> pd.DataFrame:
    """
    1. Filters out transient model firings (clusters < min_cluster_size).
    2. Coalesces valid contiguous firings to their exact temporal midpoint.
    """
    df = video_df.sort_values("timestamp_sec").reset_index(drop=True)
    
    # 1. Filter hits above probability threshold
    hits = df[df[prob_col] >= threshold].copy()
    if hits.empty:
        return pd.DataFrame(columns=df.columns)

    timestamps = hits["timestamp_sec"].values
    indices = hits.index.values

    clusters = []
    current_cluster = [0]

    # 2. Group adjacent timestamps within max_gap_sec
    for i in range(1, len(hits)):
        if timestamps[i] - timestamps[i - 1] <= max_gap_sec:
            current_cluster.append(i)
        else:
            clusters.append(current_cluster)
            current_cluster = [i]
    if current_cluster:
        clusters.append(current_cluster)

    coalesced_rows = []

    # 3. Filter size and compute midpoints
    for cluster_indices in clusters:
        cluster_len = len(cluster_indices)
        
        # Rule 1: Reject small transient clusters
        if cluster_len < min_cluster_size:
            continue
            
        # Rule 2: Coalesce to spatial midpoint index
        midpoint_idx_within_cluster = cluster_len // 2
        midpoint_hit_idx = cluster_indices[midpoint_idx_within_cluster]
        
        # Extract row corresponding to the midpoint candidate
        midpoint_row = hits.loc[indices[midpoint_hit_idx]].copy()
        
        # Compute exact time midpoint across the entire cluster span
        start_time = timestamps[cluster_indices[0]]
        end_time = timestamps[cluster_indices[-1]]
        midpoint_row["coalesced_timestamp_sec"] = (start_time + end_time) / 2.0
        midpoint_row["cluster_size"] = cluster_len
        
        coalesced_rows.append(midpoint_row)

    if not coalesced_rows:
        return pd.DataFrame(columns=df.columns)

    return pd.DataFrame(coalesced_rows)