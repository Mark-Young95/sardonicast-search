import numpy as np
import pandas as pd

def refine_isolated_sandwiches(
    video_df: pd.DataFrame, 
    prob_col: str = "prob", 
    threshold: float = 0.5,
    stride_sec: float = 10.0
) -> pd.DataFrame:
    """
    Enforces isolated sandwich logic:
    If timestamp k and k + 2*stride are boundaries, but k-3, k-2, k-1, 
    k+3, k+4, k+5 are NOT, interpolate to mark k+1 as the boundary.
    """
    df = video_df.sort_values("timestamp_sec").reset_index(drop=True)
    probs = df[prob_col].values
    timestamps = df["timestamp_sec"].values
    
    binary_hits = (probs >= threshold).astype(int)
    n = len(binary_hits)
    
    refined_indices = []
    skip_until = -1

    for i in range(3, n - 5):
        if i < skip_until:
            continue

        # Check neighborhood context: k-3, k-2, k-1 MUST be 0
        left_context_clean = (binary_hits[i-3] == 0 and binary_hits[i-2] == 0 and binary_hits[i-1] == 0)
        
        if not left_context_clean:
            continue

        # Pattern A: Sandwich [1, ?, 1] at k, k+1, k+2
        is_sandwich = (binary_hits[i] == 1 and binary_hits[i+2] == 1)
        # Right context check: k+3, k+4, k+5 MUST be 0
        right_context_clean = (binary_hits[i+3] == 0 and binary_hits[i+4] == 0 and binary_hits[i+5] == 0)

        if is_sandwich and right_context_clean:
            # Mark interpolated midpoint (k+1) as the refined boundary
            refined_indices.append(i + 1)
            skip_until = i + 6  # Jump past the evaluated window
            continue

        # Pattern B: Single Isolated Spike [0, 0, 0, 1, 0, 0, 0]
        is_single_spike = (binary_hits[i] == 1 and binary_hits[i+1] == 0)
        right_context_single_clean = (binary_hits[i+1] == 0 and binary_hits[i+2] == 0 and binary_hits[i+3] == 0)
        
        if is_single_spike and right_context_single_clean:
            refined_indices.append(i)
            skip_until = i + 4

    return df.iloc[refined_indices].copy()