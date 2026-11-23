# add_local_features.py
import pandas as pd
import numpy as np

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes differential and local window stats per video to catch semantic spikes."""
    processed_dfs = []
    
    # Process each video independently to avoid leaking across episodes
    for vid_id, group in df.groupby("video_id"):
        group = group.sort_values("timestamp_sec").copy()
        
        # 1. First-order difference (velocity of semantic change)
        group["cosine_dist_diff"] = group["cosine_dist"].diff().fillna(0.0)
        
        # 2. Local rolling mean & std (3-minute window assuming 10s stride = 18 steps)
        window_size = 18
        rolling_mean = group["cosine_dist"].rolling(window=window_size, center=True, min_periods=1).mean()
        rolling_std = group["cosine_dist"].rolling(window=window_size, center=True, min_periods=1).std().fillna(1e-5)
        
        # 3. Z-score: How extreme is this step relative to its immediate neighborhood?
        group["cosine_dist_zscore"] = (group["cosine_dist"] - rolling_mean) / (rolling_std + 1e-5)
        
        # 4. Context gap delta
        group["gap_duration_diff"] = group["gap_duration"].diff().fillna(0.0)
        
        processed_dfs.append(group)
        
    return pd.concat(processed_dfs, ignore_index=True)

if __name__ == "__main__":
    print("Enriching boundary_dataset.csv with rolling local features...")
    raw_df = pd.read_csv("boundary_dataset.csv")
    enriched_df = compute_rolling_features(raw_df)
    enriched_df.to_csv("boundary_dataset_enriched.csv", index=False)
    print(f"✅ Saved enriched dataset with {len(enriched_df)} rows.")