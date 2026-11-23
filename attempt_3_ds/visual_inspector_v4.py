# visual_inspector_v2.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from post_processor_v4 import process_episode_predictions

def plot_episode_comparison(
    video_id: str,
    df: pd.DataFrame,
    model,
    features: list,
    high_threshold: float = 0.53,
    low_threshold: float = 0.49,
    max_chapters: int = 10
):
    ep_df = df[df["video_id"] == video_id].sort_values("timestamp_sec").copy()
    if ep_df.empty:
        return

    # 1. Model Predicts Probabilities
    ep_df["prob"] = model.predict_proba(ep_df[features])[:, 1]
    
    timestamps_min = ep_df["timestamp_sec"].values / 60.0
    raw_probs = ep_df["prob"].values
    gt_boundaries_min = ep_df[ep_df["is_boundary"] == 1]["timestamp_sec"].values / 60.0
    ratings_sensor = ep_df["rating_rolling_5min"].values if "rating_rolling_5min" in ep_df.columns else np.zeros_like(raw_probs)

    # 2. Get Predictions for BOTH Scenarios
    pred_no_boost = process_episode_predictions(
        ep_df, prob_col="prob", high_threshold=high_threshold, 
        low_threshold=low_threshold, max_chapters=max_chapters, apply_rating_boost=False
    )
    
    pred_with_boost = process_episode_predictions(
        ep_df, prob_col="prob", high_threshold=high_threshold, 
        low_threshold=low_threshold, max_chapters=max_chapters, apply_rating_boost=False
    )
    boosted_probs = np.where(ratings_sensor > 0, np.minimum(0.99, raw_probs + 0.15), raw_probs)

    # 3. Setup the Dual-Plot Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    tol_min = 45.0 / 60.0

    # ==========================================
    # GRAPH 1: NO BOOST (Pure XGBoost)
    # ==========================================
    ax1.set_title(f"WITHOUT Ratings Boost (Pure XGBoost): {video_id}", fontsize=13)
    ax1.plot(timestamps_min, raw_probs, label="Raw XGBoost", color="#1f77b4", linewidth=1.5, alpha=0.85)
    ax1.axhline(y=high_threshold, color="red", linestyle="-", alpha=0.7, label=f"Peak ({high_threshold})")
    ax1.axhline(y=low_threshold, color="red", linestyle=":", alpha=0.5, label=f"Shoulder ({low_threshold})")
    
    for idx, gt_t in enumerate(gt_boundaries_min):
        ax1.axvspan(gt_t - tol_min, gt_t + tol_min, color="#2ca02c", alpha=0.18, label="Ground Truth" if idx == 0 else "")
        ax1.axvline(x=gt_t, color="#2ca02c", linestyle=":", alpha=0.8)

    if not pred_no_boost.empty:
        for idx, pred_t in enumerate(pred_no_boost["timestamp_sec"].values / 60.0):
            ax1.axvline(x=pred_t, color="#9467bd", linestyle="-", linewidth=2.5, zorder=5, label="Final Prediction" if idx == 0 else "")
            ax1.scatter([pred_t], [1.02], color="#9467bd", marker="v", s=100, zorder=6)
    
    ax1.set_ylim(-0.05, 1.3)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.95)

    # ==========================================
    # GRAPH 2: WITH BOOST (+0.15)
    # ==========================================
    # ax2.set_title(f"WITH Ratings Boost (+0.15): {video_id}", fontsize=13)
    # ax2.fill_between(timestamps_min, -0.05, 1.3, where=(ratings_sensor > 0), color='#ffcc00', alpha=0.15, label="Ratings Active (5m window)")
    
    # ax2.plot(timestamps_min, raw_probs, label="Raw XGBoost", color="#1f77b4", linestyle="--", linewidth=1.5, alpha=0.6)
    # ax2.plot(timestamps_min, boosted_probs, label="Boosted Prob", color="#ff7f0e", linewidth=2.0, alpha=0.9)
    
    # ax2.axhline(y=high_threshold, color="red", linestyle="-", alpha=0.7)
    # ax2.axhline(y=low_threshold, color="red", linestyle=":", alpha=0.5)
    
    # for idx, gt_t in enumerate(gt_boundaries_min):
    #     ax2.axvspan(gt_t - tol_min, gt_t + tol_min, color="#2ca02c", alpha=0.18)
    #     ax2.axvline(x=gt_t, color="#2ca02c", linestyle=":", alpha=0.8)

    # if not pred_with_boost.empty:
    #     for idx, pred_t in enumerate(pred_with_boost["timestamp_sec"].values / 60.0):
    #         ax2.axvline(x=pred_t, color="#9467bd", linestyle="-", linewidth=2.5, zorder=5, label="Final Prediction" if idx == 0 else "")
    #         ax2.scatter([pred_t], [1.02], color="#9467bd", marker="v", s=100, zorder=6)

    # ax2.set_xlabel("Time (Minutes)", fontsize=11)
    # ax2.set_ylim(-0.05, 1.3)
    # ax2.grid(True, linestyle=":", alpha=0.5)
    # ax2.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("boundary_dataset_enriched_v5.csv")
    
    artifact = joblib.load("boundary_classifier.joblib")
    model = artifact["model"]
    features = artifact["features"]

    # Fill NaNs with 0 to match training data
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0
    df[features] = df[features].fillna(0)

    # Sample standard episodes to inspect
    sample_vids = df["video_id"].unique()[80:99]
    for vid in sample_vids:
        print(f"Plotting predictions for video_id: {vid}")
        plot_episode_comparison(vid, df, model, features, high_threshold=0.53, low_threshold=0.49)