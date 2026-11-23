# visual_inspector_v2.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from post_processor import process_episode_predictions

def plot_episode_with_final_predictions(
    video_id: str,
    df: pd.DataFrame,
    model,
    features: list,
    high_threshold: float = 0.60,
    low_threshold: float = 0.45,
    max_chapters: int = 10
):
    ep_df = df[df["video_id"] == video_id].sort_values("timestamp_sec").copy()
    if ep_df.empty:
        return

    # 1. Generate probabilities
    ep_df["prob"] = model.predict_proba(ep_df[features])[:, 1]

    # 2. Extract final predictions via Hysteresis (0.60 peak -> 0.45 capture)
    pred_df = process_episode_predictions(
        ep_df,
        prob_col="prob",
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        max_chapters=max_chapters
    )

#     def process_episode_predictions(
#     df: pd.DataFrame, 
#     prob_col: str = "prob", 
#     high_threshold: float = 0.60, 
#     low_threshold: float = 0.45,
#     max_chapters: int = 10,
#     max_gap_sec: float = 20.0,
#     outro_window_sec: float = 600.0
# )

    timestamps_min = ep_df["timestamp_sec"].values / 60.0
    probs = ep_df["prob"].values
    gt_boundaries_min = ep_df[ep_df["is_boundary"] == 1]["timestamp_sec"].values / 60.0

    plt.figure(figsize=(15, 6))
    
    # Layer A: Model probability curve
    plt.plot(timestamps_min, probs, label="XGBoost Boundary Probability", color="#1f77b4", linewidth=1.5, alpha=0.85)
    plt.axhline(y=high_threshold, color="red", linestyle="--", alpha=0.7, label=f"Peak Threshold ({high_threshold})")
    plt.axhline(y=low_threshold, color="orange", linestyle=":", alpha=0.5, label=f"Shoulder Capture ({low_threshold})")

    # Layer B: Ground-Truth Zones (+/- 45 sec tolerance)
    tol_min = 45.0 / 60.0
    for idx, gt_t in enumerate(gt_boundaries_min):
        plt.axvspan(
            gt_t - tol_min, gt_t + tol_min,
            color="#2ca02c", alpha=0.18,
            label="Ground Truth Zone" if idx == 0 else ""
        )
        plt.axvline(x=gt_t, color="#2ca02c", linestyle=":", alpha=0.8)

    # Layer C: FINAL PREDICTED BOUNDARIES (Hysteresis & Shoulder Aligned)
    if not pred_df.empty:
        final_preds_min = pred_df["timestamp_sec"].values / 60.0
        for idx, pred_t in enumerate(final_preds_min):
            plt.axvline(
                x=pred_t, color="#9467bd", linestyle="-", linewidth=2.5, zorder=5,
                label="Final Prediction (0.60 Peak / 0.45 Right Shoulder)" if idx == 0 else ""
            )
            plt.scatter([pred_t], [1.02], color="#9467bd", marker="v", s=100, zorder=6)

    plt.title(f"Episode Boundary Predictions (Dual-Threshold Hysteresis): {video_id}", fontsize=13)
    plt.xlabel("Time (Minutes)", fontsize=11)
    plt.ylabel("Probability", fontsize=11)
    plt.ylim(-0.05, 1.3)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("boundary_dataset_enriched.csv")
    artifact = joblib.load("boundary_classifier.joblib")
    
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    features = artifact["features"] if isinstance(artifact, dict) else [
        "rel_position", "cosine_dist", "cosine_dist_diff", 
        "cosine_dist_zscore", "gap_duration", "gap_duration_diff",
        "has_outro_kw", "has_qa_kw", "has_transition_kw", "has_rating_kw"
    ]

    sample_vids = df["video_id"].unique()[70:89]
    for vid in sample_vids:
        print(f"Plotting predictions for video_id: {vid}")
        plot_episode_with_final_predictions(
            vid, df, model, features, high_threshold=0.60, low_threshold=0.45, max_chapters=8
        )