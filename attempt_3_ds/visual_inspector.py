# visual_inspector.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

def plot_episode_predictions(
    video_id: str, 
    df: pd.DataFrame, 
    model, 
    features: list, 
    threshold: float = 0.5
):
    ep_df = df[df["video_id"] == video_id].sort_values("timestamp_sec").copy()
    if ep_df.empty:
        return

    # Calculate model probabilities
    X_ep = ep_df[features]
    ep_df["prob"] = model.predict_proba(X_ep)[:, 1]

    timestamps = ep_df["timestamp_sec"].values / 60.0  # Convert to minutes
    probs = ep_df["prob"].values
    gt_boundaries = ep_df[ep_df["is_boundary"] == 1]["timestamp_sec"].values / 60.0

    plt.figure(figsize=(16, 5))
    
    # 1. Plot continuous probability curve
    plt.plot(timestamps, probs, label="XGBoost Boundary Probability", color="#1f77b4", linewidth=1.5)
    plt.axhline(y=threshold, color="red", linestyle="--", alpha=0.6, label=f"Threshold ({threshold})")

    # 2. Shade True Ground-Truth Boundary Zones (+/- 45 sec tolerance)
    tol_min = 45.0 / 60.0
    for idx, gt_t in enumerate(gt_boundaries):
        plt.axvspan(
            gt_t - tol_min, gt_t + tol_min, 
            color="green", alpha=0.18, 
            label="Ground Truth Zone" if idx == 0 else ""
        )
        plt.axvline(x=gt_t, color="green", linestyle="-", alpha=0.7)

    # 3. Mark candidate hits above threshold
    hits = ep_df[ep_df["prob"] >= threshold]
    plt.scatter(
        hits["timestamp_sec"] / 60.0, hits["prob"], 
        color="orange", zorder=5, s=30, label="Raw Model Hits"
    )

    plt.title(f"Episode Probability Profile: {video_id}", fontsize=13)
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Probability")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("boundary_dataset.csv")
    model = joblib.load("boundary_classifier.joblib")
    features = ["rel_position", "cosine_dist", "gap_duration", "has_outro_keyword"]

    # Sample top 3 unique videos to plot
    sample_videos = df["video_id"].unique()[:10]
    for vid in sample_videos:
        print(f"video_id: {vid}")
        plot_episode_predictions(vid, df, model, features, threshold=0.5)