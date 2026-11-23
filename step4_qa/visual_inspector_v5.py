# visual_inspector_v2.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mplcursors
from post_processor_v4 import process_episode_predictions

def plot_episode_predictions(
    video_id: str,
    df: pd.DataFrame,
    model,
    features: list,
    high_threshold: float = 0.51,
    low_threshold: float = 0.50,
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

    # 2. Get Final Predictions (No Boost)
    pred_df = process_episode_predictions(
        ep_df, prob_col="prob", high_threshold=high_threshold, 
        low_threshold=low_threshold, max_chapters=max_chapters, apply_rating_boost=False
    )

    # 3. Setup the Figure
    plt.figure(figsize=(15, 6))
    tol_min = 45.0 / 60.0

    plt.title(f"Episode Boundary Predictions (Pure XGBoost): {video_id}", fontsize=13)
    
    # Capture the line object for the cursor!
    line_raw = plt.plot(timestamps_min, raw_probs, label="Raw XGBoost", color="#1f77b4", linewidth=1.5, alpha=0.85)[0]
    
    plt.axhline(y=high_threshold, color="red", linestyle="-", alpha=0.7, label=f"Peak ({high_threshold})")
    plt.axhline(y=low_threshold, color="red", linestyle=":", alpha=0.5, label=f"Shoulder ({low_threshold})")
    
    # Plot Ground Truth
    for idx, gt_t in enumerate(gt_boundaries_min):
        plt.axvspan(gt_t - tol_min, gt_t + tol_min, color="#2ca02c", alpha=0.18, label="Ground Truth" if idx == 0 else "")
        plt.axvline(x=gt_t, color="#2ca02c", linestyle=":", alpha=0.8)

    # Plot Final Predictions

    if not pred_df.empty:
        for i, row in pred_df.iterrows():
            pred_sec = row["timestamp_sec"]
            label = row.get("label", "chapter")
            
            # Color-code and name the labels based on their semantic type
            if label == "qa_boundary":
                line_color = '#00ff00'  # Bright Green for Q&A
                display_text = f"Q&A [{int(pred_sec//60)}:{int(pred_sec%60):02d}]"
            elif label == "outro_boundary":
                line_color = '#ff6600'  # Orange for Outro
                display_text = f"Outro [{int(pred_sec//60)}:{int(pred_sec%60):02d}]"
            else:
                line_color = '#ffff00'  # Yellow for standard chapters
                display_text = f"Chapter [{int(pred_sec//60)}:{int(pred_sec%60):02d}]"
            
            # Add the text box to the top of the line
            plt.text(pred_sec / 60.0, 0.95, display_text, 
                    color='black', backgroundcolor=line_color, fontsize=9, rotation=90, 
                    verticalalignment='top', horizontalalignment='right')
        for idx, pred_t in enumerate(pred_df["timestamp_sec"].values / 60.0):
            plt.axvline(x=pred_t, color="#9467bd", linestyle="-", linewidth=2.5, zorder=5, label="Final Prediction" if idx == 0 else "")
            plt.scatter([pred_t], [1.02], color="#9467bd", marker="v", s=100, zorder=6)
    
    plt.xlabel("Time (Minutes)", fontsize=11)
    plt.ylabel("Probability", fontsize=11)
    plt.ylim(-0.05, 1.3)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.95)

    # ==========================================
    # INTERACTIVE HOVER CURSORS
    # ==========================================
    cursor = mplcursors.cursor(line_raw, hover=True)
    
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(f"Time: {sel.target[0]:.2f} min\nProb: {sel.target[1]:.2f}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("boundary_112_enriched_five.csv")
    
    artifact = joblib.load("boundary_classifier_five.joblib")
    model = artifact["model"]
    features = artifact["features"]

    # Fill NaNs with 0 to match training data
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0
    df[features] = df[features].fillna(0)

    # Sample standard episodes to inspect
    sample_vids = df["video_id"].unique()[60:79]
    for vid in sample_vids:
        print(f"Plotting predictions for video_id: {vid}")
        plot_episode_predictions(vid, df, model, features)