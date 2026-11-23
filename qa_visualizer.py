import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mplcursors
from qa_post_processor import process_qa_predictions

def visualize_qa_boundaries(
    video_id: str,
    dataset_path: str = "qa_dataset_enriched_five.csv",
    model_path: str = "qa_classifier.joblib"
):
    print(f"Loading data for {video_id}...")
    df = pd.read_csv(dataset_path)
    
    # Filter to only the requested video
    ep_df = df[df["video_id"] == video_id].copy()
    if ep_df.empty:
        print(f"❌ Video {video_id} not found in {dataset_path}")
        return
        
    ep_df = ep_df.sort_values("timestamp_sec")
    
    # Load Model
    print("Loading Micro Q&A Model...")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    features = artifact["features"]
    
    # Fill missing features with 0 just in case
    for feat in features:
        if feat not in ep_df.columns:
            ep_df[feat] = 0
            
    # Predict probabilities
    ep_df["prob"] = model.predict_proba(ep_df[features])[:, 1]
    
    # SETTING THRESHOLD VALUES
    high_thresh = 0.61
    low_thresh = 0.54

    # Run Q&A post-processor (45s moat, no density limits)
    print("Running Q&A post-processor...")
    pred_df = process_qa_predictions(
        ep_df,
        prob_col="prob",
        high_threshold=high_thresh,
        low_threshold=low_thresh,
        max_gap_sec=15.0
    )
    
    # Extract Ground Truths
    ground_truths = ep_df[ep_df["is_boundary"] == 1]["timestamp_sec"].values
    
    # --- PLOTTING ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Convert seconds to minutes for cleaner X-axis reading
    ep_df["timestamp_min"] = ep_df["timestamp_sec"] / 60.0
    
    # Plot probability curve
    ax.plot(ep_df["timestamp_min"], ep_df["prob"], color='#00ffcc', linewidth=1.5, label='Question Probability')

    # Capture the line object for the cursor!
    line_raw = ax.plot(ep_df["timestamp_min"], ep_df["prob"], label="Raw XGBoost", color="#1f77b4", linewidth=1.5, alpha=0.85)[0]
    
    # Shade the threshold zone
    ax.axhspan(low_thresh, high_thresh, color='gray', alpha=0.2, label='Candidate Zone ')
    ax.axhline(y=high_thresh, color='white', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot Ground Truths
    for i, gt_sec in enumerate(ground_truths):
        label = 'Ground Truth Boundary' if i == 0 else ""
        ax.axvline(x=gt_sec / 60.0, color='#ff00ff', linestyle=':', linewidth=2.5, label=label)
        
    # Plot Model Predictions
    if not pred_df.empty:
        for i, pred_sec in enumerate(pred_df["timestamp_sec"].values):
            label = 'Predicted Question' if i == 0 else ""
            ax.axvline(x=pred_sec / 60.0, color='#ffff00', linestyle='-', linewidth=2, label=label)
            ax.text(pred_sec / 60.0, 0.95, f"{int(pred_sec//60)}:{int(pred_sec%60):02d}", 
                    color='black', backgroundcolor='#ffff00', fontsize=9, rotation=90, 
                    verticalalignment='top', horizontalalignment='right')

    ax.set_title(f"Q&A Micro-Model Boundary Predictions ({video_id})", fontsize=14, pad=15)
    ax.set_xlabel("Time (Minutes)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_ylim(0.15, 1.20)

        # ==========================================
    # INTERACTIVE HOVER CURSORS
    # ==========================================
    cursor = mplcursors.cursor(line_raw, hover=True)
    
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(f"Time: {sel.target[0]:.2f} min\nProb: {sel.target[1]:.2f}")
        sel.annotation.get_bbox_patch().set(fc="black", alpha=0.9)
    
    # Clean up legends
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test it on a video ID that you know has a lot of questions
    TEST_ID_LIST = ["cSkEMK0Z3Eo", "jRGf2_A9jxI", "xAV2jXtKjW8", "9KpIgO6y2PQ", "js7B-K6BtJs", "NTxV9b4V0Ow", "qR6gD-esmbE", "XnRdkcs7Tqo", "IybB6sCakxI", "qzwXZqZqcpI", "tajqOLlRWq0", "h3xPjF4TGWk", "zrry2eQp1_8"]
    # 9KpIgO6y2PQ
    # jRGf2_A9jxI
    for TEST_ID in TEST_ID_LIST:
        visualize_qa_boundaries(TEST_ID)