# combined_visualizer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mplcursors

# Import your existing pipeline functions
from post_processor_v4 import process_episode_predictions
from qa_post_processor import process_qa_predictions
from qa_enrichment import attach_transcript_text, enrich_qa_dataset

def visualize_combined_pipeline(
    video_id: str, 
    df_macro: pd.DataFrame, 
    macro_model, 
    macro_features: list, 
    qa_model, 
    qa_features: list
):
    print(f"\n--- Processing Video ID: {video_id} ---")
    
    # ==========================================
    # 1. MACRO PIPELINE
    # ==========================================
    ep_df = df_macro[df_macro["video_id"] == video_id].sort_values("timestamp_sec").copy()
    if ep_df.empty:
        print(f"❌ Video {video_id} not found in macro dataset.")
        return

    # Predict Macro Probabilities
    ep_df["prob"] = macro_model.predict_proba(ep_df[macro_features])[:, 1]

    high_thresh_macro = 0.57
    low_thresh_macro = 0.55
    
    # Post-process to find Chapters, Q&A, and Outro boundaries
    macro_preds = process_episode_predictions(
        ep_df, prob_col="prob", 
        high_threshold=high_thresh_macro,
        low_threshold=low_thresh_macro, 
        max_chapters=10, 
        apply_rating_boost=False
    )
    
    # Extract Q&A boundaries for slicing
    qa_rows = macro_preds[macro_preds["label"] == "qa_boundary"]
    outro_rows = macro_preds[macro_preds["label"] == "outro_boundary"]
    
    has_qa_slice = False
    qa_df = pd.DataFrame()
    qa_preds = pd.DataFrame()

    high_thresh_micro = 0.60
    low_thresh_micro = 0.54
    
    if not qa_rows.empty and not outro_rows.empty:
        qa_start = qa_rows.iloc[0]["timestamp_sec"]
        outro_start = outro_rows.iloc[0]["timestamp_sec"]
        
        if qa_start < outro_start:
            has_qa_slice = True
            print(f"✅ Q&A Window Detected: {qa_start / 60:.2f}m to {outro_start / 60:.2f}m")
            
            # ==========================================
            # 2. MICRO (Q&A) PIPELINE ON THE FLY
            # ==========================================
            # Slice the episode
            qa_slice = ep_df[(ep_df["timestamp_sec"] >= qa_start) & (ep_df["timestamp_sec"] <= outro_start)].copy()
            
            # Run on-the-fly enrichment using your existing functions
            qa_slice = attach_transcript_text(qa_slice, window_sec=15.0)
            qa_slice = enrich_qa_dataset(qa_slice)
            
            # Clean up for XGBoost
            if "text" in qa_slice.columns:
                qa_slice = qa_slice.drop(columns=["text"])
            
            for feat in qa_features:
                if feat not in qa_slice.columns:
                    qa_slice[feat] = 0
            qa_slice[qa_features] = qa_slice[qa_features].fillna(0)
            
            # Predict Micro Probabilities
            qa_slice["qa_prob"] = qa_model.predict_proba(qa_slice[qa_features])[:, 1]
            qa_df = qa_slice
            
            # Post-process Q&A
            qa_preds = process_qa_predictions(
                qa_df, prob_col="qa_prob", 
                high_threshold=high_thresh_micro,
                low_threshold=low_thresh_micro, 
                max_gap_sec=15.0
            )
        else:
            print("⚠️ Q&A start is after Outro start. Skipping Micro pipeline.")
    else:
        print("⚠️ Missing Q&A or Outro labels. Skipping Micro pipeline.")


    # ==========================================
    # 3. PLOTTING (Two-Tier Light Theme)
    # ==========================================
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1]})
    fig.suptitle(f"Pipeline Visualization: {video_id}", fontsize=16, fontweight='bold', y=0.98)
    
    # ------------------------------------------
    # TOP PLOT: MACRO TIMELINE
    # ------------------------------------------
    timestamps_min = ep_df["timestamp_sec"].values / 60.0
    line_macro = ax1.plot(timestamps_min, ep_df["prob"], label="Macro Prob", color="#1f77b4", linewidth=1.5, alpha=0.85)[0]
    
    ax1.axhline(y=high_thresh_macro, color="red", linestyle="-", alpha=0.7, label=f"Peak Thresh ({high_thresh_macro})")
    ax1.axhline(y=low_thresh_macro, color="red", linestyle=":", alpha=0.5, label=f"Shoulder Thresh ({low_thresh_macro})")
    
    # Ground Truths (Macro)
    gt_macro = ep_df[ep_df["is_boundary"] == 1]["timestamp_sec"].values / 60.0
    for idx, gt_t in enumerate(gt_macro):
        ax1.axvspan(gt_t - (45/60), gt_t + (45/60), color="#2ca02c", alpha=0.18, label="Ground Truth" if idx == 0 else "")
        ax1.axvline(x=gt_t, color="#2ca02c", linestyle=":", alpha=0.8)

    # Macro Predictions
    if not macro_preds.empty:
        for _, row in macro_preds.iterrows():
            pred_sec = row["timestamp_sec"]
            label = row.get("label", "chapter")
            
            if label == "qa_boundary":
                line_color = '#00cc00'
                display_text = f"Q&A [{int(pred_sec//60)}:{int(pred_sec%60):02d}]"
            elif label == "outro_boundary":
                line_color = '#ff8800'
                display_text = f"Outro [{int(pred_sec//60)}:{int(pred_sec%60):02d}]"
            else:
                line_color = '#e6e600'
                display_text = f"[{int(pred_sec//60)}:{int(pred_sec%60):02d}]"
            
            ax1.axvline(x=pred_sec / 60.0, color="#9467bd", linestyle="-", linewidth=2.5, zorder=5)
            ax1.scatter([pred_sec / 60.0], [1.02], color="#9467bd", marker="v", s=100, zorder=6)
            ax1.text(pred_sec / 60.0, 0.95, display_text, color='black', backgroundcolor=line_color, 
                     fontsize=9, rotation=90, va='top', ha='right')

    ax1.set_title("Macro Boundary Classifier (Full Episode)", fontsize=12)
    ax1.set_ylabel("Probability")
    ax1.set_ylim(0.35, 1.3)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.95)

    # ------------------------------------------
    # BOTTOM PLOT: MICRO (Q&A) TIMELINE
    # ------------------------------------------
    if has_qa_slice:
        qa_min = qa_df["timestamp_sec"].values / 60.0
        line_micro = ax2.plot(qa_min, qa_df["qa_prob"], label="Q&A Prob", color="#008080", linewidth=2.0)[0]
        
        ax2.axhspan(low_thresh_micro, high_thresh_micro, color='gray', alpha=0.15, label='Candidate Zone')
        ax2.axhline(y=high_thresh_micro, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # QA Ground Truths
        gt_qa = qa_df[qa_df["is_boundary"] == 1]["timestamp_sec"].values / 60.0
        for i, gt_sec in enumerate(gt_qa):
            ax2.axvline(x=gt_sec, color='#ff00ff', linestyle=':', linewidth=2.5, label='Ground Truth' if i == 0 else "")
            
        # QA Predictions
        if not qa_preds.empty:
            for i, pred_sec in enumerate(qa_preds["timestamp_sec"].values):
                ax2.axvline(x=pred_sec / 60.0, color='#d62728', linestyle='-', linewidth=2, label='Predicted Question' if i == 0 else "")
                ax2.text(pred_sec / 60.0, 0.95, f"Q [{int(pred_sec//60)}:{int(pred_sec%60):02d}]",
                         color='white', backgroundcolor='#d62728', fontsize=9, rotation=90, va='top', ha='right')

        ax2.set_title(f"Micro Q&A Classifier (Zoomed to {qa_start/60:.1f}m - {outro_start/60:.1f}m)", fontsize=12)
        ax2.set_xlim(qa_start / 60.0, outro_start / 60.0)
    else:
        ax2.text(0.5, 0.5, "No Valid Q&A Section Detected / Sliced", 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=ax2.transAxes, fontsize=14, color='gray')
        line_micro = None

    ax2.set_xlabel("Time (Minutes)")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0.3, 1.2)
    ax2.grid(True, linestyle=":", alpha=0.5)
    if has_qa_slice:
        ax2.legend(loc="upper right", framealpha=0.95)

    # ------------------------------------------
    # HOVER CURSORS
    # ------------------------------------------
    cursor1 = mplcursors.cursor(line_macro, hover=True)
    @cursor1.connect("add")
    def on_add1(sel):
        sel.annotation.set_text(f"Time: {sel.target[0]:.2f} min\nProb: {sel.target[1]:.2f}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

    if line_micro:
        cursor2 = mplcursors.cursor(line_micro, hover=True)
        @cursor2.connect("add")
        def on_add2(sel):
            sel.annotation.set_text(f"Time: {sel.target[0]:.2f} min\nProb: {sel.target[1]:.2f}")
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Load Macro Data & Model
    df_macro = pd.read_csv("boundary_112_enriched_five.csv")
    macro_art = joblib.load("boundary_classifier_five.joblib")
    macro_model = macro_art["model"]
    macro_features = macro_art["features"]
    
    # Fill macro NaNs to match training
    for feat in macro_features:
        if feat not in df_macro.columns:
            df_macro[feat] = 0
    df_macro[macro_features] = df_macro[macro_features].fillna(0)

    # 2. Load Micro (Q&A) Model
    qa_art = joblib.load("qa_classifier.joblib")
    qa_model = qa_art["model"]
    qa_features = qa_art["features"]

    # 3. Sample standard episodes to inspect
    # ones i've done: 60-79, 80-99, 94-113, 40-59
    sample_vids = df_macro["video_id"].unique()[50:59]
    for vid in sample_vids:
        visualize_combined_pipeline(vid, df_macro, macro_model, macro_features, qa_model, qa_features)