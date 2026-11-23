import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Import your existing pipeline functions
from build_dataset import get_cached_transcript_and_meta, extract_features_for_candidates, parse_description_timestamps
from run_enrichment import reconstruct_windowed_text, enrich_transcript_dataset
from post_processor_v4 import process_episode_predictions

def process_and_visualize_new_episode(
    video_id: str, 
    model_path: str = "boundary_classifier_five.joblib",
    stride_sec: float = 5.0  # MUST match what the model was trained on!
):
    print(f"\n🚀 Starting End-to-End Inference for {video_id}...")
    
    # --- 1. DATA GATHERING ---
    meta, snippets = get_cached_transcript_and_meta(video_id)
    if not meta or not snippets:
        print(f"❌ Failed to load metadata or transcript for {video_id}")
        return

    # Try to grab ground truths if they exist in the description
    gt_chapters = parse_description_timestamps(meta.get("description", ""))
    
    # --- 2. VECTOR ENCODING (Cosine Distance) ---
    print("Extracting primitive features (GPU/Embeddings)...")
    rows = extract_features_for_candidates(
        snippets, 
        gt_chapters, 
        window_sec=60.0, 
        stride_sec=stride_sec
    )
    
    if not rows:
        print("❌ Feature extraction failed.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=[
        "timestamp_sec", "rel_position", "cosine_dist", 
        "gap_duration", "has_outro_keyword", "is_boundary"
    ])
    df.insert(0, "video_id", video_id)
    
    # --- 3. NLP ENRICHMENT ---
    # Hydrate the text, then run the rolling counts/densities
    df = reconstruct_windowed_text(df, "transcript_cache", window_sec=60.0)
    df = enrich_transcript_dataset(df)
    
    # --- 4. MODEL PREDICTION ---
    print("\nLoading Macro Model...")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    features = artifact["features"]
    
    # Ensure all required features exist (fill missing with 0)
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0
            
    print("Scoring timeline...")
    df["prob"] = model.predict_proba(df[features])[:, 1]
    
    # --- 5. POST-PROCESSING ---
    print("Applying post-processor rules (Moats & Density limits)...")

    # =====================================================================
    # DEBUG: The Full Feature Audit at 101 minutes (6060s)
    # =====================================================================
    debug_df = df #ep_df if 'ep_df' in locals() else df
    # Narrow it specifically to the 6060s mark to keep the table clean
    debug_mask = (debug_df["timestamp_sec"] >= 6055) & (debug_df["timestamp_sec"] <= 6065)
        
    print(f"\n--- FULL FEATURE AUDIT AT 6060s ({video_id if 'video_id' in locals() else 'Visualizer'}) ---")
    if not debug_df[debug_mask].empty:
        # Print timestamp, prob, AND every single feature the model relies on
        cols_to_print = ["timestamp_sec", "prob"] + features
        # Filter to avoid KeyError if something is completely missing
        valid_cols = [c for c in cols_to_print if c in debug_df.columns]
            
        # Transpose (.T) so the features list vertically for easy reading!
        print(debug_df[debug_mask][valid_cols].T.to_markdown())
    else:
        print("No rows found in this time window!")
    print("=====================================================================\n")

    high_thresh = 0.51
    low_thresh = 0.50
    pred_df = process_episode_predictions(
        df, 
        prob_col="prob", 
        high_threshold=high_thresh, 
        low_threshold=low_thresh, 
        max_chapters=10, 
        apply_rating_boost=False
    )
    
    # --- 6. PLOTTING ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 6))
    
    df["timestamp_min"] = df["timestamp_sec"] / 60.0
    
    # Plot probability curve
    ax.plot(df["timestamp_min"], df["prob"], color='#00aaff', linewidth=1.5, label='Chapter Probability')
    ax.axhspan(low_thresh, high_thresh, color='gray', alpha=0.2, label='Candidate Zone')
    ax.axhline(y=high_thresh, color='white', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot Ground Truths (if available)
    if not df[df["is_boundary"] == 1].empty:
        ground_truths = df[df["is_boundary"] == 1]["timestamp_sec"].values
        for i, gt_sec in enumerate(ground_truths):
            label = 'Ground Truth (from YouTube)' if i == 0 else ""
            ax.axvline(x=gt_sec / 60.0, color='#ff00ff', linestyle=':', linewidth=2.5, label=label)
            
    # Plot Predictions
    if not pred_df.empty:
        # Dictionary to track legend entries so we don't duplicate them
        legend_added = {"qa_boundary": False, "outro_boundary": False, "chapter": False}

        for i, row in pred_df.iterrows():
            pred_sec = row["timestamp_sec"]
            label_type = row.get("label", "chapter")
            
            # Color-code and name the labels based on their semantic type
            if label_type == "qa_boundary":
                line_color = '#00ff00'  # Bright Green for Q&A
                display_text = f"Q&A [{int(pred_sec//60)}:{int(pred_sec%60):02d}]"
                legend_label = "Predicted Q&A" if not legend_added["qa_boundary"] else ""
                legend_added["qa_boundary"] = True
                
            elif label_type == "outro_boundary":
                line_color = '#ff6600'  # Orange for Outro
                display_text = f"Outro [{int(pred_sec//60)}:{int(pred_sec%60):02d}]"
                legend_label = "Predicted Outro" if not legend_added["outro_boundary"] else ""
                legend_added["outro_boundary"] = True
                
            else:
                line_color = '#ffff00'  # Yellow for standard chapters
                display_text = f"Chapter [{int(pred_sec//60)}:{int(pred_sec%60):02d}]"
                legend_label = "Predicted Chapter" if not legend_added["chapter"] else ""
                legend_added["chapter"] = True
            
            # Draw the vertical line
            ax.axvline(x=pred_sec / 60.0, color=line_color, linestyle='-', linewidth=2, label=legend_label)
            
            # Add the text box to the top of the line
            ax.text(pred_sec / 60.0, 0.95, display_text, 
                    color='black', backgroundcolor=line_color, fontsize=9, rotation=90, 
                    verticalalignment='top', horizontalalignment='right')

    title_text = meta.get('title', video_id)
    ax.set_title(f"Macro Model Inference: {title_text}", fontsize=14, pad=15)
    ax.set_xlabel("Time (Minutes)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_ylim(0, 1.05)
    
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Paste a brand new episode ID here to test it out!
    NEW_EPISODE_ID = "xAV2jXtKjW8" # "cSkEMK0Z3Eo" #"0M15ka97iVA" #"9KpIgO6y2PQ" #"jRGf2_A9jxI"  # Replace with any video ID
    process_and_visualize_new_episode(NEW_EPISODE_ID)