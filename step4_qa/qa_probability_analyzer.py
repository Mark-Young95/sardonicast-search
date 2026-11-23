import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Import the post-processor used by your macro chapter model
from post_processor_v4 import process_episode_predictions

def analyze_qa_peak_probabilities(
    dataset_path: str = "boundary_112_enriched_five.csv", # Update this if your macro dataset name is different
    model_path: str = "boundary_classifier_five.joblib"
):
    print(f"Loading macro dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    print(f"Loading macro model from {model_path}...")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    features = artifact["features"]
    
    qa_probs = []
    
    print("Processing episodes to extract Q&A predicted peaks...")
    for video_id, group in df.groupby("video_id"):
        ep_df = group.sort_values("timestamp_sec").copy()
        
        # Fill missing features with 0
        for feat in features:
            if feat not in ep_df.columns:
                ep_df[feat] = 0
                
        # 1. Predict probabilities
        ep_df["prob"] = model.predict_proba(ep_df[features])[:, 1]
        
        # 2. Run post-processor to find boundaries
        pred_df = process_episode_predictions(
            ep_df,
            prob_col="prob",
            high_threshold=0.51, # Use your current thresholds
            low_threshold=0.50,
            max_chapters=10,
            apply_rating_boost=False
        )
        
        # 3. Extract the Q&A boundary probability (second-to-last chapter)
        if len(pred_df) >= 2:
            qa_prob = pred_df.iloc[-2]["prob"]
            qa_probs.append(qa_prob)
        else:
            print(f"  ⚠️ Skipping {video_id}: Model predicted fewer than 2 boundaries.")

    if not qa_probs:
        print("❌ No Q&A boundaries could be extracted.")
        return

    # --- Print Statistics ---
    qa_probs_series = pd.Series(qa_probs)
    print("\n" + "="*40)
    print(" Q&A PEAK PROBABILITY STATISTICS")
    print("="*40)
    print(f"Total Q&A Boundaries Found : {len(qa_probs)}")
    print(f"Minimum Peak Probability   : {qa_probs_series.min():.4f}")
    print(f"Maximum Peak Probability   : {qa_probs_series.max():.4f}")
    print(f"Mean (Average) Probability : {qa_probs_series.mean():.4f}")
    print(f"Median Probability         : {qa_probs_series.median():.4f}")
    print(f"Standard Deviation         : {qa_probs_series.std():.4f}")
    print("\nPercentiles:")
    print(f"  25th Percentile          : {qa_probs_series.quantile(0.25):.4f}")
    print(f"  75th Percentile          : {qa_probs_series.quantile(0.75):.4f}")
    print("="*40)
    
    # --- Plot Histogram ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the histogram
    counts, bins, patches = ax.hist(qa_probs, bins=20, color='#00ffcc', edgecolor='black', alpha=0.8)
    
    # Add Mean and Median indicator lines
    mean_val = qa_probs_series.mean()
    median_val = qa_probs_series.median()
    
    ax.axvline(mean_val, color='#ff00ff', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='#ffff00', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
    
    ax.set_title("Distribution of Q&A Peak Probabilities (Macro Model)", fontsize=14, pad=15)
    ax.set_xlabel("Peak Probability Value", fontsize=12)
    ax.set_ylabel("Number of Episodes", fontsize=12)
    
    # Optional: Highlight your current high_threshold limit
    ax.axvspan(0, 0.51, color='red', alpha=0.15, label='Below Current Threshold (0.51)')
    
    ax.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_qa_peak_probabilities()