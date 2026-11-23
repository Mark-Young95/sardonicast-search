import pandas as pd
import numpy as np
import joblib
from post_processor_v4 import process_episode_predictions

def check_ground_truth_match(pred_ts: float, gt_timestamps: list, tolerance: float = 45.0) -> str:
    """Checks if a predicted boundary is within 45 seconds of a ground truth boundary."""
    for gt_ts in gt_timestamps:
        if abs(pred_ts - gt_ts) <= tolerance:
            return "Ground Truth (True Positive)"
    return "False Positive"

def generate_tables(
    csv_path="boundary_dataset_enriched_v5.csv",
    model_path="boundary_classifier.joblib"
):
    print("Loading data and model...")
    df = pd.read_csv(csv_path)
    
    artifact = joblib.load(model_path)
    model = artifact["model"]
    features = artifact["features"]

    # Fill NaNs to match training
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0
    df[features] = df[features].fillna(0)

    table1_data = []  # Last 10 Minutes
    table2_data = []  # Pre-Q&A / Main Body Evaluated

    for video_id, group in df.groupby("video_id"):
        ep_df = group.sort_values("timestamp_sec").copy()
        ep_duration = ep_df["timestamp_sec"].max()
        
        # 1. Get Probabilities
        ep_df["prob"] = model.predict_proba(ep_df[features])[:, 1]
        
        # 2. Get Final Predicted Boundaries
        pred_df = process_episode_predictions(
            ep_df, prob_col="prob", high_threshold=0.55, 
            low_threshold=0.40, max_chapters=10, apply_rating_boost=False
        )
        
        # 3. Get Ground Truth Timestamps
        gt_timestamps = ep_df[ep_df["is_boundary"] == 1]["timestamp_sec"].values
        
        if pred_df.empty:
            continue
            
        for _, row in pred_df.iterrows():
            ts = row["timestamp_sec"]
            prob = row["prob"]
            formatted_time = f"{int(ts//60):02d}:{int(ts%60):02d}"
            
            # --- TABLE 1: Last 10 Minutes ---
            if (ep_duration - ts) <= 600:
                table1_data.append({
                    "video_id": video_id,
                    "timestamp": formatted_time,
                    "probability": round(prob, 4)
                })
                
            # --- TABLE 2: Pre-Q&A (Main Episode) Evaluated ---
            # Assuming Q&A / Outro zone is roughly the last 15-20 minutes. 
            # We will evaluate everything outside the 10-minute Outro window.
            else:
                match_status = check_ground_truth_match(ts, gt_timestamps, tolerance=45.0)
                table2_data.append({
                    "video_id": video_id,
                    "timestamp": formatted_time,
                    "status": match_status,
                    "probability": round(prob, 4)
                })

    # Convert to DataFrames for easy exporting and printing
    t1_df = pd.DataFrame(table1_data)
    t2_df = pd.DataFrame(table2_data)

    # Save to CSV
    t1_df.to_csv("table1_last_10_mins.csv", index=False)
    t2_df.to_csv("table2_pre_qa_evaluated.csv", index=False)

    # Print to console in Excel-ready Tab-Separated format
    print("\n" + "="*50)
    print("TABLE 1: Boundaries in the Last 10 Minutes")
    print("="*50)
    print("video_id\ttimestamp\tprobability")
    for _, row in t1_df.iterrows():
        print(f"{row['video_id']}\t{row['timestamp']}\t{row['probability']}")

    print("\n" + "="*50)
    print("TABLE 2: Pre-Q&A Boundaries (Evaluated)")
    print("="*50)
    print("video_id\ttimestamp\tground_truth_or_false_positive\tprobability")
    for _, row in t2_df.iterrows():
        print(f"{row['video_id']}\t{row['timestamp']}\t{row['status']}\t{row['probability']}")
        
    print("\n✅ Automatically saved to 'table1_last_10_mins.csv' and 'table2_pre_qa_evaluated.csv'")

if __name__ == "__main__":
    generate_tables()