# evaluate_coalesced_pipeline.py
import pandas as pd
import joblib
import numpy as np
from finetune_utils import coalesce_clusters_to_midpoint

def run_evaluation(csv_path="boundary_dataset.csv", model_path="boundary_classifier.joblib"):
    df = pd.read_csv(csv_path)
    clf = joblib.load(model_path)
    features = ["rel_position", "cosine_dist", "gap_duration", "has_outro_keyword"]

    df["prob"] = clf.predict_proba(df[features])[:, 1]

    tp, fp, fn = 0, 0, 0
    tolerance_sec = 45.0

    for vid_id, group in df.groupby("video_id"):
        # Ground-truth boundary timestamps for this video
        gt_times = group[group["is_boundary"] == 1]["timestamp_sec"].values
        
        # Apply cluster filtering + midpoint coalescing
        pred_df = coalesce_clusters_to_midpoint(
            group, 
            prob_col="prob", 
            threshold=0.6, 
            min_cluster_size=3
        )
        
        pred_times = pred_df["coalesced_timestamp_sec"].values if not pred_df.empty else np.array([])

        # Calculate True Positives, False Positives, False Negatives with 45s tolerance
        matched_gt = set()
        for p_t in pred_times:
            matched = False
            for gt_t in gt_times:
                if abs(p_t - gt_t) <= tolerance_sec:
                    matched = True
                    matched_gt.add(gt_t)
                    break
            if matched:
                tp += 1
            else:
                fp += 1

        fn += (len(gt_times) - len(matched_gt))

    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * (precision * recall) / max(1e-9, (precision + recall))

    print("--- Post-Processed Evaluation (Cluster >= 5 + Midpoint) ---")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1 Score:        {f1:.4f}")

if __name__ == "__main__":
    run_evaluation()