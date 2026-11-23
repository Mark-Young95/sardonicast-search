import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve

def evaluate_and_tune():
    df = pd.read_csv("boundary_dataset.csv")
    features = ["rel_position", "cosine_dist", "gap_duration", "has_outro_keyword"]
    
    X = df[features]
    y = df["is_boundary"].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = joblib.load("boundary_classifier.joblib")
    y_probs = clf.predict_proba(X_test)[:, 1]

    # 1. Sweep Thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    print(f"--- Optimal Threshold Results ---")
    print(f"Best Decision Threshold: {best_thresh:.4f}")
    print(f"Precision at Best F1:   {precisions[best_idx]:.4f}")
    print(f"Recall at Best F1:      {recalls[best_idx]:.4f}")
    print(f"Best F1-Score:          {f1_scores[best_idx]:.4f}\n")

    # 2. Evaluate with Optimal Threshold
    y_custom_pred = (y_probs >= best_thresh).astype(int)
    print("--- Classification Report at Tuned Threshold ---")
    print(classification_report(y_test, y_custom_pred, target_names=["Non-Boundary", "Boundary"]))

if __name__ == "__main__":
    evaluate_and_tune()