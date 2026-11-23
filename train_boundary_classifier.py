import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, confusion_matrix

def train_model(
    csv_path: str = "boundary_112_enriched_five.csv", 
    model_output_path: str = "boundary_classifier_five.joblib"
    # boundary_classifier for boundary
    # change this to qa_classifier_five.joblib for qa
):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Dataset is empty. Run run_enrichment.py first.")

    # Clean up potential inf/NaN values from division
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ---------------------------------------------------------
    # 1. THE LEAN & MEAN FEATURE LIST
    # We dropped raw _count and has_kw flags to remove noise.
    # We added filler features and mentions_chapter_title!
    # gap_duration acts as our "silence" tracker.
    # ---------------------------------------------------------
    candidate_features = [
        # Base structural / spatial features
        "rel_position", "time_from_start_sec", "time_from_end_sec",
        
        # Audio/Semantic disruption features
        "cosine_dist", "cosine_dist_diff", "cosine_dist_zscore", 
        "gap_duration", "gap_duration_diff", 
        
        # The new LLM-style Title Catcher
        "mentions_chapter_title",
        
        # Density Features (Accounts for chunk size)
        "qa_density", "outro_density", "transition_density", "rating_density", 
        "filler_density", # <-- Your grunt/filler tracker!
        
        # Local Context / Rolling Windows
        "qa_rolling_5min", "outro_rolling_5min", "transition_rolling_5min", "rating_rolling_5min",
        "filler_rolling_5min", "has_3x_questions",

        # The Anti-Boundary Zone
        "rating_upcoming_60s",

        # New Spoiler Windows
        "spoiler_rolling_5min",  # The trailing 5-minute diminish
        "spoiler_upcoming_5m"    # The forward 5-minute boost
    ]

    # Filter features to only those that actually exist in the CSV (Robustness!)
    features = [f for f in candidate_features if f in df.columns]

    print(f"Training with {len(features)} optimized features: {features}")
    
    X = df[features]
    y = df["is_boundary"].astype(int)
    groups = df["video_id"] if "video_id" in df.columns else df.index

    num_positives = int(y.sum())
    num_negatives = len(y) - num_positives
    print(f"Total samples: {len(df)} | Boundaries (1s): {num_positives} | Non-boundaries (0s): {num_negatives}")

    if num_positives == 0:
        raise ValueError("No positive boundary labels found in dataset.")

    # 2. Class Imbalance Weighting (Dampened by 0.5x to reduce false-positive spam)
    raw_scale_weight = num_negatives / max(1, num_positives)
    scale_pos_weight = raw_scale_weight * 0.5
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f} (raw ratio: {raw_scale_weight:.2f})")

    # 3. Group-Aware Train/Test Split (Prevents video/snippet leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 4. Model Training (With Early Stopping)
    clf = XGBClassifier(
        n_estimators=300,        # Increased max trees...
        early_stopping_rounds=20, # ...but will stop early if it overfits
        max_depth=4,             # Keep shallow to generalize
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric=["logloss", "aucpr"], # Track PR-AUC natively
        random_state=42
    )

    print("\nTraining XGBoost Model...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # 5. Model Evaluation
    y_preds = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:, 1]

    print("\n--- Evaluation Report (Group-Split Out-of-Sample) ---")
    print("Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_preds))
    print("\nClassification Report:")
    print(classification_report(y_test, y_preds, target_names=["Non-Boundary", "Boundary"], digits=4))
    
    # Calculate PR-AUC (crucial for imbalanced data)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    print(f"PR AUC Score : {pr_auc:.4f}")

    # 6. Feature Importance Readout
    print("\n--- Feature Importances ---")
    importances = clf.feature_importances_
    for feat, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {feat:25s}: {imp:.4f}")

    # 7. Save Model Artifact (Dictionary approach)
    artifact = {"model": clf, "features": features}
    joblib.dump(artifact, model_output_path)
    print(f"\n✅ Saved trained model & feature list to {model_output_path}")

if __name__ == "__main__":
    train_model()