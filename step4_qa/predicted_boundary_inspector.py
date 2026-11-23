import pandas as pd
import joblib
from post_processor_v4 import process_episode_predictions

def inspect_predicted_boundaries(
    csv_path: str = "boundary_dataset_enriched_five.csv",
    model_path: str = "boundary_classifier_five.joblib"
):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loading macro model from {model_path}...\n")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    features = artifact["features"]
    
    extracted_rows = []
    
    for video_id, group in df.groupby("video_id"):
        ep_df = group.sort_values("timestamp_sec").copy()
        
        # Fill missing features with 0 for the model
        for feat in features:
            if feat not in ep_df.columns:
                ep_df[feat] = 0
                
        # 1. Predict probabilities
        ep_df["prob"] = model.predict_proba(ep_df[features])[:, 1]
        
        # =====================================================================
        # DEBUG: The Full Feature Audit at 101 minutes (6060s)
        # =====================================================================
        debug_df = ep_df if 'ep_df' in locals() else df
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

        # 2. Run your updated post-processor
        pred_df = process_episode_predictions(
            ep_df,
            prob_col="prob",
            high_threshold=0.51, 
            low_threshold=0.50,
            max_chapters=10,
            apply_rating_boost=False
        )
        
        if pred_df.empty:
            continue
            
        # --- THE FIX: Stamp the video_id back onto the predictions ---
        pred_df["video_id"] = video_id
        
        # Ensure density columns are present for the inspector if they got stripped
        #   transition_density       : 0.4960
#   qa_density               : 0.1054
#   rel_position             : 0.0633
#   outro_density            : 0.0424
#   cosine_dist              : 0.0398
        for col in ["qa_density", "outro_density", "transition_density", "rel_position"]:
            if col in ep_df.columns and col not in pred_df.columns:
                # Map them back using the timestamp
                pred_df = pd.merge(pred_df, ep_df[["timestamp_sec", col]], on="timestamp_sec", how="left")
        # -------------------------------------------------------------

        if video_id == "xAV2jXtKjW8": #"jRGf2_A9jxI":
            extracted_rows.append(pred_df)
        # qa_density = 0.0173 vs 0.0034 
        # cosine = 0.1006 vs 0.0740
        # transition_density = 0.0035 vs 0












        # 3. Extract the labeled boundaries
        # qa_rows = pred_df[pred_df["label"] == "qa_boundary"]
        # outro_rows = pred_df[pred_df["label"] == "outro_boundary"]
        
        # if not qa_rows.empty:
        #     extracted_rows.append(qa_rows.iloc[0])
        # elif len(pred_df) >= 2:
        #     fallback_qa = pred_df.iloc[-2].copy()
        #     fallback_qa["label"] = "qa_boundary (fallback)"
        #     extracted_rows.append(fallback_qa)
            
        # if not outro_rows.empty:
        #     extracted_rows.append(outro_rows.iloc[0])
        # elif len(pred_df) >= 1:
        #     fallback_outro = pred_df.iloc[-1].copy()
        #     fallback_outro["label"] = "outro_boundary (fallback)"
        #     extracted_rows.append(fallback_outro)

    if not extracted_rows:
        print("No boundaries were predicted for any episodes.")
        return

    # Combine into a single dataframe
    #pd.concat()
    diagnostic_df = pd.concat(extracted_rows, ignore_index=True)
    #diagnostic_df = pd.DataFrame(extracted_rows)

    print("diagnostic df columns: ", diagnostic_df.columns)
    # Sort so Q&A and Outro for the same video appear right next to each other
    diagnostic_df = diagnostic_df.sort_values(["video_id", "timestamp_sec"])
    
    # Calculate time from end for validation
    if "ep_duration_sec" in diagnostic_df.columns:
        diagnostic_df["time_from_end_sec"] = diagnostic_df["ep_duration_sec"] - diagnostic_df["timestamp_sec"]
    
    # Select columns to display
    columns_to_show = [
        "video_id", 
        "label",
        "timestamp_sec",
        "time_from_end_sec",
        "prob",              # Model's confidence score
        "cosine_dist"
    ]
    
    if "qa_density" in df.columns: columns_to_show.append("qa_density")
    if "outro_density" in df.columns: columns_to_show.append("outro_density")
    if "transition_density" in df.columns: columns_to_show.append("transition_density")
    if "rel_position" in df.columns: columns_to_show.append("rel_position")
    
    final_cols = [c for c in columns_to_show if c in diagnostic_df.columns]
    
    # Format the table
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 150)
    
    print("======================================================================================================")
    print(" PREDICTED BOUNDARY FEATURES (Macro Model)")
    print("======================================================================================================")
    #condition = (diagnostic_df["label"] == "outro_boundary") & (diagnostic_df["rel_position"] < 0.96)
    condition = (diagnostic_df["label"] == "qa_boundary")
    # Apply the condition to the rows, and select 'final_cols' for the columns
    filtered_df = diagnostic_df[final_cols] #.loc[condition, final_cols]
    print(filtered_df.to_markdown(index=False))
    #print(diagnostic_df[final_cols]["label" == "outro_boundary"]["rel_position" < 0.96].to_markdown(index=False))
    print("======================================================================================================")
    
    diagnostic_df[final_cols].to_csv("predicted_boundary_inspection.csv", index=False)
    print("\n✅ Saved full table to predicted_boundary_inspection.csv")

    import matplotlib.pyplot as plt

    # Plot the histogram
    diagnostic_df[diagnostic_df["label"] == "qa_boundary"]["rel_position"].plot(kind="hist", bins=20, edgecolor="black")

    # Add labels and a title for clarity
    plt.title("Distribution of Relative Positions for Outro Boundaries")
    plt.xlabel("Relative Position")
    plt.ylabel("Frequency")

    # Display the plot
    plt.show()

if __name__ == "__main__":
    inspect_predicted_boundaries()