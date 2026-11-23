import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os

from post_processor_v4 import process_episode_predictions
from qa_post_processor import process_qa_predictions
from qa_enrichment import attach_transcript_text, enrich_qa_dataset
from llm_verifier import filter_candidates_with_llm

def load_transcript_data(video_id: str, cache_dir: str = "transcript_cache"):
    cache_path = os.path.join(cache_dir, f"{video_id}.json")
    if not os.path.exists(cache_path):
        print(f"❌ Cache file not found for {video_id}")
        return None, None
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    snippets = data.get("snippets", data) if isinstance(data, dict) else data
    episode_title = data.get("meta", {}).get("title", f"Sardonicast episode {video_id}")
    episode_date = data.get("meta", {}).get("upload_date")
    return snippets, episode_title, episode_date

def format_time(seconds):
    """Helper to convert seconds to MM:SS string for graph labels"""
    return f"{int(seconds//60)}:{int(seconds%60):02d}"

# ---------------------------------------------------------------------------
# Plotting Function (Now uses 'ax' for subplots)
# ---------------------------------------------------------------------------
def plot_verification_results(ax, ep_df, verified_df, rejected_df, video_id, mode="Macro"):
    """Generates a graph showing XGBoost probabilities and LLM decisions on a specific axis."""
    
    # Base ML Probability Line
    if "prob" in ep_df.columns:
        ax.plot(ep_df['timestamp_sec'] / 60.0, ep_df['prob'], label='XGBoost Probability', color='blue', alpha=0.6)
    elif "qa_prob" in ep_df.columns:
        ax.plot(ep_df['timestamp_sec'] / 60.0, ep_df['qa_prob'], label='XGBoost QA Probability', color='orange', alpha=0.6)

    # Plot Verified Boundaries (Green) + Timestamps
    if verified_df is not None and not verified_df.empty:
        for _, row in verified_df.iterrows():
            if row.get("label") == "intro": continue # Skip 0:00 intro marker
            t_sec = row['timestamp_sec']
            t_min = t_sec / 60.0
            ax.axvline(t_min, color='green', linestyle='-', linewidth=2, alpha=0.8)
            ax.text(t_min, 1.05, f"✅ {format_time(t_sec)}", color='green', ha='center', fontsize=10, rotation=45)

    # Plot Rejected Boundaries (Red) + Timestamps
    if rejected_df is not None and not rejected_df.empty:
        for _, row in rejected_df.iterrows():
            t_sec = row['timestamp_sec']
            t_min = t_sec / 60.0
            ax.axvline(t_min, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(t_min, 1.05, f"🚫 {format_time(t_sec)}", color='red', ha='center', fontsize=10, rotation=45)

    ax.set_title(f"[{video_id}] {mode} Verification Results")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Boundary Probability")
    # Raised the ceiling slightly so text labels don't get cut off
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper right")

# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def run_macro_verification_pipeline(video_id: str, df_macro: pd.DataFrame, macro_model, macro_features: list, ax=None):
    print(f"\n==================================================\nMACRO: {video_id}\n==================================================")
    snippets_data, episode_title, episode_date = load_transcript_data(video_id)
    if not snippets_data: return None, None

    ep_df = df_macro[df_macro["video_id"] == video_id].sort_values("timestamp_sec").copy()
    ep_df["prob"] = macro_model.predict_proba(ep_df[macro_features])[:, 1]
    
    macro_preds = process_episode_predictions(ep_df, prob_col="prob", high_threshold=0.57, low_threshold=0.55, max_chapters=12, apply_rating_boost=False)
    
    verified_macro_df, rejected_macro_df = filter_candidates_with_llm(
        candidates_df=macro_preds, full_snippets=snippets_data, episode_title=episode_title, episode_date=episode_date, mode="macro", window_sec=420.0
    )
    
    if ax is not None:
        plot_verification_results(ax, ep_df, verified_macro_df, rejected_macro_df, video_id, "Macro")
    
    return verified_macro_df, macro_preds

def run_micro_verification_pipeline(video_id: str, verified_macro_df: pd.DataFrame, raw_macro_preds: pd.DataFrame, df_macro: pd.DataFrame, qa_model, qa_features: list, ax=None):
    print(f"\n==================================================\nMICRO: {video_id}\n==================================================")
    
    qa_rows = verified_macro_df[verified_macro_df['llm_chapter_title'].str.upper().str.startswith("Q&A")]
    outro_rows = verified_macro_df[verified_macro_df['llm_chapter_title'].str.upper().str.startswith("OUTRO")]
    if qa_rows.empty: return None

    qa_start_sec = qa_rows.iloc[0]['timestamp_sec']
    outro_start_sec = outro_rows.iloc[0]['timestamp_sec'] if not outro_rows.empty else 99999.0

    ep_df = df_macro[df_macro["video_id"] == video_id].sort_values("timestamp_sec").copy()
    qa_slice = ep_df[(ep_df["timestamp_sec"] >= qa_start_sec) & (ep_df["timestamp_sec"] <= outro_start_sec)].copy()
    
    if not qa_slice.empty:
        qa_slice = attach_transcript_text(qa_slice, window_sec=15.0)
        qa_slice = enrich_qa_dataset(qa_slice)
        if "text" in qa_slice.columns: qa_slice = qa_slice.drop(columns=["text"])
        for feat in qa_features:
            if feat not in qa_slice.columns: qa_slice[feat] = 0
        qa_slice[qa_features] = qa_slice[qa_features].fillna(0)
        
        qa_slice["qa_prob"] = qa_model.predict_proba(qa_slice[qa_features])[:, 1]
        raw_qa_preds = process_qa_predictions(qa_slice, prob_col="qa_prob", high_threshold=0.60, low_threshold=0.54, max_gap_sec=15.0)
    else:
        raw_qa_preds = pd.DataFrame()

    valid_start = qa_start_sec + 30.0 
    m_filtered = raw_macro_preds[(raw_macro_preds['timestamp_sec'] >= valid_start) & (raw_macro_preds['timestamp_sec'] < outro_start_sec)].copy()
    
    if not raw_qa_preds.empty:
        combined_cands = pd.concat([m_filtered, raw_qa_preds[(raw_qa_preds['timestamp_sec'] >= valid_start) & (raw_qa_preds['timestamp_sec'] < outro_start_sec)]]).sort_values('timestamp_sec').reset_index(drop=True)
    else:
        combined_cands = m_filtered.sort_values('timestamp_sec').reset_index(drop=True)
    
    deduped_cands = []
    last_t = -9999.0
    for _, row in combined_cands.iterrows():
        t = row['timestamp_sec']
        if (t - last_t) > 45.0:
            row['label'] = 'qa_boundary' 
            deduped_cands.append(row)
            last_t = t
            
    final_micro_cands = pd.DataFrame(deduped_cands)
    snippets_data, episode_title, episode_date = load_transcript_data(video_id)
    
    verified_micro_df, rejected_micro_df = filter_candidates_with_llm(
        candidates_df=final_micro_cands, full_snippets=snippets_data, episode_title=episode_title, episode_date=episode_date,
        mode="micro", window_sec=180.0, scope_start_sec=qa_start_sec, scope_end_sec=outro_start_sec
    )

    if ax is not None:
        plot_verification_results(ax, qa_slice, verified_micro_df, rejected_micro_df, video_id, "Micro (Q&A)")
    
    return verified_micro_df

if __name__ == "__main__":
    df_macro = pd.read_csv("boundary_112_enriched_five.csv")
    macro_art = joblib.load("boundary_classifier_five.joblib")
    macro_model, macro_features = macro_art["model"], macro_art["features"]
    for feat in macro_features:
        if feat not in df_macro.columns: df_macro[feat] = 0
    df_macro[macro_features] = df_macro[macro_features].fillna(0)

    qa_art = joblib.load("qa_classifier.joblib")
    qa_model, qa_features = qa_art["model"], qa_art["features"]

    # ADD MORE VIDEOS HERE
    # ones i've done: 60-79, 80-99, 94-113, 40-59
    TEST_VIDEOS = df_macro["video_id"].unique()[40:44] #["VqoBXdHw87g",
        # Add additional video IDs here like:
        # "cSkEMK0Z3Eo",
        # "jRGf2_A9jxI"]
    
    for vid in TEST_VIDEOS:
        # Create a combined figure with two stacked charts
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        ax_macro, ax_micro = axes
        
        verified_macro_df, raw_macro_preds = run_macro_verification_pipeline(
            vid, df_macro, macro_model, macro_features, ax=ax_macro
        )

        if verified_macro_df is not None and not verified_macro_df.empty:
            verified_micro_df = run_micro_verification_pipeline(
                vid, verified_macro_df, raw_macro_preds, df_macro, qa_model, qa_features, ax=ax_micro
            )
        else:
            ax_micro.text(0.5, 0.5, "No Macro Q&A Boundary Found", ha='center', va='center', fontsize=14)
            ax_micro.set_title(f"[{vid}] Micro (Q&A) Verification Results")
            
        plt.tight_layout()
        print(f"\n⏸️ Graph generated for {vid}. Waiting... Close the pop-up window to continue to the next video!")
        
        # Blocking call! The script stops here until you click the 'X' on the window.
        plt.show()