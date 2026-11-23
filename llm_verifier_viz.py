# llm_verifier_viz.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os
import re

from post_processor_v4 import process_episode_predictions
from qa_post_processor import process_qa_predictions
from qa_enrichment import attach_transcript_text, enrich_qa_dataset
from llm_verifier import verify_macro_candidates, verify_micro_candidates, synthesize_macro_qa_title

def load_transcript_data(video_id: str, cache_dir: str = "transcript_cache"):
    cache_path = os.path.join(cache_dir, f"{video_id}.json")
    if not os.path.exists(cache_path):
        print(f"❌ Cache file not found for {video_id}")
        return None, None, None
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    snippets = data.get("snippets", data) if isinstance(data, dict) else data
    episode_title = data.get("meta", {}).get("title", f"Sardonicast episode {video_id}")
    episode_date = data.get("meta", {}).get("upload_date")
    return snippets, episode_title, episode_date

def format_time(seconds):
    """Helper to convert seconds to MM:SS string for graph labels"""
    return f"{int(seconds//60)}:{int(seconds%60):02d}"

def sync_macro_qa_from_micro(verified_macro_df: pd.DataFrame, verified_micro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolls up the verified Micro Q&A question topics into the Macro Q&A chapter's
    Title, Description, and Tags.
    """
    if verified_macro_df is None or verified_macro_df.empty or verified_micro_df is None or verified_micro_df.empty:
        return verified_macro_df

    qa_mask = (
        verified_macro_df["is_qa_segment"].fillna(False)
        | (verified_macro_df["label"] == "qa_boundary")
        | verified_macro_df["llm_chapter_title"].astype(str).str.upper().str.startswith("Q&A")
    )
    if not qa_mask.any():
        return verified_macro_df

    micro_topics = [
        re.sub(r"(?i)^Q&A:\s*", "", str(t)).strip()
        for t in verified_micro_df["llm_chapter_title"].tolist()
        if str(t).strip()
    ]

    # Combine unique Micro question topics + top tags from each question
    rolled_tags = []
    seen = set()
    for topic in micro_topics:
        if topic.lower() not in seen:
            seen.add(topic.lower())
            rolled_tags.append(topic)

    for tag_list in verified_micro_df["llm_tags"].tolist():
        if isinstance(tag_list, list):
            for tg in tag_list[:3]:  # Grab up to 3 top tags per question
                if tg.lower() not in seen:
                    seen.add(tg.lower())
                    rolled_tags.append(tg)

    # 1. Dynamic Description
    topics_summary = f"The hosts answer listener questions covering {', '.join(micro_topics[:-1])}, and {micro_topics[-1]}." if len(micro_topics) > 1 else f"The hosts answer listener questions about {micro_topics[0]}."
    
    # 2. Dynamic Chapter Title (e.g., "Q&A: Sardonicast Jingle, Yellow Paint")
    if micro_topics:
        # Take the first 3 topics for the title, truncate to ~25 chars each to prevent massive titles
        title_topics = [t.split(" (")[0].split(":")[0][:25].strip() for t in micro_topics[:3]]
        # Use the LLM synthesizer instead of [:25] string slicing
        dynamic_title = synthesize_macro_qa_title(micro_topics)
    else:
        dynamic_title = "Q&A"

    for idx in verified_macro_df[qa_mask].index:
        verified_macro_df.at[idx, "llm_chapter_title"] = dynamic_title
        verified_macro_df.at[idx, "llm_description"] = topics_summary
        verified_macro_df.at[idx, "llm_tags"] = rolled_tags

    return verified_macro_df

def print_and_save_pass2_report(
    video_id: str,
    episode_title: str,
    verified_macro_df: pd.DataFrame,
    verified_micro_df: pd.DataFrame = None,
    output_dir: str = "verified_outputs"
):
    """Prints a readable inspection table of Titles, Descriptions, and Tags and caches to CSV."""
    verified_macro_df = sync_macro_qa_from_micro(verified_macro_df, verified_micro_df)

    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 85)
    print(f"📋 PASS 2 SUMMARY INSPECTION: {episode_title} ({video_id})")
    print("=" * 85)

    rows_to_save = []

    if verified_macro_df is not None and not verified_macro_df.empty:
        print("\n[MACRO CHAPTERS]")
        for idx, row in verified_macro_df.iterrows():
            start_s = float(row.get("timestamp_sec", 0.0))
            end_s = float(row.get("end_timestamp_sec", start_s))
            title = row.get("llm_chapter_title", "Untitled")
            desc = row.get("llm_description", "")
            tags = row.get("llm_tags", [])
            tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)

            print(f"  {idx+1}. [{format_time(start_s)} -> {format_time(end_s)}] {title}")
            print(f"     ├─ Description: {desc}")
            print(f"     └─ Tags:        [{tags_str}]")

            rows_to_save.append({
                "video_id": video_id,
                "granularity": "MACRO",
                "start_sec": start_s,
                "end_sec": end_s,
                "start_mmss": format_time(start_s),
                "end_mmss": format_time(end_s),
                "title": title,
                "description": desc,
                "tags": tags_str
            })

    if verified_micro_df is not None and not verified_micro_df.empty:
        print("\n[MICRO Q&A SEGMENTS]")
        for idx, row in verified_micro_df.iterrows():
            start_s = float(row.get("timestamp_sec", 0.0))
            end_s = float(row.get("end_timestamp_sec", start_s))
            title = row.get("llm_chapter_title", "Untitled Q&A")
            desc = row.get("llm_description", "")
            tags = row.get("llm_tags", [])
            tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)

            print(f"  Q{idx+1}. [{format_time(start_s)} -> {format_time(end_s)}] {title}")
            print(f"      ├─ Description: {desc}")
            print(f"      └─ Tags:        [{tags_str}]")

            rows_to_save.append({
                "video_id": video_id,
                "granularity": "MICRO_QA",
                "start_sec": start_s,
                "end_sec": end_s,
                "start_mmss": format_time(start_s),
                "end_mmss": format_time(end_s),
                "title": title,
                "description": desc,
                "tags": tags_str
            })

    if rows_to_save:
        out_csv = os.path.join(output_dir, f"{video_id}_verified_chapters.csv")
        pd.DataFrame(rows_to_save).to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\n💾 Saved Pass 2 titles, descriptions, and tags to: {out_csv}")
    print("=" * 85)

# ---------------------------------------------------------------------------
# Plotting Function
# ---------------------------------------------------------------------------
def plot_verification_results(ax, ep_df, verified_df, rejected_df, video_id, mode="Macro"):
    """Generates a graph showing XGBoost probabilities and LLM decisions on a specific axis."""
    if "prob" in ep_df.columns:
        ax.plot(ep_df['timestamp_sec'] / 60.0, ep_df['prob'], label='XGBoost Probability', color='blue', alpha=0.6)
    elif "qa_prob" in ep_df.columns:
        ax.plot(ep_df['timestamp_sec'] / 60.0, ep_df['qa_prob'], label='XGBoost QA Probability', color='orange', alpha=0.6)

    if verified_df is not None and not verified_df.empty:
        for _, row in verified_df.iterrows():
            if row.get("label") == "intro":
                continue
            t_sec = row['timestamp_sec']
            t_min = t_sec / 60.0
            short_title = str(row.get("llm_chapter_title", ""))[:18]
            ax.axvline(t_min, color='green', linestyle='-', linewidth=2, alpha=0.8)
            ax.text(
                t_min, 1.03, f"✅ {format_time(t_sec)}\n{short_title}",
                color='green', ha='center', fontsize=8, rotation=35
            )

    if rejected_df is not None and not rejected_df.empty:
        for _, row in rejected_df.iterrows():
            t_sec = row['timestamp_sec']
            t_min = t_sec / 60.0
            ax.axvline(t_min, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(t_min, 1.03, f"🚫 {format_time(t_sec)}", color='red', ha='center', fontsize=9, rotation=35)

    ax.set_title(f"[{video_id}] {mode} Verification Results")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Boundary Probability")
    ax.set_ylim(0.3, 1.35)
    if mode == "Macro":
        ax.axhline(y=0.57, color='red', linewidth=2)
        ax.axhline(y=0.545, color='red', linestyle='--', linewidth=2)
    else:
        ax.axhline(y=0.60, color='red', linewidth=2)
        ax.axhline(y=0.55, color='red', linestyle='--', linewidth=2)
    ax.legend(loc="upper right")

# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def run_macro_verification_pipeline(video_id: str, df_macro: pd.DataFrame, macro_model, macro_features: list, ax=None):
    print(f"\n==================================================\nMACRO: {video_id}\n==================================================")
    snippets_data, episode_title, episode_date = load_transcript_data(video_id)
    if not snippets_data:
        return None, None, None

    max_time = snippets_data[-1]["start_seconds"] + 60.0 if snippets_data else 99999.0

    ep_df = df_macro[df_macro["video_id"] == video_id].sort_values("timestamp_sec").copy()
    ep_df["prob"] = macro_model.predict_proba(ep_df[macro_features])[:, 1]

    macro_preds = process_episode_predictions(
        ep_df, prob_col="prob", high_threshold=0.57, low_threshold=0.545, max_chapters=10, apply_rating_boost=False
    )

    verified_macro_df, rejected_macro_df = verify_macro_candidates(
        candidates_df=macro_preds,
        full_snippets=snippets_data,
        episode_title=episode_title,
        episode_date=episode_date,
        max_time=max_time
    )

    if ax is not None:
        plot_verification_results(ax, ep_df, verified_macro_df, rejected_macro_df, video_id, "Macro")

    return verified_macro_df, macro_preds, episode_title

def run_micro_verification_pipeline(
    video_id: str, verified_macro_df: pd.DataFrame, raw_macro_preds: pd.DataFrame, df_macro: pd.DataFrame, qa_model, qa_features: list, ax=None
):
    print(f"\n==================================================\nMICRO: {video_id}\n==================================================")

    qa_rows = verified_macro_df[
        verified_macro_df['is_qa_segment'].fillna(False) |
        verified_macro_df['llm_chapter_title'].str.upper().str.startswith("Q&A")
    ]
    outro_rows = verified_macro_df[verified_macro_df['llm_chapter_title'].str.upper().str.startswith("OUTRO")]
    if qa_rows.empty:
        return None

    qa_start_sec = float(qa_rows.iloc[0]['timestamp_sec'])
    outro_start_sec = float(outro_rows.iloc[0]['timestamp_sec']) if not outro_rows.empty else 99999.0

    ep_df = df_macro[df_macro["video_id"] == video_id].sort_values("timestamp_sec").copy()
    qa_slice = ep_df[(ep_df["timestamp_sec"] >= qa_start_sec) & (ep_df["timestamp_sec"] <= outro_start_sec)].copy()

    if not qa_slice.empty:
        qa_slice = attach_transcript_text(qa_slice, window_sec=15.0)
        qa_slice = enrich_qa_dataset(qa_slice)
        if "text" in qa_slice.columns:
            qa_slice = qa_slice.drop(columns=["text"])
        for feat in qa_features:
            if feat not in qa_slice.columns:
                qa_slice[feat] = 0
        qa_slice[qa_features] = qa_slice[qa_features].fillna(0)

        qa_slice["qa_prob"] = qa_model.predict_proba(qa_slice[qa_features])[:, 1]
        raw_qa_preds = process_qa_predictions(
            qa_slice, prob_col="qa_prob", high_threshold=0.60, low_threshold=0.55, max_gap_sec=15.0
        )
    else:
        raw_qa_preds = pd.DataFrame()

    valid_start = qa_start_sec + 30.0
    m_filtered = raw_macro_preds[
        (raw_macro_preds['timestamp_sec'] >= valid_start) & (raw_macro_preds['timestamp_sec'] < outro_start_sec)
    ].copy()

    if not raw_qa_preds.empty:
        combined_cands = pd.concat([
            m_filtered,
            raw_qa_preds[(raw_qa_preds['timestamp_sec'] >= valid_start) & (raw_qa_preds['timestamp_sec'] < outro_start_sec)]
        ]).sort_values('timestamp_sec').reset_index(drop=True)
    else:
        combined_cands = m_filtered.sort_values('timestamp_sec').reset_index(drop=True)

    deduped_cands = []
    last_t = -9999.0
    for _, row in combined_cands.iterrows():
        t = row['timestamp_sec']
        if (t - last_t) > 45.0:
            row_copy = row.copy()
            row_copy['label'] = 'qa_boundary'
            deduped_cands.append(row_copy)
            last_t = t

    final_micro_cands = pd.DataFrame(deduped_cands)
    snippets_data, episode_title, episode_date = load_transcript_data(video_id)

    verified_micro_df, rejected_micro_df = verify_micro_candidates(
        candidates_df=final_micro_cands,
        full_snippets=snippets_data,
        episode_title=episode_title,
        scope_start=qa_start_sec,
        scope_end=outro_start_sec
    )

    if ax is not None:
        plot_verification_results(ax, qa_slice, verified_micro_df, rejected_micro_df, video_id, "Micro (Q&A)")

    return verified_micro_df

if __name__ == "__main__":
    df_macro = pd.read_csv("boundary_112_enriched_five.csv")
    macro_art = joblib.load("boundary_classifier_five.joblib")
    macro_model, macro_features = macro_art["model"], macro_art["features"]
    for feat in macro_features:
        if feat not in df_macro.columns:
            df_macro[feat] = 0
    df_macro[macro_features] = df_macro[macro_features].fillna(0)

    qa_art = joblib.load("qa_classifier.joblib")
    qa_model, qa_features = qa_art["model"], qa_art["features"]

    TEST_VIDEOS = df_macro["video_id"].unique()[40:60]

    for vid in TEST_VIDEOS:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        ax_macro, ax_micro = axes

        verified_macro_df, raw_macro_preds, ep_title = run_macro_verification_pipeline(
            vid, df_macro, macro_model, macro_features, ax=ax_macro
        )

        verified_micro_df = None
        if verified_macro_df is not None and not verified_macro_df.empty:
            verified_micro_df = run_micro_verification_pipeline(
                vid, verified_macro_df, raw_macro_preds, df_macro, qa_model, qa_features, ax=ax_micro
            )
        else:
            ax_micro.text(0.5, 0.5, "No Macro Q&A Boundary Found", ha='center', va='center', fontsize=14)
            ax_micro.set_title(f"[{vid}] Micro (Q&A) Verification Results")

        # Print Pass 2 titles, descriptions, and tags to console + save to CSV
        print_and_save_pass2_report(vid, ep_title or vid, verified_macro_df, verified_micro_df)

        plt.tight_layout()
        print(f"\n⏸️ Graph generated for {vid}. Waiting... Close the pop-up window to continue to the next video!")
        plt.show()