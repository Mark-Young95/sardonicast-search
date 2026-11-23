import os
import json
import numpy as np
import pandas as pd
import regex as re
import joblib
from post_processor_v4 import process_episode_predictions

QA_KEYWORD_PATTERNS = {
    "qa_transition": [
        re.compile(r"\b(hell[\s\W]+yeah|speaking[\s\W]+of|anyway|good[\s\W]+question|good\W+conversation|thank\W+you)\b", re.IGNORECASE)
    ],
    "qa_filler": [
        # Added a dedicated standard filler set as requested
        re.compile(r"\b(yeah\W+yeah|um|uh|so|okay|well|all[\s\W]+right|alright)\b", re.IGNORECASE)
    ],
    "qa_first": [
        re.compile(r"\blet['’]s[\s\W]+start(?:[\s\W]+off)?(?:[\s\W]+with)?(?:[\s\W]+this)?[\s\W]+one(?:[\s\W]+then)?[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\bask[\s\W]+us[\s\W]+(?:anything|whatever)[\s\W]+you(?:\W+feel)?[\s\W]+like[\s\W]+just[\s\W]+like\b", re.IGNORECASE),
        re.compile(r"\bask[\s\W]+us[\s\W]+(?:anything|whatever)[\s\W]+you[\s\W]+fancy[\s\W]+just[\s\W]+like\b", re.IGNORECASE),
        re.compile(r"\bgoing[\s\W]+to[\s\W]+start[\s\W]+us[\s\W]+off(?:[\s\W]+here)?\b", re.IGNORECASE),
        re.compile(r"\bleave[\s\W]+your[\s\W]+own[\s\W]+questions?\b", re.IGNORECASE),
        re.compile(r"\b(question[\s\W]+number[\s\W]+one|this[\s\W]+is[\s\W]+from)\b", re.IGNORECASE),
        re.compile(r"\bjust\W+like\b", re.IGNORECASE),
        re.compile(r"\bjust\W+like(?:\W+\w+){0,4}\W+did\b", re.IGNORECASE),
        re.compile(r"\bfuture\W+episodes\b", re.IGNORECASE),
        re.compile(r"\bsubreddit\W+just\W+like\b", re.IGNORECASE),
        re.compile(r"\bfuture\W+episodes\W+just\W+like\b", re.IGNORECASE),
        re.compile(r"\bquestions?\W+just\W+like\b", re.IGNORECASE),
    ],
    "qa_standard": [
        # ... (All your standard and wrapup regexes from the previous step go here) ...
        re.compile(r"\blet['’]s[\s\W]+do(?:[\s\W]+this)?[\s\W]+one[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\b(?:has|is)[\s\W]+this[\s\W]+to[\s\W]+(?:ask|say)\b", re.IGNORECASE),
        re.compile(r"\bhas\W+a\W+(?:quandary|quandry)\b", re.IGNORECASE),
        re.compile(r"\b(?:put|putting)\W+you\W+on\W+the\W+spot\b", re.IGNORECASE),
        re.compile(r"\bit['’]s\W+(?:a|an)\W+\w+\W+question\b", re.IGNORECASE),
        re.compile(r"\bspeaking\W+of\b", re.IGNORECASE),
        re.compile(r"\bspeaking\W+of(?:\W+\w+){0,5}\W+actually\b", re.IGNORECASE),
        re.compile(r"\bshould\W+i\W+just\W+throw\b", re.IGNORECASE),
        re.compile(r"\bthis(?:\W+\w+){0,2}[\s\W]+to[\s\W]+say\b", re.IGNORECASE),
        re.compile(r"\bi[\s\W]+got[\s\W]+to[\s\W]+do[\s\W]+this[\s\W]+one[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\blet['’]s[\s\W]+move[\s\W]+on[\s\W]+to(?:[\s\W]+the[\s\W]+next)?[\s\W]+(?:one|question)(?:[\s\W]+from|[\s\W]+by)?\b", re.IGNORECASE),
        re.compile(r"\bhere['’]s[\s\W]+something[\s\W]+a[\s\W]+bit[\s\W]+more\b", re.IGNORECASE),
        re.compile(r"\bi['’]ll[\s\W]+bring[\s\W]+this[\s\W]+\w+[\s\W]+question[\s\W]+up[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\b(?:next|another)(?:[\s\W]+kind[\s\W]+of)?[\s\W]+question\b", re.IGNORECASE),
        re.compile(r"\bwe[\s\W]+(?:answer|answered)[\s\W]+that\b", re.IGNORECASE),
        re.compile(r"\bi[\s\W]+got[\s\W]+a[\s\W]+\w+[\s\W]+one[\s\W]+we[\s\W]+can[\s\W]+do\b", re.IGNORECASE),
        re.compile(r"\bwe[\s\W]+have[\s\W]+a(?:[\s\W]+\w+){1,4}[\s\W]+one(?:[\s\W]+here)?[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\bhas[\s\W]+something(?:[\s\W]+curious|[\s\W]+for[\s\W]+us)\b", re.IGNORECASE),
        re.compile(r"\bhas[\s\W]+a[\s\W]+kind[\s\W]+of[\s\W]+interesting[\s\W]+one\b", re.IGNORECASE),
        re.compile(r"\b(?:how[\s\W]+about[\s\W]+|oh[\s\W]+)?(?:we[\s\W]+do[\s\W]+)?this[\s\W]+one[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\blet['’]s[\s\W]+do[\s\W]+the[\s\W]+\w+[\s\W]+one[\s\W]+then[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\blet['’]s[\s\W]+do[\s\W]+(?:an|a|this)[\s\W]+\w+[\s\W]+one\b", re.IGNORECASE),
        re.compile(r"\bwho[\s\W]+(?:asks|says)(?:\W+simply)?\b", re.IGNORECASE),
        re.compile(r"\blet['’]s[\s\W]+go[\s\W]+to\b", re.IGNORECASE),
        re.compile(r"\bthis[\s\W]+one['’]s[\s\W]+kind[\s\W]+of[\s\W]+interesting\b", re.IGNORECASE),
        re.compile(r"\bgot\W+an\W+interesting\W+one\W+here\b", re.IGNORECASE),
        re.compile(r"\bi\W+had\W+to\W+read\W+it\b", re.IGNORECASE),
        re.compile(r"\bsubject\W+came\W+up\W+earlier\b", re.IGNORECASE),
        re.compile(r"\blet['’]s[\s\W]+do[\s\W]+this[\s\W]+one[\s\W]+along[\s\W]+similar[\s\W]+lines[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\b(?:on|along)[\s\W]+similar[\s\W]+lines(?:[\s\W]+actually)?\b", re.IGNORECASE),
        re.compile(r"\bon\W+a\W+similar\W+line\b", re.IGNORECASE),
        re.compile(r"\blet['’]s\W+do\W+the\W+swim\b", re.IGNORECASE),
        re.compile(r"\banswer\W+here\W+from\b", re.IGNORECASE),
        re.compile(r"\blet['’]s\W+give\W+this\W+one\W+to\b", re.IGNORECASE),
        re.compile(r"\b(?:this\W+is|i\W+got)\W+(?:a|an)\W+\w+\W+one(?:\W+here)?\W+from\b", re.IGNORECASE),
        re.compile(r"\bi\W+guess\W+that['’]s\W+that\W+one\W+then\b", re.IGNORECASE),
        re.compile(r"\bthrow[\s\W]+out[\s\W]+a[\s\W]+\w[\s\W]+one\b", re.IGNORECASE),
        re.compile(r"\bthrow\W+out\W+there\W+and\W+see\b", re.IGNORECASE),
        re.compile(r"\bhas\W+one\W+for\W+us\b", re.IGNORECASE),
        re.compile(r"\b(?:has|is|as)\W+one(?:[\s\W]+for[\s\W]+us|[\s\W]+boris)?\b", re.IGNORECASE),
        re.compile(r"\b(?:asks|says)[\s\W]+this\b", re.IGNORECASE),
        re.compile(r"\bwe[\s\W]+have[\s\W]+a[\s\W]+question[\s\W]+from[\s\W]+a\b", re.IGNORECASE),
        re.compile(r"\bwe[\s\W]+have[\s\W]+a(?:\s+\w+){1,3}\s+one[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\blet['’]s[\s\W]+try(?:\s+\w+){1,4}\s+with[\s\W]+something[\s\W]+more\b", re.IGNORECASE),
        re.compile(r"\bi[\s\W]+(?:got|have)[\s\W]+a[\s\W]+question[\s\W]+for(?:[\s\W]+both[\s\W]+of)?[\s\W]+you(?:[\s\W]+guys)?\b", re.IGNORECASE),
        re.compile(r"\bif[\s\W]+we['’]re[\s\W]+done[\s\W]+with[\s\W]+this[\s\W]+question\b", re.IGNORECASE),
        re.compile(r"\bwhat(?:[\s\W]+do|[\s\W]+are)[\s\W]+you[\s\W]+guys(?:[\s\W]+as)?(?:[\s\W]+think[\s\W]+about|[\s\W]+thoughts[\s\W]+on)\b", re.IGNORECASE),
        re.compile(r"\bbefore[\s\W]+we[\s\W]+get[\s\W]+onto[\s\W]+the[\s\W]+next[\s\W]+one\b", re.IGNORECASE),
        re.compile(r"\bthis[\s\W]+one[\s\W]+(?:then|is|uh)[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\bhas\W+this\W+one\W+then\b", re.IGNORECASE),
        re.compile(r"\bdid[\s\W]+we[\s\W]+all[\s\W]+answer\b", re.IGNORECASE),
        re.compile(r"\bthe(?:\s+\w+){1,4}[\s\W]+asks\b", re.IGNORECASE),
        re.compile(r"\bthat['’]s[\s\W]+all[\s\W]+for[\s\W]+us\b", re.IGNORECASE),
        re.compile(r"\bhere['’]s[\s\W]+a(?:\s+\w+){1,2}[\s\W]+one\b", re.IGNORECASE),
        re.compile(r"\band[\s\W]+to[\s\W]+expand[\s\W]+on[\s\W]+the\b", re.IGNORECASE),
        re.compile(r"\b(?:\w+\d|\d|one|two|three|four|five|six|seven|eight|nine|zero)[\s\W]+(?:asks|says)\b", re.IGNORECASE),
        re.compile(r"\b\d+\W+(?:did|asks|says)\b", re.IGNORECASE),
        re.compile(r"\b\w+\d+\W+(?:did|asks|says)\b", re.IGNORECASE),
        re.compile(fr"\b\d+(?:\W+\w+){0,3}\W+for\W+us\b", re.IGNORECASE),
        re.compile(fr"\bas\W+well\W+for\W+us\b", re.IGNORECASE),
        re.compile(r"\bwe[\s\W]+have[\s\W]+two[\s\W]+questions[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\bwe[\s\W]+have[\s\W]+one[\s\W]+here\b", re.IGNORECASE),
        re.compile(r"\btheir[\s\W]+reddit[\s\W]+name\b", re.IGNORECASE),
        re.compile(r"\bwe\W+can\W+do\W+this\W+\w+\W+one\b", re.IGNORECASE)
    ],
    "qa_wrapup": [
        re.compile(r"\bshould[\s\W]+we[\s\W]+do[\s\W]+(?:one|two)[\s\W]+more(?:[\s\W]+here)?\b", re.IGNORECASE),
        re.compile(r"\bi['’]m\W+down\W+for\W+one\W+more\W+question\b", re.IGNORECASE),
        re.compile(r"\byou\W+can\W+do\W+two\W+more\b", re.IGNORECASE),
        re.compile(r"\bwe[\s\W]+can[\s\W]+kind[\s\W]+of[\s\W]+wind[\s\W]+down[\s\W]+here\b", re.IGNORECASE),
        re.compile(r"\blet['’]s[\s\W]+do[\s\W]+a[\s\W]+couple[\s\W]+more[\s\W]+here(?:[\s\W]+like[\s\W]+this[\s\W]+one)?(?:[\s\W]+from)?\b", re.IGNORECASE),
        re.compile(r"\bwe[\s\W]+can[\s\W]+do[\s\W]+a[\s\W]+couple[\s\W]+more\b", re.IGNORECASE),
        re.compile(r"\bone[\s\W]+to[\s\W]+(?:close|wrap)[\s\W]+it[\s\W]+up\b", re.IGNORECASE),
        re.compile(r"\b(?:let['’]s[\s\W]+)?(?:\w+\W+)?end[\s\W]+on[\s\W]+this[\s\W]+one(?:[\s\W]+then)?[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\bone[\s\W]+more[\s\W]+or[\s\W]+we[\s\W]+done\b", re.IGNORECASE),
        re.compile(r"\blet['’]s[\s\W]+close[\s\W]+out[\s\W]+with\b", re.IGNORECASE),
        re.compile(r"\bhow[\s\W]+about[\s\W]+we[\s\W]+wrap[\s\W]+up[\s\W]+(?:on|with)[\s\W]+this[\s\W]+one\b", re.IGNORECASE),
        re.compile(r"\blet['’]s[\s\W]+wrap[\s\W]+up[\s\W]+on[\s\W]+this[\s\W]+one[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\bi[\s\W]+should[\s\W]+end[\s\W]+on[\s\W]+this[\s\W]+one[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\bi['’]m[\s\W]+fine[\s\W]+with[\s\W]+one[\s\W]+more[\s\W]+question\b", re.IGNORECASE),
        re.compile(r"\bdelve[\s\W]+into[\s\W]+this[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\bwant[\s\W]+to[\s\W]+do[\s\W]+one[\s\W]+more\b", re.IGNORECASE),
        re.compile(r"\b(?:wanna\W+do\W+)?one[\s\W]+(?:more|last)[\s\W]+question\b", re.IGNORECASE),
        re.compile(r"\bfor\W+the\W+last\W+one\b", re.IGNORECASE),
        re.compile(r"\bthere\W+was(?:\W+\w+)?\W+one\W+more\b", re.IGNORECASE),
        re.compile(r"\blet['’]s[\s\W]+(?:wrap\W+up|end)[\s\W]+on[\s\W]+this[\s\W]+(?:question|one)[\s\W]+then(?:\W+from)?\b", re.IGNORECASE),
        re.compile(r"\b(?:let['’]s\W+|we[\s\W]+can[\s\W]+)do[\s\W]+one[\s\W]+more\b", re.IGNORECASE),
        re.compile(r"\bdo[\s\W]+we[\s\W]+want[\s\W]+to[\s\W]+do[\s\W]+one[\s\W]+final[\s\W]+question\b", re.IGNORECASE),
        re.compile(r"\bwe[\s\W]+got[\s\W]+one[\s\W]+more(?:[\s\W]+question[\s\W]+here)?\b", re.IGNORECASE),
        re.compile(r"\bwith[\s\W]+this[\s\W]+one[\s\W]+then[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\bone\W+to\W+end\W+on\b", re.IGNORECASE),
        re.compile(r"\bwe\W+got\W+time\W+for\W+one\W+more\b", re.IGNORECASE),
        re.compile(r"\bwe\W+can\W+round\W+off\b", re.IGNORECASE),
        re.compile(r"\blet['’]s[\s\W]+uh[\s\W]+end[\s\W]+on[\s\W]+this[\s\W]+one[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\blet['’]s\W+\w+\W+end\W+on\W+this\W+one\W+from\W+\w+\W+\w+\W+who\W+says\b", re.IGNORECASE)
    ]
}



def attach_transcript_text(df: pd.DataFrame, window_sec: float = 60.0) -> pd.DataFrame:
    """
    Re-hydrates the 'text' column by pulling from the local JSON cache.
    Reconstructs the exact window of text around each timestamp.
    """
    print("Re-hydrating transcript text from cache for the sliced rows...")
    df = df.copy()
    # Initialize the column
    df["text"] = ""
    
    for video_id, group in df.groupby("video_id"):
        cache_path = os.path.join("transcript_cache", f"{video_id}.json")
        
        if not os.path.exists(cache_path):
            print(f"  ⚠️ Warning: Cache not found for {video_id}. Skipping text hydration.")
            continue
            
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Handle both your old List format and new Dict format
        snippets = data.get("snippets", data) if isinstance(data, dict) else data
        
        snippet_texts = np.array([s["text"] for s in snippets])
        snippet_starts = np.array([s["start_seconds"] for s in snippets])
        
        # Reconstruct the text window for each row
        for idx, row in group.iterrows():
            curr_time = row["timestamp_sec"]
            
            # Grab all snippets that fall within +/- window_sec of the timestamp
            mask = (snippet_starts >= (curr_time - window_sec)) & (snippet_starts < (curr_time + window_sec))
            
            if np.any(mask):
                window_text = " ".join(snippet_texts[mask]).lower()
                df.at[idx, "text"] = window_text
                
    return df

def slice_by_macro_predictions(df: pd.DataFrame, model_path: str = "boundary_classifier_five.joblib") -> pd.DataFrame:
    """
    Runs the macro model to find the predicted Q&A and Outro boundaries,
    and slices the dataframe to exactly that window.
    """
    print("Loading Macro Model to find exact boundaries...")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    features = artifact["features"]
    
    # Fill any missing features with 0 just like the training script
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0
    
    qa_slices = []
    
    for video_id, group in df.groupby("video_id"):
        ep_df = group.sort_values("timestamp_sec").copy()
        
        # 1. Predict probabilities using the Macro Model
        ep_df["prob"] = model.predict_proba(ep_df[features])[:, 1]
        
        # 2. Run your post-processor rules
        pred_df = process_episode_predictions(
            ep_df, 
            prob_col="prob",
            max_chapters=10, 
            apply_rating_boost=False
        )
        
        # 3. Extract exact timestamps using explicit labels
        qa_rows = pred_df[pred_df["label"] == "qa_boundary"]
        outro_rows = pred_df[pred_df["label"] == "outro_boundary"]
        
        if not qa_rows.empty and not outro_rows.empty:
            qa_start = qa_rows.iloc[0]["timestamp_sec"]
            outro_start = outro_rows.iloc[0]["timestamp_sec"]
            
            # Sanity check: Ensure QA doesn't accidentally happen AFTER Outro
            if qa_start >= outro_start:
                print(f"  ⚠️ [{video_id}] Q&A timestamp ({qa_start}s) is at or after Outro ({outro_start}s). Skipping to avoid corrupted training data.")
                continue
                
            print(f"  qa_start:{qa_start}")
            print(f"  outro_start:{outro_start}")
            print(f"  [{video_id}] Sliced via explicit labels: {qa_start}s to {outro_start}s")
            
            # Slice the dataframe exactly between these two points
            sliced_df = ep_df[(ep_df["timestamp_sec"] >= qa_start) & (ep_df["timestamp_sec"] <= outro_start)].copy()
            qa_slices.append(sliced_df)
        else:
            print(f"  ⚠️ [{video_id}] Missing 'qa_boundary' or 'outro_boundary' labels in post-processor output. Skipping.")

    return pd.concat(qa_slices, ignore_index=True)

def count_keywords(text: str, patterns: list) -> int:
    if not isinstance(text, str):
        return 0
    return sum(len(p.findall(text)) for p in patterns)

def enrich_qa_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Starting Q&A enrichment on {len(df)} rows...")
    df = df.copy()
    
    # 1. Raw Counts & Densities
    df["word_count"] = df["text"].apply(lambda txt: len(str(txt).split()))
    safe_wc = df["word_count"].replace(0, 1) 
    
    for kw_type, patterns in QA_KEYWORD_PATTERNS.items():
        df[f"{kw_type}_count"] = df["text"].apply(lambda txt: count_keywords(txt, patterns))
        df[f"{kw_type}_density"] = df[f"{kw_type}_count"] / safe_wc

#     # 2. Micro Rolling Windows (60 Seconds)
#     df["time_idx"] = pd.to_timedelta(df["timestamp_sec"], unit="s")
#     df = df.set_index("time_idx")
# # 
#     for kw_type in QA_KEYWORD_PATTERNS.keys():
#         col_name = f"{kw_type}_count"
#         rolling_col = f"{kw_type}_rolling_30s" # Update the name!
#         # Shrink to 30s for sharper probability spikes
#         df[rolling_col] = df.groupby("video_id")[col_name].rolling("30s").sum().reset_index(level=0, drop=True)

    # df = df.reset_index(drop=True)
    return df

if __name__ == "__main__":
    input_csv = "boundary_qa_enriched_five.csv"  # Output from run_enrichment.py !!! boundary_qa_enriched_five is QA DATASET
    output_csv = "qa_dataset_enriched_five.csv"       # New file for Q&A model training
    
    print(f"Loading macro-enriched dataset from {input_csv}...")
    df = pd.read_csv(input_csv)

    # 1. Slice to the exact predicted timestamps first
    df = slice_by_macro_predictions(df)
    
    # 2. Clean up the 'prob' column left behind by the macro model so it doesn't leak
    if "prob" in df.columns:
        df = df.drop(columns=["prob"])

    # 3. NEW: Re-hydrate the text column from the JSON cache!
    df = attach_transcript_text(df, window_sec=15.0)
    
    # 4. Run the micro-enrichment (30s rolling windows, Q&A regex)
    df = enrich_qa_dataset(df)
    
    # 5. Clean up the text column again so the new Q&A XGBoost model doesn't crash
    df = df.drop(columns=["text"])
    
    print(f"Saving final Q&A dataset ({len(df)} rows) to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("✅ Done!")