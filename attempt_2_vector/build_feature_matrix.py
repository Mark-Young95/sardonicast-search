import time
import pandas as pd
from pathlib import Path
from typing import List

from dataset_builder import (
    fetch_youtube_metadata,
    fetch_metadata_batch_official,
    parse_description_timestamps,
    fetch_and_clean_transcript,
    detect_transcript_gaps,
    label_transcript_chunks
)
from feature_extractor import extract_window_features

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def process_video_ids_draft(video_ids: List[str]) -> pd.DataFrame:
    """Main batch processing pipeline."""
    all_rows = []

    metas = fetch_metadata_batch_official(video_ids)
    # gt_chapters = parse_description_timestamps(metas["description"])
        
    # if not gt_chapters:
    #     print(f"  ⚠️ Error occurred: No timestamps found in description.")
    
    #for idx, raw_id in enumerate(video_ids, start=1):
    for idx, meta in enumerate(metas, start=1):
        #video_id = raw_id.split("v=")[-1].split("&")[0] if "v=" in raw_id else raw_id
        #print(f"[{idx}/{len(video_ids)}] Processing Video ID: {video_id}...")

        # 1. Fetch Metadata (Title, Date, Description)
        #meta = fetch_youtube_metadata(video_id)
        gt_chapters = parse_description_timestamps(meta["description"])
        
        if not gt_chapters:
            print(f"  ⚠️ Error occurred: No timestamps found in description.")
            continue

        # print(f"Parsed {len(gt_chapters)} Ground Truth Chapters from Description:")
        # for ch in gt_chapters:
        #     print(f"  [{ch['timestamp_str']}] {ch['title']} (Ends: {ch.get('end_seconds', 'N/A')}s)")

    # 2. Fetch Transcript
    for idx, raw_id in enumerate(video_ids, start=1):
        video_id = raw_id.split("v=")[-1].split("&")[0] if "v=" in raw_id else raw_id
        print(f"[{idx}/{len(video_ids)}] Processing Video ID: {video_id}...")
        snippets = fetch_and_clean_transcript(video_id)
        if not snippets:
            print(f"  ⚠️ Skipping {video_id}: Could not fetch transcript.")
            continue

        # 3. Detect Audio Silence Gaps
        gaps = detect_transcript_gaps(snippets, min_gap_seconds=2.0)

        # 4. Label Window Chunks
        samples = label_transcript_chunks(snippets, gt_chapters, chunk_window_sec=30.0)

        # 5. Extract Features (incorporating video_title & upload_date)
        feats = extract_window_features(
            dataset_samples=samples, 
            gap_anchors=gaps, 
            video_title=meta["title"], 
            upload_date=meta["upload_date"]
        )

        for f in feats:
            f["video_id"] = video_id
            all_rows.append(f)
        print(f"*** DONE {idx} ***")

    return pd.DataFrame(all_rows)

def process_video_ids(video_ids: List[str]) -> pd.DataFrame:
    # Clean video IDs
    clean_ids = [raw.split("v=")[-1].split("&")[0] if "v=" in raw else raw for raw in video_ids]
    
    # 1. Fetch Metadata in Batch via Official API (1 Quota Unit per 50 videos)
    print("Fetching batch metadata from YouTube API...")
    batch_metadata = fetch_metadata_batch_official(clean_ids)
    all_rows = []
    
    for idx, video_id in enumerate(clean_ids, start=1):
        print(f"[{idx}/{len(clean_ids)}] Processing Video ID: {video_id}...")

        # 2. Get Metadata (Fallback to single fetch if missing from batch response)
        meta = batch_metadata.get(video_id) or fetch_youtube_metadata(video_id)
        gt_chapters = parse_description_timestamps(meta.get("description", ""))
        
        if not gt_chapters:
            print(f"  ⚠️ Skipping {video_id}: No timestamps found in description.")
            continue

        # 3. Fetch Transcript (reads from data/transcripts_cache if available)
        snippets = fetch_and_clean_transcript(video_id)
        if not snippets:
            print(f"  ⚠️ Skipping {video_id}: Could not retrieve transcript.")
            continue

        # 4. Detect Audio Gaps
        gaps = detect_transcript_gaps(snippets, min_gap_seconds=2.0)

        # 5. Label Transcript Windows
        samples = label_transcript_chunks(snippets, gt_chapters, chunk_window_sec=30.0)

        # 6. Extract Feature Vectors
        feats = extract_window_features(
            dataset_samples=samples, 
            gap_anchors=gaps, 
            video_title=meta.get("title", ""), 
            upload_date=meta.get("upload_date", "")
        )

        for f in feats:
            f["video_id"] = video_id
            all_rows.append(f)

        # Periodic checkpoint pause every 10 videos (only pauses if fetching fresh non-cached data)
        if idx % 10 == 0 and idx < len(clean_ids):
            print("  ☕ Batch checkpoint reached. Pausing 5 seconds...")
            time.sleep(5)

    return pd.DataFrame(all_rows)

def process_testing_or_validation_video_ids(video_ids: List[str]) -> pd.DataFrame:
    # Clean video IDs
    clean_ids = [raw.split("v=")[-1].split("&")[0] if "v=" in raw else raw for raw in video_ids]
    all_rows = []
    
    for idx, video_id in enumerate(clean_ids, start=1):
        print(f"[{idx}/{len(clean_ids)}] Processing Video ID: {video_id}...")

        # 3. Fetch Transcript (reads from data/transcripts_cache if available)
        snippets = fetch_and_clean_transcript(video_id)
        if not snippets:
            print(f"  ⚠️ Skipping {video_id}: Could not retrieve transcript.")
            continue

        for f in snippets:
            f["video_id"] = video_id
            all_rows.append(f)

        # Periodic checkpoint pause every 10 videos (only pauses if fetching fresh non-cached data)
        if idx % 10 == 0 and idx < len(clean_ids):
            print("  ☕ Batch checkpoint reached. Pausing 5 seconds...")
            time.sleep(5)

    return pd.DataFrame(all_rows)

if __name__ == "__main__":
    # Test batch
    #test_episodes = ["9OZAYwBtTHo", "bsK6U6Lpbic"]  
    #test_video_ids_1_50 = ["8zNF77wosvQ", "WpC75FsTKgI", "XknVaaXJY_k", "djAw3gYe0eA", "MH0_ccLF_wQ", "jhsLkaovRf4", "aVbILMqNxgQ", "Xdrf3vy7Edk", "A0Wy9u6DPj8", "kTElzwepV88", "9dyGSglqflk", "yp8GigxUc8A", "JtEkJ22uDAY", "MCwW-vnG5S8", "GDiZzicN71w", "1PaTzu-auA8", "bAw40bYAbuQ", "KprWFzZHuvs", "0rx44FdnEIY", "WW2XiHenNTI", "dZrdd2MmkNs", "obNj3gaJros", "n8uX0X9eiVg", "cis2raB_Bdo", "Xaal5pQ-Zl0", "Nf2wmGN3mkQ", "QKA13IdBZd4", "Ce3IJL3pPwg", "cTNEIRS50D8", "vxObcVWDpJY", "FGNmMheGw8c", "1s5SWZNR0Z4", "QCv6CJNVJi8", "fExN7E8vtZ0", "Ds_FVmolrrM", "nnC-qeutqFM", "ZEz20RdBlQA", "Q_jR1U82wvc", "f-OmVBjIC5M", "jdfSNUgJxCI", "a9lovmzo94g", "g66tJLCQmlQ", "leN-vjrW1eU", "I7IycKwKZ0c", "VqoBXdHw87g", "QQ7X33bzLBQ", "GkAxbUVqHVQ", "STFAcONCXRM", "W43k4lgx6pM", "ym-hIVRp-4Q"] 
    #test_video_ids_51_100 = ["Wl-4iuEgXoo", "01Fcmcwk15E", "9FqJN-6iq9I", "stCaFJ77Nc0", "TVZBFjs8EAI", "NWRf6vTheB4", "0dRTNWui2qM", "rM6qPSeEvcA", "Tuiv84ZDRyM", "Xuu-qqq25hU", "RFetquVwQiE", "a4YvDSgzUYg", "cS0jXqcBZFg", "dpwbfTta0XE", "6O5bEy7_bSM", "YXdqlGFOmNw", "WZP1ab8i9nk", "stfN4kzaDcw", "PWyWirZiHsg", "VfBes3X1ovk", "LT_rSGNHHtA", "b7OrCQP5EC4", "anKdXQ-kfyc", "bvxlnZ5UFcs", "AuUHVdAKS_Y", "_bvT65wMeoE", "mPWvB0WEwZw", "NP-YRmcWzqE", "bUKJkaLWeGw", "abxgAZ2a-RQ", "ANsRcUCwrbI", "Zb9bB0t4tyU", "lScrhK9Za6k", "e4e0e7APJT0", "tVG-PoH-n80", "AhY_XvQt2KU", "BZLbkzMX57Y", "NiTtWpPrVp4", "kRQHSycmvd0", "1MVRQj2vr9Q", "SYjANODJkzc", "u4_9_elgJyg", "4IP8hfMPzas", "xtSnzvJFyYU", "9MTD8kOpGhk", "nMwkk5jozAw", "6QxmbwpFRF4", "8nP1uXKmwSY", "7nlPtTUtHrw", "leOu1G1faBE"]
    #test_video_ids_101 = ["aQBFixFqv6M", "HPVFbp_CBFA", "AnNI-4ihV9c", "6jGjbEm99CI", "7qxWAC4A0yk", "xVFrlN1aQIE", "egz-Hx4iLDw", "AExOwJa58no", "0OYydixtVA8", "kdlzHIbwQfs", "Ojz0sskxFBo", "5rL7bBwOuio", "FXv2XUshrNE", "WCA8Jt_qrG4", "nmKCUFGq0hA", "9OZAYwBtTHo"]
    
    #test_video_ids = ["9OZAYwBtTHo", "nmKCUFGq0hA", "WCA8Jt_qrG4"]
    
    #test_video_ids = test_video_ids_1_50
    #["8zNF77wosvQ", "WpC75FsTKgI", "XknVaaXJY_k", "djAw3gYe0eA", "MH0_ccLF_wQ", "jhsLkaovRf4", "aVbILMqNxgQ", "Xdrf3vy7Edk", "A0Wy9u6DPj8", "kTElzwepV88", "9dyGSglqflk", "yp8GigxUc8A"] 
    #["JtEkJ22uDAY", "MCwW-vnG5S8", "GDiZzicN71w", "1PaTzu-auA8", "bAw40bYAbuQ", "KprWFzZHuvs", "0rx44FdnEIY", "WW2XiHenNTI", "dZrdd2MmkNs", "obNj3gaJros", "n8uX0X9eiVg", "cis2raB_Bdo", "Xaal5pQ-Zl0", "Nf2wmGN3mkQ", "QKA13IdBZd4", "Ce3IJL3pPwg"]
    #["cTNEIRS50D8", "vxObcVWDpJY", "FGNmMheGw8c", "1s5SWZNR0Z4", "QCv6CJNVJi8", "fExN7E8vtZ0", "Ds_FVmolrrM", "nnC-qeutqFM", "ZEz20RdBlQA", "Q_jR1U82wvc", "f-OmVBjIC5M", "jdfSNUgJxCI", "a9lovmzo94g", "g66tJLCQmlQ"]
    #["leN-vjrW1eU", "I7IycKwKZ0c", "VqoBXdHw87g", "QQ7X33bzLBQ", "GkAxbUVqHVQ", "STFAcONCXRM", "W43k4lgx6pM", "ym-hIVRp-4Q", "Wl-4iuEgXoo", "01Fcmcwk15E", "9FqJN-6iq9I", "stCaFJ77Nc0", "TVZBFjs8EAI", "NWRf6vTheB4"]
    #["0dRTNWui2qM", "rM6qPSeEvcA", "Tuiv84ZDRyM", "Xuu-qqq25hU", "RFetquVwQiE", "a4YvDSgzUYg", "cS0jXqcBZFg", "dpwbfTta0XE", "6O5bEy7_bSM", "YXdqlGFOmNw", "WZP1ab8i9nk", "stfN4kzaDcw", "PWyWirZiHsg"] 
    #["VfBes3X1ovk", "LT_rSGNHHtA", "b7OrCQP5EC4", "anKdXQ-kfyc", "bvxlnZ5UFcs", "AuUHVdAKS_Y", "_bvT65wMeoE", "mPWvB0WEwZw", "NP-YRmcWzqE", "bUKJkaLWeGw", "abxgAZ2a-RQ", "ANsRcUCwrbI", "Zb9bB0t4tyU"]
    #["lScrhK9Za6k", "e4e0e7APJT0", "tVG-PoH-n80", "AhY_XvQt2KU", "BZLbkzMX57Y", "NiTtWpPrVp4", "kRQHSycmvd0", "1MVRQj2vr9Q", "SYjANODJkzc", "u4_9_elgJyg", "4IP8hfMPzas", "xtSnzvJFyYU", "9MTD8kOpGhk"]
    #["nMwkk5jozAw", "6QxmbwpFRF4", "8nP1uXKmwSY", "7nlPtTUtHrw", "leOu1G1faBE", "aQBFixFqv6M", "HPVFbp_CBFA", "AnNI-4ihV9c", "6jGjbEm99CI", "7qxWAC4A0yk", "xVFrlN1aQIE", "egz-Hx4iLDw", "AExOwJa58no"]
    #["0OYydixtVA8", "kdlzHIbwQfs", "Ojz0sskxFBo", "5rL7bBwOuio", "FXv2XUshrNE", "WCA8Jt_qrG4", "nmKCUFGq0hA", "9OZAYwBtTHo", "KprWFzZHuvs", "WW2XiHenNTI", "g66tJLCQmlQ", "NWRf6vTheB4"]
    #
    # Failures:
    # Transcript unavailable for KprWFzZHuvs.
    # Transcript unavailable for WW2XiHenNTI.
    # Attempt 1/3 failed for g66tJLCQmlQ: 
    # Attempt 3/3 failed for NWRf6vTheB4: 

    # [9/12] Processing Video ID: KprWFzZHuvs...
# cache not found: KprWFzZHuvs
#   ⚠️ Transcript unavailable for KprWFzZHuvs.
# cache not found: KprWFzZHuvs
#   ⚠️ Transcript unavailable for KprWFzZHuvs.
#   ⚠️ Skipping KprWFzZHuvs: Could not retrieve transcript.
#   ⚠️ Transcript unavailable for KprWFzZHuvs.
#   ⚠️ Skipping KprWFzZHuvs: Could not retrieve transcript.
# [10/12] Processing Video ID: WW2XiHenNTI...
#   ⚠️ Skipping KprWFzZHuvs: Could not retrieve transcript.
# [10/12] Processing Video ID: WW2XiHenNTI...
# [10/12] Processing Video ID: WW2XiHenNTI...
# cache not found: WW2XiHenNTI
#   ⚠️ Transcript unavailable for WW2XiHenNTI.
#   ⚠️ Skipping WW2XiHenNTI: Could not retrieve transcript.

    # TRAINING IDS (114)
    #test_video_ids =  ["8zNF77wosvQ", "WpC75FsTKgI", "XknVaaXJY_k", "djAw3gYe0eA", "MH0_ccLF_wQ", "jhsLkaovRf4", "aVbILMqNxgQ", "Xdrf3vy7Edk", "A0Wy9u6DPj8", "kTElzwepV88", "9dyGSglqflk", "yp8GigxUc8A", "JtEkJ22uDAY", "MCwW-vnG5S8", "GDiZzicN71w", "1PaTzu-auA8", "bAw40bYAbuQ", "0rx44FdnEIY", "dZrdd2MmkNs", "obNj3gaJros", "n8uX0X9eiVg", "cis2raB_Bdo", "Xaal5pQ-Zl0", "Nf2wmGN3mkQ", "QKA13IdBZd4", "Ce3IJL3pPwg", "cTNEIRS50D8", "vxObcVWDpJY", "FGNmMheGw8c", "1s5SWZNR0Z4", "QCv6CJNVJi8", "fExN7E8vtZ0", "Ds_FVmolrrM", "nnC-qeutqFM", "ZEz20RdBlQA", "Q_jR1U82wvc", "f-OmVBjIC5M", "jdfSNUgJxCI", "a9lovmzo94g", "g66tJLCQmlQ", "leN-vjrW1eU", "I7IycKwKZ0c", "VqoBXdHw87g", "QQ7X33bzLBQ", "GkAxbUVqHVQ", "STFAcONCXRM", "W43k4lgx6pM", "ym-hIVRp-4Q", "Wl-4iuEgXoo", "01Fcmcwk15E", "9FqJN-6iq9I", "stCaFJ77Nc0", "TVZBFjs8EAI", "NWRf6vTheB4", "0dRTNWui2qM", "rM6qPSeEvcA", "Tuiv84ZDRyM", "Xuu-qqq25hU", "RFetquVwQiE", "a4YvDSgzUYg", "cS0jXqcBZFg", "dpwbfTta0XE", "6O5bEy7_bSM", "YXdqlGFOmNw", "WZP1ab8i9nk", "stfN4kzaDcw", "PWyWirZiHsg", "VfBes3X1ovk", "LT_rSGNHHtA", "b7OrCQP5EC4", "anKdXQ-kfyc", "bvxlnZ5UFcs", "AuUHVdAKS_Y", "_bvT65wMeoE", "mPWvB0WEwZw", "NP-YRmcWzqE", "bUKJkaLWeGw", "abxgAZ2a-RQ", "ANsRcUCwrbI", "Zb9bB0t4tyU", "lScrhK9Za6k", "e4e0e7APJT0", "tVG-PoH-n80", "AhY_XvQt2KU", "BZLbkzMX57Y", "NiTtWpPrVp4", "kRQHSycmvd0", "1MVRQj2vr9Q", "SYjANODJkzc", "u4_9_elgJyg", "4IP8hfMPzas", "xtSnzvJFyYU", "9MTD8kOpGhk", "nMwkk5jozAw", "6QxmbwpFRF4", "8nP1uXKmwSY", "7nlPtTUtHrw", "leOu1G1faBE", "aQBFixFqv6M", "HPVFbp_CBFA", "AnNI-4ihV9c", "6jGjbEm99CI", "7qxWAC4A0yk", "xVFrlN1aQIE", "egz-Hx4iLDw", "AExOwJa58no", "0OYydixtVA8", "kdlzHIbwQfs", "Ojz0sskxFBo", "5rL7bBwOuio", "FXv2XUshrNE", "WCA8Jt_qrG4", "nmKCUFGq0hA", "9OZAYwBtTHo"]



    # TESTING IDS (23)
    # ["9nkBXAyaJUQ", "bsK6U6Lpbic", "Z1rsMoca8Jo", "V2YtTeafjvE", "0Seji1U6-4g", "M_XUa3L8Z4E", "_w5b_rSMZKA", "abYG_qkqhHk", "KHL6MIhGGL4", "qk6gQ0Be5ac", "aMSwB0hx4CE", "L2yoBj6k2gw"]
    # ["zxUEH_Nak8Y", "rXb8vjfszLY", "yoINw3EkUg8", "igEAXji4CK8", "rBlyN8R6zFU", "99kau024yso", "rf2FNwHif_I", "BJLFZhHN3Ls", "2y77PyEVsUY", "t9O3Rtfqrkg", "ZgJCKrCrIvs"]

    # VALIDATION IDS (22)
    # ["ohzJhlzULU8", "RxrhZNcbPss", "JjVLu6gW1QA", "p26Jg_0aZME", "joOYxzdUhAk", "K6qLLeiOwuA", "RLm3t3V-0Ys", "kSuuuxolw-U", "HjWABCVmzU0", "74RSO6AKcIY", "LArpDeGO7L4"]
    #test_video_ids = ["m5BkmRovPfQ", "Kgiw3C2e7W0", "B4K72jClGhE", "04K88bwT558", "S4gLhgS_Btc", "3YKhtvSLvLo", "WELb_tDvSf0", "UxaoKwWDUvI", "wIkPmEqewRs", "HnDnmxspTFo", "YO9wkuvTsJE"]
    test_video_ids = ["99kau024yso"]

    
    #df = process_video_ids(test_video_ids)
    df = process_testing_or_validation_video_ids(test_video_ids)
    # print(f"\nExtracted total {len(df)} feature rows across {len(test_video_ids)} episodes.")



    # # END HERE for TESTING and VALIDATION
    # # Save outputs
    # csv_path = DATA_DIR / "sardonicast_features.csv"
    # parquet_path = DATA_DIR / "sardonicast_features.parquet"
    
    # df.to_csv(csv_path, index=False)
    # df.to_parquet(parquet_path, index=False)
    
    # print(f"Saved dataset matrices:\n - {csv_path}\n - {parquet_path}")