import os
import sys
import time
import litellm
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from difflib import SequenceMatcher
from pydantic import BaseModel, Field

# Ensure parent directory is accessible
sys.path.append(str(Path(__file__).resolve().parent.parent))
load_dotenv()

litellm.suppress_debug_info = True

from stage1_ingestion.chapter_segmenter_schema import DetectedTransitions, EpisodeChapters
from stage1_ingestion.validation_report_pyd import ValidationReport, validate_and_correct_chapters, PASS2_SYSTEM_PROMPT

#MOD_LL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MOD_OA="openai/gpt-4o-mini"

#Extraction Rules:
# - DO NOT miss any movie title being discussed. Pay attention when hosts transition topics (e.g., "Moving on to...", "My recommendation was...", "Next up...").
# - For any movie discussion, list the exact movie title in `movies_discussed`.

# PASS 1 PROMPT: High recall for transitions
PASS1_SYSTEM_PROMPT = """
You are a transcript transition detector for the podcast 'Sardonicast'.
Your sole job is to identify EXACT timestamps where the hosts shift to a NEW topic.

Look for cues like:
- Moving from Intro to recent film reviews or news.
- Switching from one movie review to another.
- Moving to the main Episode Film Recommendation (crucial!).
- Transitioning to Listener/Community Questions (e.g., 'Let's answer some questions', 'First question from...').
- Transitioning to Next Episode Film Recommendation.
- Outro / Sign-off.

Pay special attention to silent gap timestamps provided in the prompt—they often align with topic shifts.

Output ONLY candidate transitions found in the provided transcript slice.
"""











def detect_transcript_gaps(snippets: List[Dict[str, Any]], min_gap_seconds: float = 2.0) -> List[Dict[str, Any]]:
    """Identifies silent gaps between transcript snippets that signal topic shifts."""
    gap_anchors = []
    
    for i in range(len(snippets) - 1):
        curr_end = snippets[i]["start_seconds"] + snippets[i].get("duration", 0.0)
        next_start = snippets[i + 1]["start_seconds"]
        gap = next_start - curr_end
        
        if gap >= min_gap_seconds:
            gap_anchors.append({
                "timestamp": snippets[i + 1]["timestamp"],
                "start_seconds": snippets[i + 1]["start_seconds"],
                "gap_duration": round(gap, 2)
            })
            
    return gap_anchors



def detect_raw_transitions(chunks: List[Dict[str, Any]], 
                        gap_anchors: List[Dict[str, Any]],
                        window_size: int = 15) -> List[Dict[str, Any]]:
    """PASS 1: Scans 15-minute transcript slices to collect candidate transition points."""
    candidate_transitions = []
    
    # Split chunks into ~15-minute windows
    windows = [chunks[i:i + window_size] for i in range(0, len(chunks), window_size)]
    print(f"Pass 1: Scanning {len(windows)} transcript windows for topic transitions...")

    for idx, window in enumerate(windows):
        text_slice = "\n".join([f"[{c['timestamp']}] {c['text']}" for c in window])

        # Filter gaps relevant to this window
        win_start = window[0]["start_seconds"]
        win_end = window[-1]["start_seconds"]
        win_gaps = [g for g in gap_anchors if win_start <= g["start_seconds"] <= win_end]
        gaps_text = "\n".join([f"- Gap at [{g['timestamp']}] ({g['gap_duration']}s pause)" for g in win_gaps])
        print(f"Gaps Text: {gaps_text}")

        prompt_content = f"Transcript Slice:\n{text_slice}\n\nDetected Silence Gaps in this Slice:\n{gaps_text if gaps_text else 'None'}"
        
        try:
            response = litellm.completion(
                model=MOD_OA,
                messages=[
                    {"role": "system", "content": PASS1_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_content}
                    #{"role": "user", "content": f"Scan this transcript slice and identify transitions:\n\n{text_slice}"}
                ],
                response_format=DetectedTransitions,
                temperature=0.0
            )
            
            raw_json = response.choices[0].message.content
            parsed = DetectedTransitions.model_validate_json(raw_json)
            for t in parsed.transitions:
                candidate_transitions.append(t.model_dump())
                
        except Exception as e:
            print(f"Pass 1 error in window {idx + 1}: {e}")

    return candidate_transitions


def consolidate_chapters(
    chunks: List[Dict[str, Any]], 
    #high_probability_cut_points: List[Dict[str, Any]],
    candidate_transitions: List[Dict[str, Any]],
    video_title: str = "",
    upload_date: str = ""
) -> List[Dict[str, Any]]:
    """PASS 2: Combines candidates and verifies full chapter start/end times."""
    print(f"\nPass 2: Consolidating {len(candidate_transitions)} candidate transitions into final chapters...")
    
    candidates_text = "\n".join([f"- [{t['timestamp']}] Candidate: {t['topic_hint']}" for t in candidate_transitions])
    sparse_transcript = "\n".join([f"[{c['timestamp']}] {c['text'][:100]}..." for c in chunks])

    prompt_content = f"""
        YouTube Video Metadata:
        - Title: "{video_title}"
        - Publish Date: "{upload_date}"

        Candidate Transitions Found:
        {candidates_text}

        Sparse Episode Outline:
        {sparse_transcript}

        Instructions:
        1. Use the YouTube Title ("{video_title}") and Publish Date ("{upload_date}") as primary anchors to identify the main movies reviewed in this episode.
        2. Movies released around {upload_date} or explicitly listed in the title are the main reviews. 
        3. Tangential discussions (e.g., comparing a film to Oppenheimer) should NOT replace the main review chapter (e.g., The Odyssey).
    """

    #High Probability Cut Points:
    #    {high_probability_cut_points}
      #  4. Use the High Probability Cut Points as likely timestamps where chapters end or start.
    

    try:
        response = litellm.completion(
            model=MOD_OA,
            messages=[
                {"role": "system", "content": PASS2_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content}
            ],
            response_format=EpisodeChapters,
            temperature=0.1
        )
        
        raw_json = response.choices[0].message.content
        parsed = EpisodeChapters.model_validate_json(raw_json)
        return [ch.model_dump() for ch in parsed.chapters]

    except Exception as e:
        print(f"Pass 2 Error: {e}")
        return []




def locate_exact_snippet_timestamp(quote: str, raw_snippets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Finds the precise raw caption snippet using sequence alignment (order matters!)."""
    if not quote or not raw_snippets:
        return raw_snippets[0]

    quote_clean = quote.lower().strip()
    
    # 1. First Pass: Direct Substring Search (Fast & Exact)
    for i, snippet in enumerate(raw_snippets):
        # Look for exact sequence in single snippet
        if quote_clean in snippet['text'].lower():
            return snippet
            
        # Look for exact sequence across a 2-snippet boundary (e.g., snippet i + snippet i+1)
        if i < len(raw_snippets) - 1:
            combined = f"{snippet['text']} {raw_snippets[i+1]['text']}".lower()
            if quote_clean in combined:
                # Sequence spans the boundary; return the snippet where it starts
                return snippet

    # 2. Second Pass: Sequence Alignment Ratio (Order-Aware Fuzzy Matching)
    best_snippet = raw_snippets[0]
    best_ratio = 0.0

    for i in range(len(raw_snippets)):
        # Check a sliding 2-snippet window to preserve order across caption breaks
        current_text = raw_snippets[i]['text'].lower()
        if i < len(raw_snippets) - 1:
            window_text = f"{current_text} {raw_snippets[i+1]['text'].lower()}"
        else:
            window_text = current_text

        # SequenceMatcher measures contiguous substring similarity (order matters)
        matcher = SequenceMatcher(None, quote_clean, window_text)
        # find_longest_match gives us the longest contiguous block of matching words
        match_info = matcher.find_longest_match(0, len(quote_clean), 0, len(window_text))
        
        # We score based on the length of the longest ordered sequence match
        match_score = match_info.size / len(quote_clean) if len(quote_clean) > 0 else 0

        if match_score > best_ratio:
            best_ratio = match_score
            best_snippet = raw_snippets[i]

    return best_snippet




def refine_chapter_timestamps_with_quotes(
    chapters: List[Dict[str, Any]], 
    raw_snippets: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Snaps chapter timestamps to exact raw snippets using transition quotes,
    guaranteeing strict chronological order.
    """
    if not chapters or not raw_snippets:
        return chapters

    refined_chapters = []
    last_valid_index = 0  # Force forward search only

    # 1. Pin exact start timestamps using restricted forward-search
    for ch in chapters:
        quote = ch.get("transition_quote", "")
        
        # Pass remaining snippets starting from the end of the last found chapter
        search_window = raw_snippets[last_valid_index:]
        matched_snippet = locate_exact_snippet_timestamp(quote, search_window)
        
        # Find index in full raw_snippets list
        offset = next(
            (i for i, s in enumerate(search_window) if s["start_seconds"] == matched_snippet["start_seconds"]), 
            0
        )
        
        # Advance index to prevent backwards time travel
        last_valid_index += offset

        ch["start_seconds"] = matched_snippet["start_seconds"]
        ch["start_timestamp"] = matched_snippet["timestamp"]
        refined_chapters.append(ch)

    # 2. Align end timestamps continuously (Chapter N end = Chapter N+1 start)
    for i in range(len(refined_chapters) - 1):
        # Guarantee end_seconds >= start_seconds
        if refined_chapters[i + 1]["start_seconds"] < refined_chapters[i]["start_seconds"]:
            refined_chapters[i + 1]["start_seconds"] = refined_chapters[i]["start_seconds"] + 1.0
            
        refined_chapters[i]["end_seconds"] = refined_chapters[i + 1]["start_seconds"]
        refined_chapters[i]["end_timestamp"] = refined_chapters[i + 1]["start_timestamp"]

    # 3. Final chapter end time
    if refined_chapters:
        last_snippet = raw_snippets[-1]
        last_start = last_snippet.get("start_seconds", 0.0)
        duration = last_snippet.get("duration", 5.0)
        total_end_seconds = round(last_start + duration, 2)
        
        final_h = int(total_end_seconds // 3600)
        final_m = int((total_end_seconds % 3600) // 60)
        final_s = int(total_end_seconds % 60)
        
        refined_chapters[-1]["end_seconds"] = total_end_seconds
        refined_chapters[-1]["end_timestamp"] = f"{final_h:02d}:{final_m:02d}:{final_s:02d}"

    return refined_chapters



def run_two_pass_segmentation(chunks: List[Dict[str, Any]], raw_snippets: List[Dict[str, Any]], video_title: str = "", upload_date: str = "", max_validation_retries: int = 2) -> List[Dict[str, Any]]:
    """Executes Pass 1 (Spotting) then Pass 2 (Consolidation)."""
    gap_anchors = detect_transcript_gaps(raw_snippets)
    print(f"Detected {len(gap_anchors)} silence gaps (>= 2.0s) in transcript.")

    raw_candidates = detect_raw_transitions(chunks, gap_anchors)
    #high_probability_cut_points = detect_transcript_gaps(raw_snippets)
    #print(raw_candidates)
    unrefined_chapters = consolidate_chapters(chunks, raw_candidates, video_title, upload_date)

    # 3. Post-Processing: Pin exact timestamps from raw_snippets using quotes
    #refined_chapters = refine_chapter_timestamps_with_quotes(unrefined_chapters, raw_snippets)
    current_chapters = refine_chapter_timestamps_with_quotes(unrefined_chapters, raw_snippets)

    

    # Stabilization Loop
    for attempt in range(1, max_validation_retries + 1):
        print(f"\n--- Validation Pass {attempt}/{max_validation_retries} ---")
        patched_chapters = validate_and_correct_chapters(current_chapters, chunks, video_title, upload_date)

        print(f"Num (new) patched chapters: {len(patched_chapters)}")
        print(f"Num prev) current chapters: {len(current_chapters)}")

        # If validation passed without needing patches, stop looping
        if patched_chapters is current_chapters:
            print("✅ Chapters stabilized and validated.")
            break

        # Re-snap quotes for patched chapters
        current_chapters = refine_chapter_timestamps_with_quotes(patched_chapters, raw_snippets)

    return current_chapters

    # patched_chapters = validate_and_correct_chapters(refined_chapters, chunks, video_title, upload_date)

    # print(f"Num (new) patched chapters: {len(patched_chapters)}")
    # print(f"Num prev) refined chapters: {len(refined_chapters)}")
    # # Re-snap quotes if a patch occurred to guarantee precise timestamps for inserted chapters
    # #if len(patched_chapters) != len(refined_chapters):
    # if patched_chapters is not refined_chapters:
    #     print("Re-refining timestamps for patched chapters...")
    #     final_chapters = refine_chapter_timestamps_with_quotes(patched_chapters, raw_snippets)
    # else:
    #     final_chapters = patched_chapters

    # return final_chapters



if __name__ == "__main__":
    from transcript_fetcher import fetch_youtube_transcript, fetch_youtube_metadata
    from stage1_ingestion.chunker import create_sliding_window_chunks

    test_video_ids = ["9OZAYwBtTHo"] #, "nmKCUFGq0hA", "WCA8Jt_qrG4"] #, "FXv2XUshrNE", "5rL7bBwOuio", "Ojz0sskxFBo", "kdlzHIbwQfs", "0OYydixtVA8", "AExOwJa58no", 
    # "egz-Hx4iLDw", "xVFrlN1aQIE", "7qxWAC4A0yk", "6jGjbEm99CI", "AnNI-4ihV9c", "HPVFbp_CBFA", "aQBFixFqv6M", "leOu1G1faBE", "7nlPtTUtHrw", "8nP1uXKmwSY", "6QxmbwpFRF4", 
    # "nMwkk5jozAw", "YO9wkuvTsJE", "9MTD8kOpGhk", "xtSnzvJFyYU", "4IP8hfMPzas", "0Seji1U6-4g", "V2YtTeafjvE", "Z1rsMoca8Jo", "50a38cwyvdM", "bsK6U6Lpbic", 
    # "9JIveRHOVVQ", "aGBDLj4LWEY", "kjtJIsUpoSQ", "9TVhA0iKI-8", "JzS6NDPTuuM", "7bSKhAzFusU", "ut4ZvbEnzuU", "R6oPFvxIoJk", "MwdXyumoXiE", "Dj4Mg9FeE-M", 
    # "9nkBXAyaJUQ", "yfGkuJBY2SE", "sqyyO9kIQV8", "krAxPEEpLTc", "1V1yTmH44XI", "cKIXdtntMac", "lRCDT1iMdRg", "s5_aI6T9OIQ", "cSkEMK0Z3Eo", "URfbzdgxFmE"]

    for vid_id in test_video_ids:
        TEST_VIDEO_ID = vid_id
        print(f"Fetching transcript for video ID: {TEST_VIDEO_ID}...")
        # Extract Title and Upload Date
        meta = fetch_youtube_metadata(TEST_VIDEO_ID)
        print(f"Fetched Metadata -> Title: '{meta['title']}', Date: '{meta['upload_date']}'")
        snippets = fetch_youtube_transcript(TEST_VIDEO_ID)
        chunks = create_sliding_window_chunks(snippets)

        print(f"Running Two-Pass Chapter Segmentation  with Quote Snapping using model: {MOD_OA}...")
        chapters = run_two_pass_segmentation(chunks, snippets, video_title=meta['title'], upload_date=meta['upload_date'])

        print(f"\nSuccessfully Extracted {len(chapters)} Chapters:\n")
        for ch in chapters:
            print(f"📌 [{ch['start_timestamp']} - {ch['end_timestamp']}] {ch['title']}")
            print(f"   Movies: {ch.get('movies_discussed', [])}")
            print(f"   Summary: {ch['summary']}\n")