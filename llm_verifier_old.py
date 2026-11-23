# llm_verifier.py
import os
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
client = OpenAI()

# ---------------------------------------------------------------------------
# Structured Output Schemas (Pydantic)
# ---------------------------------------------------------------------------

class BoundaryDecision(BaseModel):
    # 1. EXTRACT THE SUBJECTS
    anchor_topic: str = Field(description="Name the main movie, question, or subject being discussed in the Anchor.")
    post_topic: str = Field(description="Name the main movie, question, or subject being discussed in the Post-Boundary.")
    
    # 2. FEATURE EXTRACTION FLAGS (No more 'tangent' logic)
    is_new_movie_review: bool = Field(description="True ONLY if the hosts formally shift to reviewing or recommending a COMPLETELY DIFFERENT movie (e.g., moving from reviewing 'Glass Onion' to recommending 'Mean Girls'). Note: Different adaptations like Dune 1984 vs 2021 count as different movies.")
    is_movie_news_or_trailer: bool = Field(description="True ONLY if the hosts shift to discussing industry news, box office, or trailer reactions (e.g., Megalopolis streaming release).")
    is_new_qa_question: bool = Field(description="True ONLY if the exact transition quote shows them reading a NEW audience question or Reddit username. False if they are just bringing up examples to answer the current question.")
    
    is_same_topic_continued: bool = Field(description="True if the post-boundary is fundamentally continuing the anchor topic. This includes discussing the same movie's director, or bringing up other films just to make a point about the anchor topic.")
    
    # 3. LLM DECISION
    reasoning: str = Field(description="Explain your reasoning based on these flags.")
    confidence_score: int = Field(description="Score from 0 to 100.")
    is_major_boundary: bool = Field(description="True if (is_new_movie_review OR is_movie_news_or_trailer OR is_new_qa_question) AND NOT is_same_topic_continued.")
    classification: str = Field(description="One of: COMPLETE_TOPIC_PIVOT, SAME_TOPIC_CONTINUED, Q_AND_A, NEWS_SEGMENT, NO_CHANGE")
    chapter_title: Optional[str] = Field(description="Concise chapter title if this is a major boundary.")

class SegmentSummary(BaseModel):
    chapter_title: str = Field(description="Natural, polished podcast chapter title (Max 6 words).")
    description: str = Field(description="A 1-2 sentence summary of the discussion and the specific directions it takes.")
    tags: List[str] = Field(default_factory=list, description="Exhaustive list of all meaningful movies, directors, and subtopics mentioned throughout the entire segment.")

# ---------------------------------------------------------------------------
# Core Pipeline Functions
# ---------------------------------------------------------------------------

LLM_MODEL = "gpt-4o-mini"

def summarize_segment_with_llm(text: str, default_title: str, segment_type: str = "standard") -> Dict[str, Any]:
    formatting_rules = ""
    if segment_type == "outro":
        formatting_rules = """
        CRITICAL OUTRO RULE:
        For `chapter_title`, output ONLY the name of the movie or media being recommended for the next episode. 
        Example: If they recommend the Cube films, your chapter_title must be exactly "Cube Film Series".
        DO NOT add "Discussion", "Review", or "Outro". ONLY the name of the media.
        """
    elif segment_type == "qa":
        formatting_rules = """
        TITLE FORMATTING: Make the title about the SPECIFIC questions being asked. DO NOT use generic words like "Audience", "Questions", or "Q&A".
        """
    else:
        formatting_rules = """
        TITLE FORMATTING: Keep the title natural, descriptive, and punchy (Max 6 words). 
        CRITICAL: If the segment is primarily a review of a specific movie, the chapter_title MUST contain the movie's name. Do not name the chapter after a brief side topic.
        """

    prompt = f"""
    You are a podcast editor. Read the following complete podcast segment.
    
    [FULL SEGMENT TEXT]
    {text}
    
    TASK: 
    1. Provide a natural, polished chapter title.
    2. Write a 1-2 sentence description summarizing the main discussion.
    3. Extract an exhaustive list of tags covering all meaningful subjects mentioned.
    
    {formatting_rules}
    """
    for attempt in range(4):
        try:
            response = client.beta.chat.completions.parse(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=SegmentSummary,
                temperature=0.1
            )
            return response.choices[0].message.parsed.model_dump()
        except Exception as e:
            if "429" in str(e) or "50" in str(e):
                time.sleep((2 ** attempt) + 2)
            else:
                break
    return {"chapter_title": default_title, "description": "API Error", "tags": []}

def verify_boundary_with_llm(chapter_start_text: str, pre_boundary_text: str, post_boundary_text: str, mode: str = "macro") -> Dict[str, Any]:
    if mode == "macro":
        rules_text = """
        RULES FOR VERIFICATION:
        1. NEW MOVIE REVIEWS (TRUE BOUNDARY): If the hosts officially move from reviewing/recommending Movie A to a dedicated segment on Movie B (e.g., from 'Glass Onion' to 'Mean Girls'), this is a TRUE BOUNDARY. The fact that they compare Movie B to Movie A does NOT invalidate the boundary.
        2. CONTINUING THE SAME TOPIC (FALSE POSITIVE): If the hosts are still fundamentally discussing Movie A, but bring up other movies, directors, or industry trends just to make a point or contextualize Movie A, they have NOT started a new review. This is the SAME topic.
        3. ADAPTATION SPLITS (TRUE BOUNDARY): A review of 'Dune (2021)' and a review of 'Dune (1984)' are COMPLETELY DIFFERENT movie reviews. Shifting between them is a boundary.
        4. MOVIE NEWS (TRUE BOUNDARY): Discussing industry news (e.g., Megalopolis streaming release) is a new topic.
        """
    else:
        rules_text = """
        RULES FOR VERIFICATION (Q&A SECTION):
        1. NEW QUESTION (TRUE BOUNDARY): A boundary occurs ONLY when the hosts explicitly read a NEW audience question or Reddit username. 
        2. ANSWERING THE QUESTION (FALSE POSITIVE): If the hosts bring up specific movies or directors (e.g., David Fincher) to answer the CURRENT question, they are still on the SAME topic. Do not split.
        """

    prompt = f"""
    You are an expert podcast editor for "Sardonicast". Determine if the transition point is a TRUE boundary or a FALSE POSITIVE.

    {rules_text}

    ================= [ESTABLISHED CHAPTER TOPIC (ANCHOR)] =================
    {chapter_start_text}

    ================= [TRANSITION POINT - BEFORE BOUNDARY] =================
    {pre_boundary_text}

    ================= [TRANSITION POINT - AFTER BOUNDARY] =================
    {post_boundary_text}

    TASK: Evaluate the transition using your boolean flags. Did they actually start a new review/question, or are they still on the same fundamental topic?
    """

    for attempt in range(4):
        try:
            response = client.beta.chat.completions.parse(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=BoundaryDecision,
                temperature=0.1
            )
            return response.choices[0].message.parsed.model_dump()
        except Exception as e:
            if "429" in str(e) or "50" in str(e):
                time.sleep((2 ** attempt) + 2)
            else:
                break

    return {"reasoning": "API error.", "is_major_boundary": True, "classification": "API_ERROR", "chapter_title": "Chapter", "confidence_score": 0}

def extract_text_window(snippets: List[Dict[str, Any]], start_sec: float, end_sec: float) -> str:
    texts = [s["text"].strip() for s in snippets if start_sec <= s["start_seconds"] < end_sec]
    return " ".join(texts)

def filter_candidates_with_llm(
    candidates_df: pd.DataFrame, 
    full_snippets: List[Dict[str, Any]], 
    episode_title: str,
    mode: str = "macro",
    window_sec: float = 420.0,
    scope_start_sec: float = 0.0,
    scope_end_sec: Optional[float] = None
):
    print(f"\n--- Running LLM Verification ({mode.upper()}) on {len(candidates_df)} candidates ---")

    verified_chapters = []
    rejected_chapters = []
    
    last_verified_timestamp = scope_start_sec 
    
    if scope_end_sec is not None:
        max_time = scope_end_sec
    else:
        max_time = full_snippets[-1]["start_seconds"] + 60.0 if full_snippets else 99999.0

    POST_WINDOW = 180.0

    for i in range(len(candidates_df)):
        row = candidates_df.iloc[i]
        curr_cand_time = row["coalesced_timestamp_sec"] if "coalesced_timestamp_sec" in row else row["timestamp_sec"]
        ml_label = row.get("label", "chapter")

        if ml_label == "outro_boundary":
            print(f"\nEvaluating Candidate at {curr_cand_time/60:.2f} min:\n  ✅ AUTO-CONFIRMED: Outro")
            row_dict = row.to_dict()
            row_dict["llm_chapter_title"] = "Recommendation & Outro"
            row_dict["llm_reasoning"] = "Auto-approved ML Outro label."
            verified_chapters.append(row_dict)
            last_verified_timestamp = curr_cand_time
            continue

        if i + 1 < len(candidates_df):
            next_row = candidates_df.iloc[i+1]
            t_next = next_row["coalesced_timestamp_sec"] if "coalesced_timestamp_sec" in next_row else next_row["timestamp_sec"]
        else:
            t_next = max_time

        # Pre-Boundary is the 7 minutes immediately before the transition
        pre_start = max(last_verified_timestamp, curr_cand_time - window_sec)
        pre_end = curr_cand_time
        
        # Anchor is the 7 minutes immediately before the Pre-Boundary
        anchor_start = max(last_verified_timestamp, pre_start - window_sec)
        anchor_end = pre_start

        post_start = curr_cand_time
        post_end = min(t_next, post_start + POST_WINDOW) 

        print(f"\nEvaluating Candidate at {curr_cand_time/60:.2f} min:")
        print(f"  ├─ Anchor Window: [{anchor_start/60:.2f}m - {anchor_end/60:.2f}m]")
        print(f"  ├─ Pre-Boundary:  [{pre_start/60:.2f}m - {pre_end/60:.2f}m]")
        print(f"  └─ Post-Boundary: [{post_start/60:.2f}m - {post_end/60:.2f}m]")

        chapter_start_text = extract_text_window(full_snippets, anchor_start, anchor_end)
        pre_boundary_text = extract_text_window(full_snippets, pre_start, pre_end)
        post_boundary_text = extract_text_window(full_snippets, post_start, post_end)

        decision = verify_boundary_with_llm(chapter_start_text, pre_boundary_text, post_boundary_text, mode)

        is_approved = decision.get("is_major_boundary", False)
        confidence = decision.get("confidence_score", 100)
        
        # Feature Extraction Flags
        is_new_review = decision.get("is_new_movie_review", False)
        is_news = decision.get("is_movie_news_or_trailer", False)
        is_new_qa = decision.get("is_new_qa_question", False)
        is_same_topic = decision.get("is_same_topic_continued", False)
        
        was_overridden = False

        # PYTHON HARD OVERRIDES BASED ON EXTRACTED FEATURES
        if (is_new_review or is_news or is_new_qa) and not is_same_topic:
            if not is_approved:
                is_approved = True
                was_overridden = True
                decision["reasoning"] = "[PYTHON OVERRIDE: Valid New Topic Flagged] " + decision.get("reasoning", "")
        elif is_same_topic:
            if is_approved:
                is_approved = False
                was_overridden = True
                decision["reasoning"] = "[PYTHON OVERRIDE: Same Topic Flagged] " + decision.get("reasoning", "")

        if is_approved:
            title = decision.get("chapter_title", "Chapter") or "Chapter"
            if was_overridden and (title == "Chapter" or not title):
                title = "Recovered Segment"

            is_ml_qa = (ml_label == "qa_boundary")
            is_llm_qa = is_new_qa or "audience question" in decision.get("reasoning", "").lower() or "q&a" in title.lower() or "question" in title.lower()
            
            if (is_ml_qa or is_llm_qa) and not title.upper().startswith("Q&A"):
                if title.upper().startswith("Q & A:"): title = title[6:].strip()
                title = "Q&A: " + title

            if was_overridden:
                print(f"  ⚠️ OVERRIDE EXECUTED (Check Flags)")
            else:
                print(f"  ✅ CONFIRMED: {title}")
                
            print(f"     Confidence:   {confidence}%")
            print(f"     Anchor Topic: {decision.get('anchor_topic', '')}")
            print(f"     Post Topic:   {decision.get('post_topic', '')}")
            print(f"     [Flags] NewReview: {is_new_review} | News: {is_news} | NewQA: {is_new_qa} | SameTopic: {is_same_topic}")
            print(f"     Reasoning:    {decision.get('reasoning', '')}")
            
            row_dict = row.to_dict()
            row_dict["llm_chapter_title"] = title
            row_dict["llm_reasoning"] = decision.get("reasoning", "")
            verified_chapters.append(row_dict)
            last_verified_timestamp = curr_cand_time
        else:
            print(f"  🚫 REJECTED ({decision.get('classification', '')})")
            if was_overridden:
                print(f"  ⚠️ OVERRIDE EXECUTED (Check Flags)")
            print(f"     Confidence:   {confidence}%")
            print(f"     Anchor Topic: {decision.get('anchor_topic', '')}")
            print(f"     Post Topic:   {decision.get('post_topic', '')}")
            print(f"     [Flags] NewReview: {is_new_review} | News: {is_news} | NewQA: {is_new_qa} | SameTopic: {is_same_topic}")
            print(f"     Reasoning:    {decision.get('reasoning', '')}")
            
            row_dict = row.to_dict()
            row_dict["llm_classification"] = decision.get("classification", "")
            row_dict["llm_reasoning"] = decision.get("reasoning", "")
            rejected_chapters.append(row_dict)
            
        time.sleep(1)

    # -------------------------------------------------------------
    # PASS 2: FULL SEGMENT SUMMARIZATION
    # -------------------------------------------------------------
    print("\n--- Starting Pass 2: Comprehensive Segment Summarization ---")
    
    if scope_start_sec == 0.0:
        intro_row = {
            "timestamp_sec": 0.0,
            "label": "intro",
            "llm_chapter_title": "Intro & Guest Interview",
            "llm_reasoning": "Episode start."
        }
        verified_chapters.insert(0, intro_row)
    elif mode == "micro":
        qa_intro_row = {
            "timestamp_sec": scope_start_sec,
            "label": "qa_boundary",
            "llm_chapter_title": "Q&A: First Question",
            "llm_reasoning": "Start of Q&A section."
        }
        verified_chapters.insert(0, qa_intro_row)

    for i in range(len(verified_chapters)):
        start_sec = verified_chapters[i]["timestamp_sec"]
        end_sec = verified_chapters[i+1]["timestamp_sec"] if (i + 1 < len(verified_chapters)) else max_time
        
        verified_chapters[i]["end_timestamp_sec"] = end_sec
        verified_chapters[i]["end_timestamp_min"] = f"{int(end_sec//60)}:{int(end_sec%60):02d}"
        length_sec = end_sec - start_sec
        verified_chapters[i]["length_str"] = f"{int(length_sec//60)}:{int(length_sec%60):02d}"

        is_outro = verified_chapters[i].get("label") == "outro_boundary"
        text_end_sec = min(end_sec, max_time - 60.0) if is_outro else end_sec
        text_end_sec = max(start_sec, text_end_sec)

        print(f"\nExtracting full text for segment {i+1}/{len(verified_chapters)} [{start_sec/60:.2f}m to {end_sec/60:.2f}m]...")
        
        full_chapter_text = extract_text_window(full_snippets, start_sec, text_end_sec)
        prelim_title = verified_chapters[i]["llm_chapter_title"]
        current_label = verified_chapters[i].get("label", "chapter")
        
        if current_label == "intro":
            seg_type = "intro"
        elif current_label == "outro_boundary":
            seg_type = "outro"
        elif current_label == "qa_boundary" or "Q&A" in prelim_title.upper():
            seg_type = "qa"
        else:
            seg_type = "standard"
        
        summary = summarize_segment_with_llm(full_chapter_text, prelim_title, segment_type=seg_type)
        
        raw_title = summary["chapter_title"].strip()
        
        for forbidden in [" Discussion", " Review", " Analysis", " Segment", " discussion", " review"]:
            raw_title = raw_title.replace(forbidden, "")

        if seg_type == "outro":
            final_title = f"Outro + Recommendation: {raw_title}"
        elif seg_type == "qa" and not raw_title.upper().startswith("Q&A"):
            if raw_title.upper().startswith("Q & A:"): raw_title = raw_title[6:].strip()
            
            if raw_title.lower().startswith("films that"):
                removals = ["Audience ", "Questions "]
            else:
                removals = ["Audience ", "Questions ", "Films "]
                
            for word in removals:
                if raw_title.lower().startswith(word.lower()):
                    raw_title = raw_title[len(word):].strip()
                    
            final_title = f"Q&A: {raw_title.strip()}"
        elif seg_type == "intro" and length_sec < 180.0:
            if raw_title.upper().startswith("INTRO"):
                clean_raw = raw_title.split(" ", 1)[-1].strip("-: ")
                final_title = f"Intro: {clean_raw}"
            else:
                final_title = f"Intro: {raw_title}"
        else:
            final_title = raw_title
            
        verified_chapters[i]["llm_chapter_title"] = final_title
        verified_chapters[i]["llm_description"] = summary["description"]
        verified_chapters[i]["llm_tags"] = summary["tags"]
        
        print(f"  ├─ Title: {final_title}")
        print(f"  ├─ Description: {summary['description']}")
        print(f"  └─ Tags: {len(summary['tags'])} extracted.")
        
        time.sleep(1)

    return pd.DataFrame(verified_chapters), pd.DataFrame(rejected_chapters)