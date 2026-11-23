# llm_verifier.py
import os
import re
import time
import datetime
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
client = OpenAI()

# ---------------------------------------------------------------------------
# Helper: Date Math
# ---------------------------------------------------------------------------
def get_cutoff_date(date_str: str) -> str:
    """Subtracts 90 days from the episode date to establish the 'Current Release' window."""
    if not date_str or date_str == "Late 2022 / Early 2023":
        return "3 months prior to episode release"
    try:
        if len(date_str) == 8 and date_str.isdigit():
            dt = datetime.datetime.strptime(date_str, "%Y%m%d")
        else:
            dt = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        cutoff = dt - datetime.timedelta(days=90)
        return cutoff.strftime("%Y-%m-%d")
    except Exception:
        return "3 months prior to episode release"

# ---------------------------------------------------------------------------
# Structured Output Schemas (Pydantic)
# ---------------------------------------------------------------------------

class LeadInCheck(BaseModel):
    did_announce_new_topic: bool = Field(description="True if the hosts explicitly announce they are moving on to a new movie or topic.")
    expected_new_topic: str = Field(description="The name of the new movie or topic they announced. If none, output 'None'.")

class AnchorTopic(BaseModel):
    topic_name: str = Field(description="The primary concrete movie, question, or news item being discussed. If it's intro banter, output 'Intro'.")

class TopicPresence(BaseModel):
    is_anchor_discussed: bool = Field(description="True if the hosts EXPLICITLY discuss the Anchor Topic in this chunk.")
    is_new_topic_discussed: bool = Field(description="True if the hosts discuss a completely NEW movie review, audience question, or sustained trend/news item.")
    new_topic_name: str = Field(description="If is_new_topic_discussed is True, provide the exact name of the topic. MUST MATCH an existing topic if semantically similar.")
    reasoning: str = Field(description="Briefly explain what they are discussing in this chunk and how it follows the rules.")

class NewBoundaryClassification(BaseModel):
    new_topic_name: str = Field(description="Name of the completely new movie review, audience question, or news item.")
    is_qa_segment: bool = Field(description="True ONLY if they are explicitly reading a NEW audience question or Reddit username.")
    is_news_or_trailer: bool = Field(description="True ONLY if they are discussing industry news, casting, or trailer reactions.")

class SegmentSummary(BaseModel):
    chapter_title: str = Field(description="Natural, polished podcast chapter title (Max 6 words).")
    description: str = Field(description="A 1-2 sentence summary of the discussion and the specific directions it takes.")
    tags: List[str] = Field(default_factory=list, description="Exhaustive list of all meaningful movies, directors, and subtopics mentioned throughout the entire segment.")

# ---------------------------------------------------------------------------
# CORE PIPELINE FUNCTIONS: MICRO (Q&A)
# ---------------------------------------------------------------------------

LLM_MODEL = "gpt-4o-mini"

def get_micro_anchor_topic(text: str, cutoff_date: str, episode_title: str) -> Dict[str, Any]:
    prompt = f"""
    Read the following podcast transcript excerpt from the VERY START of a Q&A segment.
    Identify the OVERARCHING QUESTION being asked by the audience.
    
    CRITICAL RULES: 
    1. THE CUTOFF DATE RULE: Check your internal knowledge. If a mentioned movie/show was released BEFORE {cutoff_date} (e.g. 'Happy End', 'SpongeBob'), IT CANNOT BE THE TOPIC. It is merely an answer/example. Ignore it and extract the broader theme.
    2. Focus ONLY on the prompt/question itself.
    3. GUEST SPELLING: The video title is '{episode_title}'. If a guest is answering the question, use the title to flawlessly spell their name.
    4. BANNED WORDS: Do not use vague wrapper terms like 'opinions', 'perspectives', 'perceptions', or 'thoughts'. Extract the specific concrete subject.
    
    [TRANSCRIPT]
    {text}
    """
    for attempt in range(4):
        try:
            res = client.beta.chat.completions.parse(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], response_format=AnchorTopic, temperature=0.1
            )
            return res.choices[0].message.parsed.model_dump()
        except Exception as e:
            if "429" in str(e) or "50" in str(e): time.sleep((2 ** attempt) + 2)
            else: break
    return {"topic_name": "Unknown Topic"}

def check_micro_topic_presence(chunk_text: str, anchor_topic: str, expected_new_topic: str, running_topics: List[str], cutoff_date: str, previously_detected_new_topic: str) -> Dict[str, Any]:
    topics_str = ", ".join(running_topics) if running_topics else "None yet"
    prompt = f"""
    You are a podcast editor analyzing a Q&A segment. 
    The Anchor Question is: [{anchor_topic}]
    Expected New Topic (if announced): [{expected_new_topic}]
    
    Determine if they are STILL answering the Anchor Question, OR if they have read a NEW question.
    
    CRITICAL RULES:
    1. is_anchor_discussed: TRUE if they are bringing up examples/movies to explicitly ANSWER the overarching question.
    2. is_new_topic_discussed: TRUE ONLY if they explicitly read a completely NEW audience question from a user.
    3. NO SEMANTIC STRETCHING: Do not stretch the definition of the Anchor. If the core subject changes, it is a NEW topic.
    4. OLD MEDIA RULE: If they discuss an old movie/show, abstract it to the broader theme (e.g., 'Nostalgia').
    5. TOPIC CONSOLIDATION: Active new topics: [{topics_str}]. If the discussion matches one, output the EXACT string to consolidate points.
    
    [1-MINUTE CHUNK]
    {chunk_text}
    """
    for attempt in range(4):
        try:
            res = client.beta.chat.completions.parse(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], response_format=TopicPresence, temperature=0.1
            )
            return res.choices[0].message.parsed.model_dump()
        except Exception as e:
            if "429" in str(e) or "50" in str(e): time.sleep((2 ** attempt) + 2)
            else: break
    return {"is_anchor_discussed": True, "is_new_topic_discussed": False, "new_topic_name": "None", "reasoning": "API Error"}

def classify_micro_boundary(post_text: str) -> Dict[str, Any]:
    prompt = f"""
    Read the following text block occurring right after a potential Q&A topic boundary.
    
    RULES:
    1. Q&A segments are PERMITTED. If they explicitly read an audience question or Reddit username, set is_qa_segment to True.
    2. Actor names (e.g., "Chris Pratt") are perfectly acceptable as topics in Q&A.
    
    [POST-BOUNDARY TEXT]
    {post_text}
    """
    for attempt in range(4):
        try:
            res = client.beta.chat.completions.parse(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], response_format=NewBoundaryClassification, temperature=0.1
            )
            return res.choices[0].message.parsed.model_dump()
        except Exception as e:
            if "429" in str(e) or "50" in str(e): time.sleep((2 ** attempt) + 2)
            else: break
    return {"new_topic_name": "Unknown", "is_qa_segment": False, "is_news_or_trailer": False, "reasoning": "API Error"}

# ---------------------------------------------------------------------------
# CORE PIPELINE FUNCTIONS: MACRO (Movie Reviews)
# ---------------------------------------------------------------------------

def get_macro_anchor_topic(text: str, episode_date: str, cutoff_date: str, episode_title: str) -> Dict[str, Any]:
    prompt = f"""
    Read the following podcast transcript excerpt. Identify the PRIMARY movie, TV show, or specific topic being reviewed or discussed.
    
    CRITICAL RULES:
    - FRANCHISE RETROSPECTIVES: Check the episode date ({episode_date}). If they discuss multiple movies from a franchise (e.g., 'Dead Reckoning' vs 'Final Reckoning'), use your internal database to check if the newest release falls within the current window (between {cutoff_date} and {episode_date}). If YES, extract the SPECIFIC new release (e.g., 'Mission: Impossible - Dead Reckoning Part One'). If NO, it is a general retrospective; continue normally without overriding.
    - If it is a movie, include the director or year (e.g., 'Pinocchio (Guillermo del Toro)').
    - GUEST SPELLING: The video title is '{episode_title}'. Use this to flawlessly spell any guest names.
    - If the text is just podcast greeting or banter, output 'Intro'.
    - BANNED WORDS: Do not use vague wrapper terms like 'opinions', 'perspectives', or 'perceptions'. Extract concrete nouns.
    
    [TRANSCRIPT]
    {text}
    """
    for attempt in range(4):
        try:
            res = client.beta.chat.completions.parse(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], response_format=AnchorTopic, temperature=0.1
            )
            return res.choices[0].message.parsed.model_dump()
        except Exception as e:
            if "429" in str(e) or "50" in str(e): time.sleep((2 ** attempt) + 2)
            else: break
    return {"topic_name": "Unknown Topic"}

def check_macro_topic_presence(chunk_text: str, anchor_topic: str, expected_new_topic: str, running_topics: List[str], previously_detected_new_topic: str) -> Dict[str, Any]:
    topics_str = ", ".join(running_topics) if running_topics else "None yet"
    prompt = f"""
    You are a podcast editor analyzing a movie review segment. 
    The Anchor Topic is: [{anchor_topic}]
    Expected New Topic (if announced): [{expected_new_topic}]
    
    Determine if they are STILL discussing the Anchor Topic, AND/OR if they have started discussing a completely NEW topic/movie.
    
    CRITICAL RULES:
    1. CONTEXT CONTINUITY & PLOT DETAILS: In the previous minute, you detected they were discussing: [{previously_detected_new_topic}]. 
       - If they use generic pronouns ('it', 'the director') OR discuss specific plot details (e.g. 'vampire', 'telescope'), check your internal knowledge for [{previously_detected_new_topic}]! 
       - DO NOT blindly attribute random plot points to the Anchor Topic just because you are unsure. If the plot matches the new topic, they are STILL discussing the new topic.
       - If no new topic has been detected yet ([{previously_detected_new_topic}] is None), assume generic terms refer to the Anchor.
    2. NON-LINEAR INTROS / GUESTS: The hosts sometimes do housekeeping or introduce a 'special guest' in the MIDDLE of the episode. If they say 'we have a guest' or do standard banter, set is_new_topic_discussed=TRUE and name it 'Guest Interview & Banter'.
    3. Count the movies mentioned:
       - 1 or 2 films: Dedicated Movie Discussion.
       - 5+ different films rapidly: This is a Trend, News, or Oscars discussion.
    4. SEQUELS & COMPARISONS: If they discuss a new release (e.g., 'Glass Onion') and compare it to its predecessor ('Knives Out'), the overarching topic is the NEW release ('Glass Onion'). Group them under the specific new release. 
    5. ADAPTATION CONTRAST: If they discuss characters with traits that belong to a DIFFERENT adaptation (e.g., a "humorous" Baron vs the serious Dune 2021), they have pivoted to the other adaptation. Set is_anchor_discussed=FALSE and is_new_topic_discussed=TRUE.
    6. INTRO PIVOT: If the Anchor Topic is 'Intro', any shift to an actual movie means is_new_topic_discussed is True.
    7. TOPIC CONSOLIDATION: Active new topics: [{topics_str}]. If the discussion is semantically the same, output the EXACT string to consolidate points.
    
    [1-MINUTE CHUNK]
    {chunk_text}
    """
    for attempt in range(4):
        try:
            res = client.beta.chat.completions.parse(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], response_format=TopicPresence, temperature=0.1
            )
            return res.choices[0].message.parsed.model_dump()
        except Exception as e:
            if "429" in str(e) or "50" in str(e): time.sleep((2 ** attempt) + 2)
            else: break
    return {"is_anchor_discussed": True, "is_new_topic_discussed": False, "new_topic_name": "None", "reasoning": "API Error"}

def classify_macro_boundary(post_text: str, top_topics_hint: List[str], allow_qa: bool) -> Dict[str, Any]:
    hint_str = ", ".join(top_topics_hint) if top_topics_hint else "None"
    qa_instruction = "Q&A segments are PERMITTED. If they explicitly read an audience question, set is_qa_segment to True." if allow_qa else "Q&A segments are STRICTLY FORBIDDEN in this half of the podcast. Force is_qa_segment to False."
    
    prompt = f"""
    Read the following text block occurring right after a potential topic boundary.
    
    CRITICAL RESOLUTION RULES:
    1. MACRO TOPIC RESOLUTION: The minute-by-minute scan flagged these potential topics: [{hint_str}]. Use your internal database to resolve franchises (e.g., if one is a sequel to another, the true new topic is the SEQUEL). Do NOT name the topic after an actor unless it is a career retrospective.
    2. FAST-TRACK CHECKS: 
       - {qa_instruction} Look carefully for phrases like "let's do questions", "cast community" or "subreddit", or reading a community username. If found, you MUST set is_qa_segment to True.
       - If they pivot to movie announcements, box office, or trailer reactions, set is_news_or_trailer to True.
    
    [POST-BOUNDARY TEXT]
    {post_text}
    """
    for attempt in range(4):
        try:
            res = client.beta.chat.completions.parse(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], response_format=NewBoundaryClassification, temperature=0.1
            )
            return res.choices[0].message.parsed.model_dump()
        except Exception as e:
            if "429" in str(e) or "50" in str(e): time.sleep((2 ** attempt) + 2)
            else: break
    return {"new_topic_name": "Unknown", "is_qa_segment": False, "is_news_or_trailer": False, "reasoning": "API Error"}

# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------

def extract_text_window(snippets: List[Dict[str, Any]], start_sec: float, end_sec: float) -> str:
    texts = [s["text"].strip() for s in snippets if start_sec <= s["start_seconds"] < end_sec]
    return " ".join(texts)

def check_lead_in(text: str) -> Dict[str, Any]:
    prompt = f"""
    Read this 30-second transcript snippet spanning the exact moment of a potential topic transition.
    Did the hosts explicitly wrap up their current topic and announce what they are moving on to? (e.g. "Anyway, let's talk about Pinocchio").
    
    [TRANSCRIPT]
    {text}
    """
    for attempt in range(4):
        try:
            res = client.beta.chat.completions.parse(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], response_format=LeadInCheck, temperature=0.1
            )
            return res.choices[0].message.parsed.model_dump()
        except Exception as e:
            if "429" in str(e) or "50" in str(e): time.sleep((2 ** attempt) + 2)
            else: break
    return {"did_announce_new_topic": False, "expected_new_topic": "None"}

def summarize_segment_with_llm(text: str, default_title: str, segment_type: str, anchor_topic: str, episode_title: str) -> Dict[str, Any]:
    formatting_rules = f"""
    - OVERARCHING CONTEXT: The primary topic of this segment was '{anchor_topic}'. 
    - Your `chapter_title` MUST reflect this overarching topic. Do not name the chapter after a brief tangent.
    - GUEST RULE: The video title is '{episode_title}'. If the segment focuses on an interview with a guest, or them answering Q&A, you MUST include their correctly spelled name in the `chapter_title`.
    - Keep the title natural, descriptive, and punchy (Max 6 words). 
    """
    
    if segment_type == "intro":
        formatting_rules += "\n- For `chapter_title`, IGNORE routine podcast catchphrases or standard movie quotes that happen every episode. Focus on the unique banter, housekeeping, or personal stories shared."
    elif segment_type == "outro":
        formatting_rules += "\n- For `chapter_title`, output ONLY the name of the movie or media being recommended."
    elif segment_type == "qa":
        formatting_rules += "\n- For `chapter_title`, DO NOT use filler words like 'Audience Questions' or 'Films That'. Just state the core subject."

    prompt = f"""
    You are a podcast editor. Read the following complete podcast segment.
    [FULL SEGMENT TEXT]
    {text}
    
    TASK: 
    1. Provide a natural, polished chapter title. Do NOT include phrases like 'Podcast Intro', 'Intro:', or 'Discussion' in your output title. Just name the subject.
    2. Write a 1-2 sentence description summarizing the main discussion.
    3. Extract an exhaustive list of tags.
    
    {formatting_rules}
    """
    for attempt in range(4):
        try:
            res = client.beta.chat.completions.parse(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], response_format=SegmentSummary, temperature=0.1
            )
            return res.choices[0].message.parsed.model_dump()
        except Exception as e:
            if "429" in str(e) or "50" in str(e): time.sleep((2 ** attempt) + 2)
            else: break
    return {"chapter_title": default_title, "description": "API Error", "tags": []}

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def filter_candidates_with_llm(
    candidates_df: pd.DataFrame, 
    full_snippets: List[Dict[str, Any]], 
    episode_title: str,
    episode_date: str = "Late 2022 / Early 2023", 
    mode: str = "macro",
    window_sec: float = 420.0,
    scope_start_sec: float = 0.0,
    scope_end_sec: Optional[float] = None
):
    print(f"\n--- Running LLM Verification ({mode.upper()}) ---")

    verified_chapters = []
    rejected_chapters = []
    
    last_verified_timestamp = scope_start_sec 
    max_time = scope_end_sec if scope_end_sec is not None else (full_snippets[-1]["start_seconds"] + 60.0 if full_snippets else 99999.0)
    midpoint = max_time / 2.0
    
    # -------------------------------------------------------------
    # INITIALIZE STATE BEFORE THE LOOP
    # -------------------------------------------------------------
    is_qa_context = (mode.lower() == "micro")
    has_seen_qa = is_qa_context

    if mode == "micro":
        verified_chapters.append({
            "timestamp_sec": scope_start_sec, 
            "label": "qa_boundary", 
            "llm_chapter_title": "Q&A: First Question", 
            "is_qa_segment": True, 
            "anchor_topic": "Q&A"
        })
    elif scope_start_sec == 0.0:
        verified_chapters.append({
            "timestamp_sec": 0.0, 
            "label": "intro", 
            "llm_chapter_title": "Intro & Guest Interview", 
            "is_qa_segment": False, 
            "anchor_topic": "Intro"
        })

    cutoff_date = get_cutoff_date(episode_date)
    print(f"Computed Target Cutoff Date: {cutoff_date}")

    for i in range(len(candidates_df)):
        row = candidates_df.iloc[i]
        curr_cand_time = row["timestamp_sec"] 
        ml_label = row.get("label", "chapter")

        if ml_label == "outro_boundary":
            print(f"\nEvaluating Candidate at {curr_cand_time/60:.2f} min:\n  ✅ AUTO-CONFIRMED: Outro")
            row_dict = row.to_dict()
            row_dict["llm_chapter_title"] = "Outro + Recommendation"
            row_dict["is_qa_segment"] = False
            row_dict["anchor_topic"] = "Outro"
            verified_chapters.append(row_dict)
            last_verified_timestamp = curr_cand_time
            continue

        print(f"\nEvaluating Candidate at {curr_cand_time/60:.2f} min:")

        # -------------------------------------------------------------
        # STEP 1: Fast-Track Classification (News & Q&A)
        # -------------------------------------------------------------
        allow_qa = (curr_cand_time >= midpoint) and not has_seen_qa
        post_text_fast = extract_text_window(full_snippets, curr_cand_time, curr_cand_time + 60.0)
        
        if is_qa_context:
            new_boundary = classify_micro_boundary(post_text_fast)
        else:
            new_boundary = classify_macro_boundary(post_text_fast, top_topics_hint=[], allow_qa=allow_qa) 
        
        is_qa_detected = new_boundary.get("is_qa_segment", False)
        is_news_detected = new_boundary.get("is_news_or_trailer", False)
        new_title = new_boundary.get("new_topic_name", "New Chapter")
        
        if is_qa_detected or is_news_detected:
            print(f"  ✅ FAST-TRACK CONFIRMED: {new_title}")
            print(f"     [Flags] QA: {is_qa_detected} | News: {is_news_detected}")
            if is_qa_detected:
                has_seen_qa = True
            
            row_dict = row.to_dict()
            row_dict["llm_chapter_title"] = new_title
            row_dict["is_qa_segment"] = is_qa_detected
            row_dict["anchor_topic"] = new_title
            verified_chapters.append(row_dict)
            last_verified_timestamp = curr_cand_time
            continue

        # -------------------------------------------------------------
        # STEP 2: The 30-Second Transition Splice (Lead-In Check)
        # -------------------------------------------------------------
        transition_start = max(last_verified_timestamp, curr_cand_time - 15.0)
        transition_end = min(curr_cand_time + 15.0, max_time)
        transition_text = extract_text_window(full_snippets, transition_start, transition_end)
        
        leadin_data = check_lead_in(transition_text)
        expected_new_topic = leadin_data.get("expected_new_topic", "None")
        if leadin_data.get("did_announce_new_topic", False):
            print(f"  ├─ Transition Detected: Hosts announced '{expected_new_topic}'")

        # -------------------------------------------------------------
        # STEP 3: Anchor Extraction (MACRO vs MICRO)
        # -------------------------------------------------------------
        start_t1 = last_verified_timestamp
        
        if is_qa_context:
            start_t2 = min(curr_cand_time, start_t1 + 15.0)
            anchor_text = extract_text_window(full_snippets, start_t1, start_t2)
            anchor_windows_str = f"[{start_t1/60:.2f}m - {start_t2/60:.2f}m]"
            anchor_data = get_micro_anchor_topic(anchor_text, cutoff_date, episode_title)
        else:
            start_t2 = min(curr_cand_time - 15.0, start_t1 + 180.0)
            if start_t2 < start_t1: start_t2 = start_t1
            start_text = extract_text_window(full_snippets, start_t1, start_t2)
            
            end_t2 = curr_cand_time - 15.0 
            end_t1 = max(start_t2, end_t2 - 120.0)
            if end_t1 < start_t1: end_t1 = start_t1
            
            end_text = extract_text_window(full_snippets, end_t1, end_t2)
            if start_t1 >= end_t1: 
                anchor_text = start_text
                anchor_windows_str = f"[{start_t1/60:.2f}m - {start_t2/60:.2f}m]"
            else:
                anchor_text = f"[START OF SEGMENT]\n{start_text}\n...\n[END OF SEGMENT]\n{end_text}"
                anchor_windows_str = f"[{start_t1/60:.2f}m - {start_t2/60:.2f}m] AND [{end_t1/60:.2f}m - {end_t2/60:.2f}m]"
            
            anchor_data = get_macro_anchor_topic(anchor_text, episode_date, cutoff_date, episode_title)

        anchor_topic = anchor_data.get("topic_name", "Unknown")
        print(f"  ├─ Anchor Text Window(s): {anchor_windows_str}")
        print(f"  ├─ Anchor Topic: '{anchor_topic}'")

        # -------------------------------------------------------------
        # STEP 4: "Best of 5" Match (Capped at t_next)
        # -------------------------------------------------------------
        anchor_score = 0
        new_topic_scores = {}
        is_abandoned = None
        leading_new_topic = "None"
        
        t_next = candidates_df.iloc[i+1]["timestamp_sec"] if i + 1 < len(candidates_df) else max_time
        scan_end_time = min(t_next, curr_cand_time + 300.0) 
        
        total_chunks = int((scan_end_time - curr_cand_time) / 60.0)
        if total_chunks == 0 and (scan_end_time - curr_cand_time) > 10.0:
            total_chunks = 1 

        for min_offset in range(total_chunks):
            chunk_start = curr_cand_time + (min_offset * 60.0)
            chunk_end = min(chunk_start + 60.0, max_time)
            
            chunk_text = extract_text_window(full_snippets, chunk_start, chunk_end)
            if not chunk_text.strip():
                continue
                
            running_topics = list(new_topic_scores.keys())
            
            if is_qa_context:
                presence = check_micro_topic_presence(chunk_text, anchor_topic, expected_new_topic, running_topics, cutoff_date, leading_new_topic)
            else:
                presence = check_macro_topic_presence(chunk_text, anchor_topic, expected_new_topic, running_topics, leading_new_topic)
                
            is_anchor = presence.get("is_anchor_discussed", False)
            is_new = presence.get("is_new_topic_discussed", False)
            raw_new_topic_name = presence.get("new_topic_name", "None")
            
            if is_anchor: 
                anchor_score += 1
                
            if is_new and raw_new_topic_name and raw_new_topic_name.lower() != "none":
                matched_key = raw_new_topic_name.strip()
                new_topic_scores[matched_key] = new_topic_scores.get(matched_key, 0) + 1

            highest_new_score = max(new_topic_scores.values()) if new_topic_scores else 0
            leading_new_topic = max(new_topic_scores, key=new_topic_scores.get) if new_topic_scores else "None"
                
            print(f"  ├─ Minute {min_offset+1} [{chunk_start/60:.2f}m - {chunk_end/60:.2f}m]: Anchor? {is_anchor} | New? {is_new} ({raw_new_topic_name}) -> {presence.get('reasoning')}")
            print(f"  │  Score -> Anchor: {anchor_score} | Highest New ({leading_new_topic}): {highest_new_score}")
            
            time.sleep(0.5)

        # -------------------------------------------------------------
        # STEP 5: Final Evaluation
        # -------------------------------------------------------------
        print(f"  ├─ Final Tennis Match Scores: {new_topic_scores}")
        highest_new_score = max(new_topic_scores.values()) if new_topic_scores else 0
        leading_new_topics = [k for k, v in new_topic_scores.items() if v >= highest_new_score]
        
        is_abandoned = highest_new_score > anchor_score

        if anchor_score == 0 and highest_new_score >= 3:
            is_abandoned = True

        if not is_abandoned:
            print(f"  🚫 REJECTED (Anchor Maintained Relevance)")
            row_dict = row.to_dict()
            row_dict["llm_classification"] = "SIDEBAR_TANGENT"
            rejected_chapters.append(row_dict)
        else:
            allow_qa = (curr_cand_time >= midpoint) and not has_seen_qa
            post_text = extract_text_window(full_snippets, curr_cand_time, curr_cand_time + 180.0)
            
            if is_qa_context:
                new_boundary = classify_micro_boundary(post_text)
            else:
                new_boundary = classify_macro_boundary(post_text, top_topics_hint=leading_new_topics, allow_qa=allow_qa)
            
            is_qa = new_boundary.get("is_qa_segment", False)
            if is_qa:
                has_seen_qa = True
                
            title = new_boundary.get("new_topic_name", "New Chapter")
            print(f"  ✅ CONFIRMED: {title}")
            print(f"     [Flags] QA: {is_qa} | News/Trailer: {new_boundary.get('is_news_or_trailer', False)}")
            
            row_dict = row.to_dict()
            row_dict["llm_chapter_title"] = title
            row_dict["is_qa_segment"] = is_qa
            row_dict["anchor_topic"] = title 
            verified_chapters.append(row_dict)
            last_verified_timestamp = curr_cand_time
            
        time.sleep(1)

    # -------------------------------------------------------------
    # PASS 2: FULL SEGMENT SUMMARIZATION
    # -------------------------------------------------------------
    print("\n--- Starting Pass 2: Comprehensive Segment Summarization ---")

    for i in range(len(verified_chapters)):
        start_sec = verified_chapters[i]["timestamp_sec"]
        end_sec = verified_chapters[i+1]["timestamp_sec"] if (i + 1 < len(verified_chapters)) else max_time
        
        verified_chapters[i]["end_timestamp_sec"] = end_sec
        verified_chapters[i]["end_timestamp_min"] = f"{int(end_sec//60)}:{int(end_sec%60):02d}"
        
        current_label = verified_chapters[i].get("label", "chapter")
        prelim_title = verified_chapters[i].get("llm_chapter_title", "Chapter")
        segment_anchor_topic = verified_chapters[i].get("anchor_topic", prelim_title)
        
        duration = end_sec - start_sec

        # Fix for long intros (e.g. non-linear podcasts launching straight into a review)
        if current_label == "intro":
            if duration <= 180.0:
                seg_type = "intro"
            else:
                seg_type = "standard"
                # If it's standard, the initial "Intro" anchor is wrong. Dynamically extract the real one.
                if segment_anchor_topic in ["Intro", "Podcast Intro"]:
                    start_text = extract_text_window(full_snippets, start_sec, min(start_sec + 180.0, end_sec))
                    anchor_data = get_macro_anchor_topic(start_text, episode_date, cutoff_date, episode_title)
                    segment_anchor_topic = anchor_data.get("topic_name", "Podcast Discussion")
        elif current_label == "outro_boundary": 
            seg_type = "outro"
        elif current_label == "qa_boundary" or "Q&A" in prelim_title.upper(): 
            seg_type = "qa"
        else: 
            seg_type = "standard"

        is_outro = (seg_type == "outro")
        text_end_sec = min(end_sec, max_time - 60.0) if is_outro else end_sec
        text_end_sec = max(start_sec, text_end_sec)

        print(f"\nExtracting full text for segment {i+1}/{len(verified_chapters)} [{start_sec/60:.2f}m to {end_sec/60:.2f}m]...")
        
        full_chapter_text = extract_text_window(full_snippets, start_sec, text_end_sec)
        
        summary = summarize_segment_with_llm(full_chapter_text, prelim_title, segment_type=seg_type, anchor_topic=segment_anchor_topic, episode_title=episode_title)
        raw_title = summary["chapter_title"].strip()

        # Safely remove redundant intro phrasing hallucinated by LLM
        clean_title = raw_title
        prefixes_to_remove = [
            r"(?i)^Intro\s*[:-]\s*",
            r"(?i)^Intro\s+to\s+",
            r"(?i)^Podcast Intro\s*[:-]\s*",
            r"(?i)^Podcast Intro\s+and\s+",
            r"(?i)^Podcast Intro\s+",
            r"(?i)^Intro\s+"
        ]
        for pattern in prefixes_to_remove:
            clean_title = re.sub(pattern, "", clean_title).strip()
            
        if seg_type == "intro":
            raw_title = f"Intro: {clean_title}" if clean_title else "Intro"
        elif seg_type == "outro":
            raw_title = f"Outro + Recommendation: {raw_title}"
        elif seg_type == "qa" and not raw_title.upper().startswith("Q&A"):
            if raw_title.lower().startswith("films that"):
                removals = ["Audience ", "Questions "]
            else:
                removals = ["Audience ", "Questions ", "Films "]
            for word in removals:
                if raw_title.lower().startswith(word.lower()):
                    raw_title = raw_title[len(word):].strip()
            raw_title = f"Q&A: {raw_title.strip()}"
        
        verified_chapters[i]["llm_chapter_title"] = raw_title
        verified_chapters[i]["llm_description"] = summary.get("description", "")
        verified_chapters[i]["llm_tags"] = summary.get("tags", [])
        
        print(f"  ├─ Title: {raw_title}")
        print(f"  ├─ Description: {summary.get('description', '')}")
        print(f"  └─ Tags: {len(summary.get('tags', []))} extracted.")
        time.sleep(1)

    return pd.DataFrame(verified_chapters), pd.DataFrame(rejected_chapters)











    is_news_or_trailer: bool = Field(description="True ONLY if they explicitly announce they are discussing industry news or trailer reactions.")
        2. If they pivot to movie news or trailer reactions, or use one of these phrases "movie news" or "some news", set is_news_or_trailer to True.
        is_news_or_trailer: bool = Field(description="True ONLY if they are discussing industry news or trailer reactions.")