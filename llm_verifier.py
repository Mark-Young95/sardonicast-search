# llm_verifier.py
import os
import re
import time
import datetime
import pandas as pd
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
client = OpenAI()

# Using GPT-5.4-Mini for 2026 film knowledge & superior logic parsing
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")

# Toggle this to False in final one-time production if you want to send full 60-min transcripts
USE_STRIDED_PASS2_SAMPLING = True
STRIDED_THRESHOLD_SEC = 360.0  # 6 minutes

# ---------------------------------------------------------------------------
# Core Helpers
# ---------------------------------------------------------------------------
def get_cutoff_date(date_str: str, days: int = 90) -> str:
    if not date_str or date_str == "Late 2022 / Early 2023":
        return f"{days // 30} months prior to episode release"
    try:
        dt = (
            datetime.datetime.strptime(date_str, "%Y%m%d")
            if len(date_str) == 8 and date_str.isdigit()
            else datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        )
        return (dt - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    except Exception:
        return f"{days // 30} months prior to episode release"

def extract_text_window(snippets: List[Dict[str, Any]], start_sec: float, end_sec: float) -> str:
    return " ".join([s["text"].strip() for s in snippets if start_sec <= s["start_seconds"] < end_sec])

def _call_structured_llm(
    user_prompt: str,
    schema_model: Any,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1
) -> Optional[Dict[str, Any]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    for attempt in range(4):
        try:
            res = client.beta.chat.completions.parse(
                model=LLM_MODEL,
                messages=messages,
                response_format=schema_model,
                temperature=temperature
            )
            return res.choices[0].message.parsed.model_dump()
        except Exception as e:
            if "429" in str(e) or "50" in str(e):
                time.sleep((2 ** attempt) + 2)
            else:
                break
    return None

def normalize_text(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', text.lower())

# ===========================================================================
# POST-VALIDATION BACKWARDS Q&A BOUNDARY DETECTOR
# ===========================================================================

def score_qa_window_90s(
    snippets: List[Dict[str, Any]],
    cand_sec: float,
    max_time: float,
    xgb_prob: float = 0.55
) -> Tuple[float, int, int, Dict[str, int]]:
    """
    Captures [-90s, +90s] around a boundary and counts non-overlapping occurrences of the 6 Q&A keyphrases:
      1. "some questions"
      2. "questions" (counted only outside of "some questions")
      3. "subreddit"
      4. "reddit" (counted only outside of "subreddit")
      5. "sardonicast" (including YouTube caption splits like "sardoni cast")
      6. "thread"

    Returns:
      (qa_probability, unique_keyphrase_count, total_keyword_hits, matched_counts_dict)
    """
    win_start = max(0.0, cand_sec - 90.0)
    win_end = min(max_time, cand_sec + 90.0)
    raw_text = extract_text_window(snippets, win_start, win_end).lower()

    counts: Dict[str, int] = {}

    # 1. Match "some questions" first, then mask it so standalone "questions" doesn't double-count it
    some_q_matches = re.findall(r"\bsome\s+questions\b", raw_text)
    if some_q_matches:
        counts["some questions"] = len(some_q_matches)
    masked_text = re.sub(r"\bsome\s+questions\b", " ", raw_text)

    # 2. Match standalone "questions" (or "question")
    q_matches = re.findall(r"\bquestions\b", masked_text)
    if q_matches:
        counts["questions"] = len(q_matches)

    # 3. Match "subreddit" (word-bounded so it never triggers "reddit")
    sub_matches = re.findall(r"\bsubreddit\b", raw_text)
    if sub_matches:
        counts["subreddit"] = len(sub_matches)

    # 4. Match standalone "reddit"
    reddit_matches = re.findall(r"\breddit\b", raw_text)
    if reddit_matches:
        counts["reddit"] = len(reddit_matches)

    # 5. Match "sardonicast" (plus common YouTube auto-caption phonetic splits)
    sard_matches = re.findall(r"\b(sardonicast|sardoni\s*cast|sardonic\s*cast)\b", raw_text)
    if sard_matches:
        counts["sardonicast"] = len(sard_matches)

    # 6. Match "thread"
    thread_matches = re.findall(r"\bthread\b", raw_text)
    if thread_matches:
        counts["thread"] = len(thread_matches)

    unique_count = len(counts)
    total_hits = sum(counts.values())

    # Require at least 2 unique keyphrases to reach high probability
    if unique_count < 2:
        qa_prob = min(0.45, 0.20 * unique_count + 0.05 * total_hits)
    else:
        # Base 0.82 for 2 unique keyphrases + 0.05 per extra unique phrase + 0.03 per extra hit + XGB tiebreaker
        qa_prob = 0.82 + 0.05 * (unique_count - 2) + 0.03 * max(0, total_hits - 2) + 0.05 * max(0.0, xgb_prob - 0.50)
        qa_prob = min(0.99, qa_prob)

    return qa_prob, unique_count, total_hits, counts

def assign_qa_boundary_backwards(
    verified: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    full_snippets: List[Dict[str, Any]],
    max_time: float,
    prob_threshold: float = 0.90
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Post-validation backwards Q&A boundary selector:
      1. Resets any tentative Q&A flags so only ONE authoritative Macro Q&A start boundary is chosen.
      2. Starts from the last non-outro boundary (or the very last boundary if no outro was captured)
         and walks backwards 1-by-1 through boundaries in the 2nd half (>= 50% of episode).
      3. Locks in the first boundary with >= 2 unique keyphrases and qa_prob >= 90% in [-90s, +90s].
      4. If no verified boundary hits >= 90%, checks rejected 2nd-half boundaries backwards for >= 90%.
      5. If still none hit >= 90%, falls back to the highest-probability verified boundary in the 2nd half.
    """
    print("\n=== Running Post-Validation Backwards Q&A Boundary Detection ([-90s, +90s] Window) ===")
    half_time = max_time * 0.50

    # Ensure chronological sort
    verified = sorted(verified, key=lambda x: float(x["timestamp_sec"]))

    # Identify if the final boundary is an Outro (either by label or terminal position > 92% with outro title)
    outro_idx = None
    if len(verified) > 1:
        last_row = verified[-1]
        last_ts = float(last_row["timestamp_sec"])
        if last_row.get("label") == "outro_boundary" or (last_ts / max_time >= 0.92 and "outro" in str(last_row.get("llm_chapter_title", "")).lower()):
            outro_idx = len(verified) - 1

    # Clear tentative Q&A flags on verified rows before assigning the single authoritative Q&A start
    llm_flagged_qa_indices = set()
    for idx, row in enumerate(verified):
        if row.get("is_qa_segment") or str(row.get("llm_chapter_title", "")).upper().startswith("Q&A"):
            llm_flagged_qa_indices.add(idx)
        if idx != outro_idx and row.get("label") != "intro":
            row["label"] = "chapter"
            row["is_qa_segment"] = False

    # Build backwards search indices: start at second-to-last if last is outro, else start at very last boundary!
    start_idx = (len(verified) - 2) if outro_idx is not None else (len(verified) - 1)
    chosen_qa_idx: Optional[int] = None

    # PASS A: Walk backwards 1-by-1 through verified boundaries in the 2nd half
    for idx in range(start_idx, 0, -1):
        cand = verified[idx]
        ts = float(cand["timestamp_sec"])
        if ts < half_time:
            break  # Reached the first half of the episode; stop backwards scan

        xgb_p = float(cand.get("prob", 0.55))
        qa_prob, uniq_cnt, tot_hits, match_dict = score_qa_window_90s(full_snippets, ts, max_time, xgb_p)
        print(
            f"  [Backwards Check @ {ts/60:.2f}m] Unique Keyphrases: {uniq_cnt} | "
            f"Total Hits: {tot_hits} {match_dict} -> QA Prob: {qa_prob*100:.1f}%"
        )

        if uniq_cnt >= 2 and qa_prob >= prob_threshold:
            chosen_qa_idx = idx
            print(f"  ✅ LOCKED Q&A BOUNDARY (Backwards Keyword Match >= 90%) at {ts/60:.2f}m!")
            break

    # PASS B: What if the true Q&A boundary was rejected by the forward LLM pass, but has >= 90% keyword certainty?
    if chosen_qa_idx is None and rejected:
        outro_sec = float(verified[outro_idx]["timestamp_sec"]) if outro_idx is not None else max_time
        rejected_2nd_half = sorted(
            [r for r in rejected if half_time <= float(r["timestamp_sec"]) < (outro_sec - 45.0)],
            key=lambda x: float(x["timestamp_sec"]),
            reverse=True
        )
        for r_cand in rejected_2nd_half:
            ts = float(r_cand["timestamp_sec"])
            xgb_p = float(r_cand.get("prob", 0.55))
            qa_prob, uniq_cnt, tot_hits, match_dict = score_qa_window_90s(full_snippets, ts, max_time, xgb_p)
            if uniq_cnt >= 2 and qa_prob >= prob_threshold:
                print(
                    f"  🛟 RESCUED REJECTED Q&A BOUNDARY at {ts/60:.2f}m "
                    f"(Unique: {uniq_cnt}, Hits: {tot_hits} {match_dict}, QA Prob: {qa_prob*100:.1f}%)"
                )
                rescued_row = dict(r_cand) | {
                    "label": "qa_boundary",
                    "llm_chapter_title": "Q&A",
                    "is_qa_segment": True,
                    "anchor_topic": "Q&A"
                }
                verified.append(rescued_row)
                verified = sorted(verified, key=lambda x: float(x["timestamp_sec"]))
                rejected = [r for r in rejected if float(r["timestamp_sec"]) != ts]
                chosen_qa_idx = next(i for i, v in enumerate(verified) if float(v["timestamp_sec"]) == ts)
                break

    # PASS C: Fallback to the highest-probability boundary in the 2nd half of the episode
    if chosen_qa_idx is None:
        second_half_indices = [
            idx for idx in range(1, start_idx + 1)
            if float(verified[idx]["timestamp_sec"]) >= half_time
        ]
        if second_half_indices:
            # Pick highest XGBoost probability boundary in the 2nd half
            chosen_qa_idx = max(second_half_indices, key=lambda idx: float(verified[idx].get("prob", 0.0)))
            ts = float(verified[chosen_qa_idx]["timestamp_sec"])
            p_val = float(verified[chosen_qa_idx].get("prob", 0.0))
            print(f"  ⚠️ No boundary hit >= 90% keyword threshold. Fell back to highest-prob 2nd-half boundary at {ts/60:.2f}m (prob={p_val:.3f}).")
        elif start_idx >= 1:
            # Extreme fallback if zero verified boundaries exist after 50%
            chosen_qa_idx = start_idx
            ts = float(verified[chosen_qa_idx]["timestamp_sec"])
            print(f"  🚨 No boundaries after 50% mark! Fell back to latest non-outro boundary at {ts/60:.2f}m.")

    # Apply the Q&A label to the winning boundary
    if chosen_qa_idx is not None:
        verified[chosen_qa_idx]["label"] = "qa_boundary"
        verified[chosen_qa_idx]["is_qa_segment"] = True
        verified[chosen_qa_idx]["llm_chapter_title"] = "Q&A"
        verified[chosen_qa_idx]["anchor_topic"] = "Q&A"

        # If the forward LLM loop also allowed a sub-question *inside* Q&A (after chosen_qa_idx)
        # that the LLM itself flagged as QA_SEGMENT, demote that sub-question to Micro so Q&A isn't split,
        # while keeping any genuine non-QA chapter (the 0.5% post-QA movie discussion edge case)!
        indices_to_drop = [
            idx for idx in llm_flagged_qa_indices
            if idx > chosen_qa_idx and idx != outro_idx
        ]
        if indices_to_drop:
            for d_idx in sorted(indices_to_drop, reverse=True):
                dropped_row = verified.pop(d_idx)
                print(f"  🧹 Consolidated sub-question boundary at {float(dropped_row['timestamp_sec'])/60:.2f}m into Micro Q&A scope.")
                rejected.append(dropped_row)

    return verified, rejected

# ===========================================================================
# MACRO PIPELINE: UNIFIED TRAJECTORY VERIFICATION
# ===========================================================================

class MacroCategory(str, Enum):
    MOVIE_DISCUSSION = "MOVIE_DISCUSSION"
    QA_SEGMENT = "QA_SEGMENT"
    OUTRO = "OUTRO"
    INTRO = "INTRO"
    GUEST_INTERVIEW = "GUEST_INTERVIEW"
    MOVIE_NEWS = "MOVIE_NEWS"
    INDUSTRY_TREND_OR_OTHER = "INDUSTRY_TREND_OR_OTHER"

class PartAssessment(BaseModel):
    part_index: int = Field(description="Part 1 to 5")
    active_subject: str = Field(description="Briefly state what the hosts are specifically discussing in this part.")

class MacroTrajectoryVerdict(BaseModel):
    reasoning: str = Field(
        description="STEP 1: Trajectory Analysis. Briefly acknowledge the Pre-Boundary context, but FOCUS on the 5 POST-BOUNDARY parts! What is the ACTUAL sustained topic happening AFTER the transition window?"
    )
    candidate_topic_name: str = Field(
        description="STEP 2: Look EXCLUSIVELY at the 5 Post-Boundary parts. Name the ACTUAL DOMINANT new topic occurring there."
    )
    candidate_category: MacroCategory = Field(
        description="STEP 3: Macro category of the new topic."
    )
    is_distinct_new_chapter: bool = Field(
        description="STEP 4: Compare `candidate_topic_name` STRICTLY against the declared `Pre-Boundary Anchor Topic` string—NEVER against a late transition bleed in the last minute of the Pre-Boundary text! "
        "True if `candidate_topic_name` is a different standalone single-title movie/TV/game review, sequel/prequel/remake (e.g. 'Cube 2: Hypercube' -> 'Cube Zero'), shifting INTO a Guest Interview from a movie review, or Q&A. "
        "CRITICAL GUEST INTERVIEW VETO: Must be FALSE if `Pre-Boundary Anchor Topic` is 'Guest Interview' (or if the pre-boundary text is already welcoming/introducing the guest) AND the post-boundary discussion is: "
        "(a) interviewing the guest about their career/channel, (b) discussing a movie on the guest's CV ('The Wild'), (c) discussing genre/industry topics ('animated horror'), OR (d) a rapid-fire 'lightning round' of movie takes, favorite musicals/shows, or general entertainment preferences!"
    )
    is_topic_originating_at_boundary: bool = Field(
        description="STEP 5: True if the sustained discussion of the NEW topic begins AT or AFTER the Transition Window. False if they were ALREADY having a sustained discussion about this exact topic > 1 minute before the boundary, OR if the pre-boundary text was already introducing/welcoming the same guest being interviewed after the boundary."
    )
    confidence_score_of_valid_boundary: int = Field(
        description="STEP 6: 0 to 100. How confident are you that a clean, distinct topic transition occurred here?"
    )

MACRO_TRAJECTORY_SYSTEM_PROMPT = """
You are an expert podcast chapter boundary verifier for the film podcast Sardonicast.
Evaluate potential chapter boundaries strictly using these CRITICAL RULES:

1. THE PRE-BOUNDARY BLEED SAFEGUARD: A transition hint or briefly joking about a movie early in the Pre-Boundary zone does NOT mean the topic started early. Set `is_topic_originating_at_boundary = False` ONLY if the actual SUSTAINED discussion of the NEW CANDIDATE TOPIC started > 1 minute before the boundary.
2. DOMINANT ACTUAL TOPIC: Check the 5 post-boundary parts fully. Name the ACTUAL dominant sustained topic.
3. HOST SIDE-QUESTS: If a new review is announced, but a host goes on a tangential rant to delay it, THAT RANT BELONGS TO THE NEW REVIEW.
4. THE TOPICAL UMBRELLA & POSTER TRACEBACK: If the hosts discuss 'the poster', 'the marketing', or 'the theater experience' WITHOUT explicitly naming a new title, TRACE IT BACK! This refers to the Anchor (`is_distinct_new_chapter = False`). HOWEVER, if they explicitly name a NEW movie (e.g., comparing the 'In Cold Blood' poster to the 'Capote' poster), this is a transition to the NEW movie! (`is_distinct_new_chapter = True`).
5. THE "EVERYTHING IS MOVIES" FALLACY: Check your internal database: 'AFRAID' and 'AfrAId' are the EXACT SAME MOVIE. Set `is_distinct_new_chapter = False`.
6. SEQUELS, REMAKES & COMPANION PIECES (ABSOLUTE): A prequel, sequel (e.g., 'Cube Zero', 'Hypercube'), remake, or companion biopic (e.g., 'Capote') is a COMPLETELY DISTINCT MOVIE. The moment the hosts begin setting up, introducing, or explaining why they watched the sequel/companion piece, THAT IS THE BOUNDARY. Do not wait for the formal 'deep dive' to start! You are STRICTLY FORBIDDEN from treating sequels/companions as a 'continuation' of the original movie. It is ALWAYS a VALID BOUNDARY (`is_distinct_new_chapter = True`).
7. TV SHOWS & VIDEO GAMES (ABSOLUTE): Shifting to a dedicated review of a specific TV Show (e.g., 'Baby Reindeer', 'The Curse') or Video Game (e.g., 'Halo') is ALWAYS a VALID BOUNDARY. Do NOT treat them as 'general media banter'. Set `is_distinct_new_chapter = True`.
8. GUEST INTERVIEWS ARE INDIVISIBLE UMBRELLAS (CRITICAL):
   - Welcoming/introducing a guest in the opening minutes and then asking them questions is ONE continuous 'Guest Interview'—NEVER split the guest greeting/setup from the interview itself (`is_distinct_new_chapter = False`, `is_topic_originating_at_boundary = False`).
   - If the PRE-BOUNDARY ANCHOR is 'Guest Interview', then industry/censorship conversations, genre/thematic discussions (e.g., 'animated horror movies'), asking the guest about movies THEY WORKED ON (e.g., 'The Wild'), AND rapid-fire 'lightning rounds' of recent movie takes, musical theater favorites, or entertainment preferences ARE ALL PART OF THE INTERVIEW (`is_distinct_new_chapter = False`).
   - EXCEPTIONS: You may ONLY exit a 'Guest Interview' (`is_distinct_new_chapter = True`) if they formally transition into a SUSTAINED, SINGLE-TITLE DEDICATED REVIEW of ONE specific assigned movie/TV show/game (never a multi-movie 'lightning round'!), or OSCARS ACADEMY AWARDS NOMINATIONS, or Q&A. CONVERSELY, shifting from a standalone movie review INTO a Guest Interview (even after a brief podcast intro reset) is ALWAYS a valid boundary (`is_distinct_new_chapter = True`).
9. STILL BANTERING & WATCHLISTS: If the Anchor is 'Intro: Mixed Banter', and they transition into a 'watchlist' discussion listing various titles they MIGHT watch, this is STILL BANTER! Set `is_distinct_new_chapter = False`. HOWEVER, if they actually settle on a single specific TV show or game to review deeply, that IS a new distinct chapter (True).

Read the trajectory, follow the steps in the schema exactly, and output your verdict.
""".strip()

def evaluate_macro_trajectory(
    anchor_topic: str,
    anchor_reasoning: str,
    announced_hint: str,
    pre_text: str,
    transition_window: str,
    trajectory_chunks: List[str],
    episode_date: str,
    cutoff_date: str,
    allow_qa: bool
) -> Dict[str, Any]:

    trajectory_str = f"[PRE-BOUNDARY ZONE]\n{pre_text}\n\n"
    trajectory_str += "[POST-BOUNDARY ZONE]\n"
    trajectory_str += f"--- TRANSITION WINDOW (15s before boundary) ---\n{transition_window}\n\n"
    for idx, chunk in enumerate(trajectory_chunks):
        trajectory_str += f"--- PART {idx+1} OF 5 ---\n{chunk}\n\n"

    qa_rule = (
        "SECOND-HALF Q&A SUPREMACY: We are in the SECOND HALF of the episode! If they pivot to a listener question, set `candidate_category = QA_SEGMENT`, set `is_distinct_new_chapter = True`, and assign high confidence!"
        if allow_qa
        else "FIRST-HALF Q&A BAN: Q&A NEVER starts in the first half. STRICTLY FORBIDDEN from setting `candidate_category = QA_SEGMENT`."
    )

    user_prompt = f"""
Episode Date: {episode_date} | New Release Cutoff: {cutoff_date}
- Pre-Boundary Anchor Topic: [{anchor_topic}]
- Pre-Boundary Anchor Reasoning (CONTEXT): [{anchor_reasoning}]
- Announced Transition Hint: [{announced_hint}]
- Q&A Rule (Rule 10): {qa_rule}

{trajectory_str}
""".strip()

    res = _call_structured_llm(user_prompt, MacroTrajectoryVerdict, system_prompt=MACRO_TRAJECTORY_SYSTEM_PROMPT)
    return res or {
        "candidate_topic_name": "Error",
        "is_distinct_new_chapter": False,
        "is_topic_originating_at_boundary": False,
        "confidence_score_of_valid_boundary": 0,
        "reasoning": "API Error"
    }

def check_macro_lead_in(text: str) -> str:
    class LeadIn(BaseModel):
        announced_topic: str = Field(description="The concrete name of the newly announced topic, or 'None'.")
    prompt = f"Did the hosts explicitly announce their next topic in this text?\n{text}"
    res = _call_structured_llm(prompt, LeadIn)
    return res.get("announced_topic", "None") if res else "None"

ANCHOR_SHORT_SYSTEM_PROMPT = """
Identify the PRIMARY overarching macro topic of this short podcast chapter (< 8 minutes) using these CRITICAL RULES:
1. GUEST INTERVIEW SUPREMACY (CRITICAL): If the hosts welcome, introduce, or begin interviewing a special guest during this chapter (especially if the guest setup/introduction spans Parts 3-5 or Parts 4-5), the Anchor MUST be 'Guest Interview'! Welcoming a guest in the opening minutes is already part of the Guest Interview—NEVER output 'Intro' or 'Intro: Mixed Banter' when a guest is being introduced/interviewed in the second half of the window. (However, if the guest is merely co-hosting a dedicated review of one specific movie, output that movie title).
2. THE 5-PART TALLY: Award 1 point to the dominant subject of each of the 5 parts. Normally, a 3-point majority wins. HOWEVER, see Rule 4 for early-episode exceptions!
3. IGNORE HOST SIDE-QUEST TANGENTS: If they are reviewing a specific title (e.g., Parts 1-3), but a host goes on a wild tangent (Parts 4-5), the Anchor is STILL the main title. Do not de-weight it!
4. INTRO DE-WEIGHTING & FINAL PIVOT (CRITICAL): The beginning of a podcast is chaotic. If the hosts initiate a hard pivot across BOTH Part 4 AND Part 5 to EITHER a substantive movie/game (e.g., 'Halo', 'The Super Mario Bros. Movie') OR welcoming/interviewing a guest ('Guest Interview'), this sustained 2-part pivot DICTATES the Anchor! You MUST de-weight the early intro banter (Parts 1-3), even if the intro banter had a 3-point majority. (NOTE: If Parts 1-4 are banter and a movie only appears in Part 5 alone, do NOT veto Parts 1-4—output 'Intro: Mixed Banter'!).
5. MIXED BANTER: If the text jumps between unrelated IPs, NO topic secures 3 points, and there is no 2-part Final Pivot (to a movie/game or 'Guest Interview') across Parts 4-5, output EXACTLY 'Intro: Mixed Banter'.
6. OPENING SKIT QUOTES: You are STRICTLY FORBIDDEN from assigning a fake title (e.g. 'Kentucky Fried Panda 4') as an Anchor. If the episode starts with a joke title, the Anchor is 'Intro: Mixed Banter'.
7. DEDUCE VAGUE TITLES: If they discuss a massive cultural release vaguely (e.g., 'the Nintendo movie'), deduce and output the exact title (e.g., 'The Super Mario Bros. Movie').
""".strip()

ANCHOR_LONG_SYSTEM_PROMPT = """
Identify the PRIMARY overarching macro topic of this long podcast chapter (> 8 minutes) using these CRITICAL RULES:
1. GUEST INTERVIEW VS CO-HOSTING: If the primary focus is welcoming and interviewing a guest about their life, career, credits, channel, or general movie/entertainment preferences (including rapid-fire 'lightning rounds' of movie opinions), the Anchor is 'Guest Interview'. HOWEVER, if the guest is participating in a dedicated, in-depth review of ONE specific assigned movie/TV show, the Anchor MUST be that specific title, NOT 'Guest Interview'.
2. THE "SNIPPET" SCALE RULE: You are only reading a few minutes from the START of this chapter, and the LAST 2 MINUTES before the boundary. Do not let a 1-2 minute side-tangent at the very end (such as asking a guest about one movie credit on their CV or proposing a quick lightning round) overwrite the main overarching topic.
3. ANCHOR HIERARCHY: Evaluate using STRICT PRIORITY: (1) MOST FREQUENT, (2) START, (3) RECENT.
4. IGNORE HOST SIDE-QUEST TANGENTS: If they are reviewing a specific title, but a host goes on a wild tangent to delay the review, the Anchor is STILL the main title being reviewed, NOT the tangent!
5. NEVER USE "FRANCHISE" OR "SERIES" AS AN ANCHOR: Even if the hosts say they are covering a whole franchise or trilogy (e.g., "all the Cubes"), you are STRICTLY FORBIDDEN from outputting 'Cube franchise' or 'Cube series' as the Anchor. You MUST output ONLY the single specific movie currently being reviewed (e.g., 'Cube', 'Cube 2: Hypercube', 'Cube Zero').
6. CONTINUITY: If the text continues the currently tracked topic provided in the prompt, output that EXACT string—UNLESS the currently tracked topic is the initial placeholder 'Intro', in which case you MUST output the actual Movie Title or 'Guest Interview' instead of 'Intro'.
7. DEDUCE VAGUE TITLES: If they discuss a massive cultural release vaguely (e.g., 'the Nintendo movie' in 2023), use your knowledge to deduce and output the exact title (e.g., 'The Super Mario Bros. Movie').
""".strip()

def get_macro_anchor_topic(
    text: str, current_chapter_title: str, is_past_intro_window: bool, is_short_chapter: bool
) -> Tuple[str, str]:
    if is_short_chapter:
        class TopicTally(BaseModel):
            topic_name: str = Field(description="The name of the topic/concept (use 'Guest Interview' if introducing/interviewing a guest).")
            points_awarded: int = Field(description="Number of parts this topic dominated.")

        class AnchorShort(BaseModel):
            part_by_part_breakdown: List[str] = Field(description="STEP 1: Briefly state the dominant subject of each of the 5 parts.")
            tallies: List[TopicTally] = Field(description="STEP 2: Award 1 point to the dominant subject of each of the 5 parts.")
            mixed_banter_check: str = Field(description="STEP 3: Do the parts establish a 'Guest Interview' or a single movie/game (either by 3-part majority or a Parts 4-5 Final Pivot), or are they just unrelated intro banter?")
            step_by_step_analysis: str = Field(description="STEP 4: Final explanation, explicitly noting if Parts 1-3 were de-weighted because Parts 4-5 pivoted to a 'Guest Interview' or a specific movie/game.")
            winning_topic: str = Field(description="STEP 5: The concrete Anchor. Output 'Guest Interview' if a guest is introduced/interviewed across Parts 4-5 (or majority). Output the movie/game title if a single title wins or pivots across Parts 4-5. Otherwise output 'Intro: Mixed Banter'.")

        sys_prompt = ANCHOR_SHORT_SYSTEM_PROMPT
        schema_to_use = AnchorShort
    else:
        class AnchorLong(BaseModel):
            step_by_step_analysis: str = Field(description="STEP 1: Analyze the text using the Anchor Hierarchy (Frequent > Start > Recent). Consider the 'Snippet' Scale Rule.")
            winning_topic: str = Field(description="STEP 2: The concrete Anchor. Name the specific film title, 'Guest Interview', or topic.")

        sys_prompt = ANCHOR_LONG_SYSTEM_PROMPT
        schema_to_use = AnchorLong

    user_prompt = f"Currently Tracked Topic: '{current_chapter_title}'\n\n[TEXT]\n{text}"
    res = _call_structured_llm(user_prompt, schema_to_use, system_prompt=sys_prompt)
    if res:
        topic = res.get("winning_topic", current_chapter_title)
        if topic.strip().lower() == "intro":
            topic = "Intro: Banter"
        return topic, res.get("step_by_step_analysis", "No reasoning provided.")
    return current_chapter_title, "API Error"

# ===========================================================================
# MICRO PIPELINE: 3-PRONG SAFETY VALVE (WITH TEMPORAL CHUNKING)
# ===========================================================================

class MicroQuestionAnchor(BaseModel):
    question_topic: str = Field(description="Concise 2-6 word core subject of the listener's question.")

def get_micro_question_anchor(text: str) -> str:
    prompt = f"Read the start of this Sardonicast Q&A segment. Extract the core subject of the listener's question.\n\n[Q&A START SNIPPET]\n{text}"
    res = _call_structured_llm(prompt, MicroQuestionAnchor)
    return res.get("question_topic", "First Question") if res else "First Question"

class MicroBoundaryVerification(BaseModel):
    has_handoff_phrase_or_username: bool = Field(description="True if a handoff phrase or listener username/handle is spoken.")
    is_explicit_question_read: bool = Field(description="True if a NEW, fully-read question is presented.")
    is_topic_completely_unrelated: bool = Field(description="True if the new question premise/angle is distinct from the active question.")
    window_question_subject: str = Field(description="Brief 3-6 word summary of the question.")
    reasoning: str = Field(description="Explanation of the 3 prongs.")

MICRO_VERIFICATION_SYSTEM_PROMPT = """
Verify a candidate timestamp inside a Sardonicast Q&A segment.
The text is temporally segmented into [PRE-BOUNDARY], [POST-BOUNDARY: FIRST 30s], and [POST-BOUNDARY: SECOND 30s].

WEIGHTING RULES:
- DE-WEIGHT THE TRANSITION: [PRE-BOUNDARY] and [POST-BOUNDARY: FIRST 30s] often contain lingering discussion of the previous question. Give these lower weight.
- Look heavily at the [POST-BOUNDARY: SECOND 30s] to determine what the conversation settles into.

RULES FOR THE 3-PRONG CHECK:
1. THE HANDOFF (Syntax): Look for structural triggers: "[x] has one for us", "[x] asks", "this one from [x]", or "is this to say".
   - THE "FROM [WEIRD PHRASE]" TRIGGER (CRITICAL): If the host says "from [X]", and [X] is a bizarre, contextually out-of-place phrase (e.g., 'from herbal Coast', 'from an actual crocodile'), it is a username! Do NOT trigger this for normal conversational usage like 'from the movie' or 'from my perspective'. Flag `has_handoff_phrase_or_username = True`.
   - SENTENCE-AS-USERNAME RULE: Listener handles are frequently absurd noun phrases. If a bizarre phrase is immediately followed by 'say this', 'says', 'asks', or 'is this to say', it IS a username!
2. THE PROMPT (Syntax): If the host reads a NEW prompt/question directed at them (e.g., 'thoughts on...'), flag `is_explicit_question_read = True`.
   - LATE HANDOFF RULE: If the handoff/username occurs at the very END of the snippet and the actual question is NOT fully read in the text, you MUST set `is_explicit_question_read = False`.
3. TOPIC DISCONNECT (Semantics): Evaluate the heavily-weighted later portions. If the question shifts to a new premise not present in the active question, flag `is_topic_completely_unrelated = True`.
   - THE "Q&A UMBRELLA" FALLACY (CRITICAL): You are STRICTLY FORBIDDEN from linking two texts just because they are both 'listener questions', 'podcast recommendations', or part of the 'Q&A segment'. You MUST compare the specific subject matter!
   - THE "EVERYTHING IS MOVIES" FALLACY: You are STRICTLY FORBIDDEN from using the words 'film', 'movie', or 'cinema' to link topics.
   - LOGISTICS VS PHILOSOPHY: If the active question is about event planning/logistics (e.g., 'BFI film festival plans') and the new question is philosophical/analytical (e.g., 'art criticism'), they are COMPLETELY UNRELATED premises. Flag `is_topic_completely_unrelated = True`.
   - SAME-FRANCHISE Q&A RULE (CRITICAL): Shifting from 'UK accents in The Hobbit film' to 'reading the LOTR books' is a COMPLETELY DIFFERENT question premise. Flag `is_topic_completely_unrelated = True`!
   - INSIDE JOKES & SLANG: If a new phrase (e.g. 'baby brain syndrome') is just a slang term or podcast inside joke referring to the exact same concept as the active question (e.g. 'child plot armor'), it is NOT a disconnect.
4. HOST PARAPHRASING IS NOT A BOUNDARY: If the host simply restates the active question, this is a CONTINUATION.
""".strip()

def verify_micro_boundary_window(
    inspection_text: str, current_topic: str, episode_title: str
) -> Dict[str, Any]:
    user_prompt = f"""
Episode: '{episode_title}'
CRUCIAL: The CURRENTLY ACTIVE QUESTION being answered before this boundary is: [{current_topic}].

{inspection_text}
""".strip()
    res = _call_structured_llm(user_prompt, MicroBoundaryVerification, system_prompt=MICRO_VERIFICATION_SYSTEM_PROMPT)
    return res or {
        "has_handoff_phrase_or_username": False,
        "is_explicit_question_read": False,
        "is_topic_completely_unrelated": False,
        "window_question_subject": "Unknown Question",
        "reasoning": "API Error"
    }

# ===========================================================================
# PASS 2: SUMMARIZATION (ZERO IMPACT ON PASS 1 VALIDATION)
# ===========================================================================

# ===========================================================================
# PASS 2: SUMMARIZATION (COMPLETE & SELF-CONTAINED)
# ===========================================================================

MAX_INTRO_DURATION_SEC = 480.0  # 8 minutes max for an Intro chapter

# Blacklisted GLOBALLY across all segments (unless featured in episode_title as a guest)
GLOBAL_HOST_BLACKLIST = {
    "adam": ["adam"], "adamjohnston": ["adam"],
    "yms": ["yms", "yourmoviesucks", "your movie sucks"],
    "yourmoviesucks": ["yms", "yourmoviesucks", "your movie sucks"],
    "alex": ["alex"], "ihe": ["ihe", "i hate everything", "ihateeverything"],
    "ihateeverything": ["ihe", "i hate everything", "ihateeverything"],
    "ralph": ["ralph"], "ralphsepe": ["ralph"],
    "ralphthemoviemaker": ["ralph", "ralphthemoviemaker"],
    "jake": ["jake", "mancarryingthing", "man carrying thing", "womancarryingman"],
    "mancarryingthing": ["jake", "mancarryingthing", "man carrying thing"],
    "womancarryingman": ["jake", "womancarryingman", "woman carrying man"],
}

# Substrings blacklisted ONLY in the Outro (catches "Sardonicast premium", "patreon.com/...", "merch", etc.)
OUTRO_META_SUBSTRINGS = (
    "sardonicast", "patreon", "merch", "premium", "highlight",
    "reddit", "subreddit", "youtube", "letterboxd", "itunes", "spotify"
)

SMALL_TITLE_WORDS = {
    "a", "an", "the", "and", "but", "or", "for", "nor",
    "on", "at", "to", "from", "by", "in", "of", "vs", "vs."
}

def smart_title_case(title: str) -> str:
    """Polishes lowercase strings ('Boogie2988 documentary and drama channels') into proper Title Case."""
    words = title.split()
    out = []
    for idx, w in enumerate(words):
        if any(c.isupper() for c in w[1:]) or w.startswith("("):
            out.append(w)
        elif 0 < idx < len(words) - 1 and w.lower() in SMALL_TITLE_WORDS and not out[-1].endswith(":"):
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)

def extract_timestamped_window(snippets: List[Dict[str, Any]], start_sec: float, end_sec: float) -> str:
    """Includes [MM:SS] markers so the Pass 2 LLM can distinguish a 5-second joke from a 4-minute topic."""
    return " ".join([
        f"[{int(s['start_seconds'] // 60):02d}:{int(s['start_seconds'] % 60):02d}] {s['text'].strip()}"
        for s in snippets if start_sec <= s["start_seconds"] < end_sec
    ])

def extract_pass2_summary_text(
    snippets: List[Dict[str, Any]],
    start_sec: float,
    end_sec: float,
    use_strided: bool = USE_STRIDED_PASS2_SAMPLING,
    is_outro: bool = False
) -> str:
    raw_duration = end_sec - start_sec
    # 1. Unless intro < 2 mins (120s), skip the first 25s on 0:00 chapters (cuts 0:13 gag, keeps 0:49 guest intro)
    effective_start = 25.0 if (start_sec == 0.0 and raw_duration >= 120.0) else start_sec

    # 2. Trim the final 15-45s transition handoff on non-outro chapters >= 2 mins (120s)
    if not is_outro and raw_duration >= 120.0:
        end_trim = min(20.0, raw_duration * 0.15) if raw_duration <= STRIDED_THRESHOLD_SEC else min(45.0, raw_duration * 0.08)
        effective_end = max(effective_start + 20.0, end_sec - end_trim)
    else:
        effective_end = end_sec

    duration = max(0.0, effective_end - effective_start)

    if not use_strided or duration <= STRIDED_THRESHOLD_SEC:
        return extract_timestamped_window(snippets, effective_start, effective_end)

    num_windows, win_len = 6, 60.0
    zone_start = effective_start if start_sec == 0.0 else (effective_start + min(15.0, duration * 0.03))
    zone_end = max(zone_start + win_len, effective_end)
    usable_span = max(win_len, zone_end - zone_start)

    blocks = []
    for idx in range(num_windows):
        frac = idx / float(num_windows - 1)
        w_start = zone_start + frac * max(0.0, usable_span - win_len)
        sample_txt = extract_timestamped_window(snippets, w_start, min(effective_end, w_start + win_len))
        if sample_txt.strip():
            blocks.append(f"[EXCERPT {idx + 1}/{num_windows}]\n{sample_txt}")
    return "\n\n".join(blocks)

def clean_segment_tags(tags: List[str], seg_type: str, episode_title: str) -> List[str]:
    """Context-aware tag filter: Global host ban + Outro boilerplate substring ban."""
    ep_title_lower = (episode_title or "").lower()
    cleaned, seen = [], set()

    for t in tags:
        t_clean = t.strip()
        norm = normalize_text(t_clean)
        if not norm or norm in seen:
            continue

        if norm in GLOBAL_HOST_BLACKLIST:
            guest_markers = GLOBAL_HOST_BLACKLIST[norm]
            if not any(re.search(rf"\b{re.escape(m)}\b", ep_title_lower) for m in guest_markers):
                continue

        if seg_type == "outro" and (norm in ("qa", "qanda") or any(sub in norm for sub in OUTRO_META_SUBSTRINGS)):
            continue

        seen.add(norm)
        cleaned.append(t_clean)
    return cleaned

def clean_film_chapter_title(title: str) -> str:
    """Strips lazy words ('Review', 'Discussion', 'Introduction', 'Episode Setup', trailing 'Q&A') without cutting 'vs X'."""
    t = re.sub(r"(?i)^(Intro|Q&A|Podcast Intro|Audience Questions|Outro\s*\+?\s*Recommendation)\s*[:-]?\s*", "", title.strip()).strip()
    t = re.sub(
        r"(?<!\bvs)(?<!\bvs\.)(?<!\bversus)(?<!\band)(?<!\bor)(?<!\bof)(?<!\bfor)(?<!\bto)\s*[-:]?\s*\b(Introduction|Intro|Review|Discussion|Analysis|Breakdown|Overview|Spoiler\s*Discussion|Spoilers|Q\s*&\s*A)\b$",
        "", t, flags=re.IGNORECASE
    ).strip()
    t = re.sub(r"(?i)\s*(?:and|&|,)\s*(?:Episode\s+)?(?:Setup|Housekeeping|Intro|Introduction|Preview|Plans)\b$", "", t).strip()
    return t

def clean_outro_rec_title(title: str, work_title: str = "", description: str = "") -> str:
    """Extracts a clean recommended movie title, falling back to work_title or description if title is 'Outro'."""
    t = clean_film_chapter_title(title)
    t = re.sub(r"\b(Marathon|Next\s+Episode|Episode|Recommendation|Recommendations|Pick|Picks|Outro)\b", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s{2,}", " ", t).strip(" -:,")
    if ":" in t and not re.search(r"(?i)\b(Cube\s*2|Part|Vol|Episode|Chapter)\b", t):
        t = t.split(":")[0].strip()

    if not t and work_title and work_title.lower() != "outro":
        t = clean_film_chapter_title(work_title)

    if not t and description:
        m = re.search(r"recommend(?:ing|s|ed)?\s+([A-Z][\w\s':,-]+?(?:\(\d{4}\))?)(?:\s+by\b|\s+and\b|\s+as\b|[.,])", description)
        if m:
            t = m.group(1).strip()

    return t or "Next Episode Pick"

def enforce_max_two_points_in_title(title: str) -> str:
    """Collapses 3-item lists ('Title: A, B, and C') down to 2 points ('Title: A and B') on non-Q&A titles."""
    if ":" not in title:
        return title
    prefix, subtitle = title.split(":", 1)
    m = re.match(r"^\s*([^,]+),\s*([^,]+?)(?:,)?\s+(?:and|&)\s+(.+)$", subtitle.strip(), flags=re.IGNORECASE)
    if m:
        return f"{prefix.strip()}: {m.group(1).strip()} and {m.group(2).strip()}"
    return title

class SegmentSummary(BaseModel):
    description: str = Field(
        description="STEP 1 (WRITE FIRST): A 1-2 sentence summary of the substantive points, critique, or recommendation. NEVER describe the segment as 'introducing' or 'kicking off' a topic, and do NOT claim they 'mostly compare' two movies if the second movie is only mentioned in passing. Mention at most 2-3 illustrative titles in prose. "
        "OUTRO DESCRIPTION RULE: For outro segments, describe ONLY the recommended film, why they chose it / their expectations, and any unique closing bit or tangent (e.g. a joke about Black Christmas). NEVER mention routine every-episode sign-off boilerplate such as encouraging listeners to watch it before the next episode, plugging Patreon/merch/highlights/channels, or saying goodbye!"
    )
    segment_kind: str = Field(
        description="STEP 2: Classify based on your description: 'intro_banter' (ONLY if < 8 min opening banter), 'guest_interview', 'film_or_show_discussion' (includes movies, TV shows, games, AND documentaries like the Boogie2988 documentary), 'news_or_thematic_topic', 'qa_question' (for audience Q&A questions), or 'outro_recommendation'."
    )
    is_guest_segment: bool = Field(
        description="STEP 3: True if this chapter is primarily an interview, catch-up, or pre-review banter centered around a special guest. False if it is a dedicated movie/documentary review or Q&A."
    )
    primary_work_title: str = Field(
        description="STEP 4: If 'film_or_show_discussion' OR 'outro_recommendation', output the exact name of the primary movie/show/documentary being reviewed or recommended (e.g. 'Boogie2988 Documentary', 'Misericordia (2024)', 'Dune: Part Two'). MUST be empty '' if `segment_kind == 'qa_question'`! NEVER output 'Outro'!"
    )
    chapter_title: str = Field(
        description="STEP 5 (Use Proper Title Case; keep it natural, complete, and conversational): "
        "- NO GENERIC 'CHAOS': NEVER use the standalone noun 'Chaos' in a title (e.g. NO 'Camp and Chaos' or 'Blind Watch Chaos')! You may only use the adjective 'Chaotic' when it directly modifies a specific craft element (e.g. 'Chaotic Cinematography', 'Chaotic Script'). "
        "- If 'qa_question': Summarize the listener's overarching QUESTION PREMISE in 3-6 words (e.g. 'Undersold Past Recommendations', 'Favorite Forever Songs', 'Art Criticism vs Recommendation'). NEVER reduce a Q&A title to a single movie title (like 'Her' or 'Megalopolis') mentioned as an example or closing joke! "
        "- If 'film_or_show_discussion' (including documentaries!): Center on `primary_work_title` ONLY (never 'Film A vs. Film B' unless both are reviewed in full). Never append 'Review' or 'Discussion'. If the title is short (1-2 words) or a documentary, add EITHER the release year/remake label OR a natural subtitle with AT MOST 1 or 2 concrete points (e.g. 'Dune (1984): Campy Acting and Grotesque Effects', 'To Freedom: An Absurd Blind Watch'). If the film title is already 3+ words, use the full title or short title + 1 simple phrase. "
        "- If 'news_or_thematic_topic': State the specific news angle or director drama being discussed (e.g. 'Coppola\\'s Megalopolis Rollout & Eddington Trailer'). "
        "- If 'guest_interview' or `is_guest_segment == True`: Prefix with 'Guest: ' and include the guest's main handle (e.g. 'Guest: Kayla Says on Kayla Reads & Film Industry'). "
        "- If 'outro_recommendation': Output ONLY the recommended movie title from `primary_work_title` (plus year if known) or '[Title] Franchise'. NEVER output 'Outro'! "
        "- If 'intro_banter': Write a specific 2-5 word subject on what they joke/talk about (NEVER include 'Banter', 'Episode Setup', 'Housekeeping', or 'Spoilers')."
    )
    tags: List[str] = Field(
        default_factory=list,
        description="STEP 6: Specific movies, directors, guests (always include the guest's main channel/handle if present), and concepts discussed."
    )

def summarize_segment_with_llm(
    text: str, default_title: str, segment_type: str, anchor_topic: str, episode_title: str, episode_date: str = ""
) -> Dict[str, Any]:
    if segment_type == "outro":
        clean_hint = (
            "THIS IS THE EPISODE OUTRO & RECOMMENDATION. Identify the exact movie/media recommended for the next episode. "
            "In `description`, state ONLY what was recommended, why they chose it (or how they found it), and any specific joke/tangent—STRICTLY OMIT standard podcast sign-offs like encouraging listeners to watch it before the follow-up, plugging Patreon/merch/highlights, or saying goodbye."
        )
    elif segment_type == "qa_micro":
        clean_hint = (
            f"THIS IS AN AUDIENCE Q&A SEGMENT (Initial Question Subject: '{anchor_topic}'). "
            "Your `chapter_title` MUST capture the listener's question premise/theme (e.g. 'Undersold Past Recommendations', 'Favorite Forever Songs'). "
            "You are STRICTLY FORBIDDEN from naming a Q&A segment after a single movie title (e.g. 'Her', 'Megalopolis') that the hosts merely bring up as an example or end-of-answer joke!"
        )
    elif anchor_topic.strip().lower().startswith(("intro", "outro", "chapter")) and segment_type != "intro":
        clean_hint = "Determine the main film, documentary, guest interview, or topic directly from the text."
    else:
        clean_hint = f"Overarching topic hint: '{anchor_topic}'."

    rules = f"""
- CONTEXT: Full Episode Video Title = '{episode_title}' | Episode Upload Date = '{episode_date}'.
- {clean_hint}
- MULTI-TOPIC VIDEO TITLE WARNING: The Video Title ('{episode_title}') lists ALL movies across the entire 2-hour episode! Focus strictly on the ONE topic dominating the excerpts below.
- USE THE [MM:SS] TIMESTAMPS TO IGNORE BRIEF JOKES & HANDOFFS: If a movie or quote is only mentioned for a few seconds as an opening gag or a closing transition handoff/joke (like 'Kung Fu Panda 4' at the end of an intro or 'Megalopolis' at the end of a Q&A answer), exclude it from `chapter_title`!
- PROPER TITLE CASE & BANNED FILLER WORDS: Always use Title Case on `chapter_title`. Never use lazy filler words like 'Chaos' (as a standalone noun), 'Episode Setup', 'Housekeeping', 'Spoilers', or 'Review'.
- OUTRO RECOMMENDATIONS: When `segment_type == 'outro'`, both `primary_work_title` and `chapter_title` MUST be the exact title of the movie recommended for next episode (e.g., 'Misericordia (2024)'). NEVER output the word 'Outro'!
""".strip()

    res = _call_structured_llm(f"Summarize this Sardonicast segment.\n{rules}\n\n[TEXT]\n{text}", SegmentSummary)
    return res or {
        "description": "Summary unavailable.",
        "segment_kind": segment_type,
        "is_guest_segment": False,
        "primary_work_title": "",
        "chapter_title": default_title,
        "tags": []
    }

class MacroQASynthesis(BaseModel):
    synthesized_qa_title: str = Field(
        description="A natural, complete 4-8 word title summarizing the most interesting Q&A topics from the list (e.g. 'Sardonicast Jingle, Yellow Paint & Childhood Cringe'). NEVER cut off or truncate words, and do NOT include the 'Q&A:' prefix."
    )

def synthesize_macro_qa_title(micro_topics: List[str]) -> str:
    """Uses the LLM to cleanly synthesize Micro Q&A topics into a Macro Q&A title without word truncation."""
    if not micro_topics:
        return "Q&A"
    prompt = f"Synthesize these listener question topics into a single punchy 4-8 word Q&A header (no 'Q&A:' prefix, no cut-off words):\n{micro_topics}"
    res = _call_structured_llm(prompt, MacroQASynthesis)
    title = res.get("synthesized_qa_title", "").strip() if res else ""
    title = re.sub(r"(?i)^Q&A\s*[:-]?\s*", "", title).strip()
    return f"Q&A: {title}" if title else "Q&A"

def finalize_pass2_summaries(
    verified_chapters: List[Dict[str, Any]],
    full_snippets: List[Dict[str, Any]],
    max_time: float,
    episode_title: str,
    episode_date: str = "",
    mode: str = "macro"
) -> List[Dict[str, Any]]:
    use_strided = USE_STRIDED_PASS2_SAMPLING if mode == "macro" else False
    for i in range(len(verified_chapters)):
        start_sec = float(verified_chapters[i]["timestamp_sec"])
        end_sec = float(verified_chapters[i + 1]["timestamp_sec"]) if (i + 1 < len(verified_chapters)) else max_time
        verified_chapters[i]["end_timestamp_sec"] = end_sec
        duration = end_sec - start_sec

        current_label = verified_chapters[i].get("label", "chapter")
        prelim_title = verified_chapters[i].get("llm_chapter_title", "Chapter")

        if mode == "macro" and start_sec == 0.0 and current_label == "intro":
            seg_type = "intro" if duration <= MAX_INTRO_DURATION_SEC else "standard"
        elif current_label == "outro_boundary":
            seg_type = "outro"
        elif mode == "micro":
            seg_type = "qa_micro"
        elif bool(verified_chapters[i].get("is_qa_segment")) or current_label == "qa_boundary":
            seg_type = "qa_macro"
        else:
            seg_type = "standard"

        if seg_type == "qa_macro":
            verified_chapters[i]["llm_chapter_title"] = "Q&A"
            verified_chapters[i]["llm_description"] = "The hosts answer questions from the community."
            verified_chapters[i]["llm_tags"] = []
            verified_chapters[i]["is_qa_segment"] = True
            continue

        is_outro = (seg_type == "outro")
        text_end_sec = max(start_sec, min(end_sec, max_time - 30.0) if is_outro else end_sec)
        sampled_text = extract_pass2_summary_text(
            full_snippets, start_sec, text_end_sec, use_strided=use_strided, is_outro=is_outro
        )
        summary = summarize_segment_with_llm(
            sampled_text, prelim_title, seg_type, prelim_title, episode_title, episode_date
        )

        kind = summary.get("segment_kind", "")
        is_guest_seg = bool(summary.get("is_guest_segment", False))
        desc_text = summary.get("description", "")
        work_title = clean_film_chapter_title(summary.get("primary_work_title", ""))
        clean_title = clean_film_chapter_title(summary.get("chapter_title", prelim_title))

        # GUARDRAIL 1: Prevent a movie review with guests on the panel (e.g. "Cube" at 14:30) from being mislabeled as "Guest: ..."
        prelim_lower = prelim_title.lower()
        if (kind == "guest_interview" or is_guest_seg) and start_sec > 0.0 and not any(k in prelim_lower for k in ("guest", "interview", "intro")):
            kind = "film_or_show_discussion"
            is_guest_seg = False
            if not work_title:
                work_title = clean_film_chapter_title(prelim_title.split("/")[0].strip())
            if clean_title.lower().startswith("guest"):
                clean_title = work_title

        # GUARDRAIL 2: Prepend short movie/documentary root ONLY on non-Q&A discussions if omitted
        if seg_type != "qa_micro" and kind == "film_or_show_discussion" and work_title:
            short_root = re.split(r":|\bor\s*\(|\(", work_title, maxsplit=1)[0].strip()
            short_norm = normalize_text(short_root)
            if short_norm and short_norm not in normalize_text(clean_title):
                clean_title = f"{short_root}: {clean_title}"

        # GUARDRAIL 3: Enforce max 2 points after colon on non-Q&A titles
        if seg_type not in ("qa_macro", "qa_micro"):
            clean_title = enforce_max_two_points_in_title(clean_title)

        # GUARDRAIL 4: Enforce Proper Title Case
        clean_title = smart_title_case(clean_title)

        if seg_type == "intro" and duration <= MAX_INTRO_DURATION_SEC and kind == "intro_banter" and not is_guest_seg:
            clean_title = re.sub(r"(?i)\bBanter\b", "", clean_title).strip()
            raw_title = f"Intro: {clean_title}" if clean_title else "Intro"
        elif seg_type == "outro":
            clean_rec = smart_title_case(clean_outro_rec_title(clean_title, work_title=work_title, description=desc_text))
            raw_title = f"Outro + Recommendation: {clean_rec}"
        elif seg_type == "qa_micro":
            raw_title = f"Q&A: {clean_title}"
        elif (kind == "guest_interview" or is_guest_seg or "guest" in prelim_lower) and not clean_title.lower().startswith("guest"):
            raw_title = f"Guest: {clean_title}"
        else:
            raw_title = clean_title

        verified_chapters[i]["llm_chapter_title"] = raw_title
        verified_chapters[i]["llm_description"] = desc_text
        verified_chapters[i]["llm_tags"] = clean_segment_tags(summary.get("tags", []), seg_type, episode_title)
        verified_chapters[i]["is_qa_segment"] = (seg_type == "qa_micro")

    return verified_chapters

# ===========================================================================
# EXECUTION LOOPS
# ===========================================================================

def verify_macro_candidates(
    candidates_df: pd.DataFrame, full_snippets: List[Dict[str, Any]], episode_title: str, episode_date: str, max_time: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n=== Running Unified Trajectory MACRO Verification ({LLM_MODEL}) ===")

    verified, rejected = [], []
    verified.append({
        "timestamp_sec": 0.0,
        "label": "intro",
        "llm_chapter_title": "Intro",
        "is_qa_segment": False,
        "anchor_topic": "Intro"
    })
    cutoff_date = get_cutoff_date(episode_date)

    for i in range(len(candidates_df)):
        row = candidates_df.iloc[i]
        curr_cand = float(row["timestamp_sec"])
        last_verified_time = verified[-1]["timestamp_sec"]

        if row.get("label") == "outro_boundary":
            verified.append(row.to_dict() | {
                "label": "outro_boundary",
                "llm_chapter_title": "Outro",
                "is_qa_segment": False,
                "anchor_topic": "Outro"
            })
            continue

        print(f"\nEvaluating Macro Candidate at {curr_cand / 60:.2f} min:")

        if (curr_cand - last_verified_time) < 100.0 and last_verified_time > 0.0:
            print("  🚫 REJECTED (< 100 sec since last boundary)")
            rejected.append(row.to_dict())
            continue

        current_chapter = verified[-1].get("anchor_topic", "Intro")
        is_first_boundary = (last_verified_time == 0.0)
        anchor_end_offset = 10.0 if is_first_boundary else 30.0
        anchor_end_time = curr_cand - anchor_end_offset

        chapter_duration = anchor_end_time - last_verified_time
        is_short_chapter = (chapter_duration <= 480.0)

        if anchor_end_time <= last_verified_time:
            pre_text = "[PRE-BOUNDARY ZONE: Too short to extract]"
        else:
            if is_short_chapter:
                chunk_duration = chapter_duration / 5.0
                pre_text_blocks = []
                for p in range(5):
                    chunk_start = last_verified_time + (p * chunk_duration)
                    chunk_end = chunk_start + chunk_duration
                    chunk_txt = extract_text_window(full_snippets, chunk_start, chunk_end)
                    if not chunk_txt.strip():
                        chunk_txt = "(Silence / No dialogue)"
                    pre_text_blocks.append(f"[PART {p+1} OF 5]\n{chunk_txt}")
                pre_text = "\n\n".join(pre_text_blocks)
            else:
                start_window_end = min(last_verified_time + 180.0, anchor_end_time)
                start_blocks = []
                for m in range(int((start_window_end - last_verified_time) / 60.0) + 1):
                    s = last_verified_time + (m * 60.0)
                    e = min(s + 60.0, start_window_end)
                    block = extract_text_window(full_snippets, s, e)
                    if block.strip():
                        start_blocks.append(f"[MINUTE {m+1}]\n{block}")
                start_window = "\n\n".join(start_blocks)

                end_window_start = max(start_window_end, anchor_end_time - 120.0)
                end_window = (
                    "(Too short to extract)"
                    if anchor_end_time <= end_window_start
                    else extract_text_window(full_snippets, end_window_start, anchor_end_time)
                )
                pre_text = f"[START OF CURRENT CHAPTER]\n{start_window}\n\n[LAST 2 MINUTES BEFORE BOUNDARY]\n{end_window}"

        is_past_intro = bool(curr_cand > 360.0)
        anchor_topic, anchor_reasoning = get_macro_anchor_topic(pre_text, current_chapter, is_past_intro, is_short_chapter)

        if anchor_topic == "Unknown":
            anchor_topic = "Movie Discussion"

        lead_in_text = extract_text_window(full_snippets, max(last_verified_time, curr_cand - 25.0), curr_cand + 20.0)
        announced_hint = check_macro_lead_in(lead_in_text)

        t_next = float(candidates_df.iloc[i + 1]["timestamp_sec"]) if i + 1 < len(candidates_df) else max_time
        forward_duration = t_next - curr_cand
        is_short_forward = (forward_duration <= 480.0)

        trajectory_chunks = []
        transition_window = extract_text_window(full_snippets, max(last_verified_time, curr_cand - 15.0), curr_cand)

        if is_short_forward:
            chunk_duration = forward_duration / 5.0
            for p in range(5):
                chunk_start = curr_cand + (p * chunk_duration)
                chunk_end = chunk_start + chunk_duration
                chunk_txt = extract_text_window(full_snippets, chunk_start, chunk_end)
                trajectory_chunks.append(chunk_txt if chunk_txt.strip() else "(Silence / No dialogue)")
        else:
            for m in range(5):
                chunk_start = curr_cand + (m * 60.0)
                chunk_end = chunk_start + 60.0
                chunk_txt = extract_text_window(full_snippets, chunk_start, chunk_end)
                trajectory_chunks.append(chunk_txt if chunk_txt.strip() else "(Silence / No dialogue)")

        is_second_half = bool(curr_cand >= (max_time / 2.0))

        verdict = evaluate_macro_trajectory(
            anchor_topic, anchor_reasoning, announced_hint, pre_text, transition_window, trajectory_chunks,
            episode_date, cutoff_date, allow_qa=is_second_half
        )

        candidate_topic = verdict.get("candidate_topic_name", "None")
        is_qa = bool(is_second_half and verdict.get("candidate_category") == "QA_SEGMENT")
        is_distinct = verdict.get("is_distinct_new_chapter", False)
        originated_at_boundary = verdict.get("is_topic_originating_at_boundary", False)
        confidence = verdict.get("confidence_score_of_valid_boundary", 0)

        print(f"  ├─ Anchor Reasoning: {anchor_reasoning}")
        print(f"  ├─ Trajectory Analysis: {verdict.get('reasoning', 'No reasoning provided.')}")
        print(f"  ├─ Anchor: '{anchor_topic}' | Candidate: '{candidate_topic}'")
        print(f"  ├─ Boundary Confidence: {confidence}% | Distinct: {is_distinct} | Originated at Boundary: {originated_at_boundary}")

        # Allow a >=90% confidence boundary with `originated_at_boundary=True` to pass if the LLM only
        # flipped `is_distinct=False` due to <1m pre-boundary bleed (unless inside a Guest Interview umbrella)
        is_guest_anchor = ("guest" in anchor_topic.lower() or "interview" in anchor_topic.lower())
        high_conf_bleed_override = (
            confidence >= 90
            and originated_at_boundary
            and not is_guest_anchor
            and normalize_text(candidate_topic) != normalize_text(anchor_topic)
        )

        if not is_qa and not is_distinct and not high_conf_bleed_override:
            print("  🚫 REJECTED (LLM flagged candidate as not distinct from Anchor)")
            rejected.append(row.to_dict())
            continue

        if not is_qa and not originated_at_boundary:
            print("  🚫 REJECTED (LLM flagged candidate topic as starting significantly BEFORE the boundary)")
            rejected.append(row.to_dict())
            continue

        if not is_qa and (
            normalize_text(candidate_topic) == normalize_text(anchor_topic)
            or normalize_text(candidate_topic) == normalize_text(current_chapter)
        ):
            print("  🚫 REJECTED BY PYTHON (Candidate Topic exactly identical to Anchor Topic)")
            rejected.append(row.to_dict())
            continue

        if confidence >= 75:
            print(f"  ✅ CONFIRMED: {candidate_topic}")
            verified.append(row.to_dict() | {
                "label": "chapter",
                "llm_chapter_title": candidate_topic,
                "is_qa_segment": is_qa,
                "anchor_topic": candidate_topic
            })
        else:
            print(f"  🚫 REJECTED (Confidence Score {confidence}% < 75%)")
            rejected.append(row.to_dict())

    # Post-validation backwards Q&A boundary assignment ([-90s, +90s] keyword probability + 2nd-half fallback)
    verified, rejected = assign_qa_boundary_backwards(verified, rejected, full_snippets, max_time, prob_threshold=0.90)

    verified = finalize_pass2_summaries(
        verified, full_snippets, max_time, episode_title, episode_date=episode_date, mode="macro"
    )
    return pd.DataFrame(verified), pd.DataFrame(rejected)

def verify_micro_candidates(
    candidates_df: pd.DataFrame, full_snippets: List[Dict[str, Any]], episode_title: str, scope_start: float, scope_end: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n=== Running 3-Prong MICRO Verification ({LLM_MODEL}) ===")

    verified, rejected = [], []

    first_q_text = extract_text_window(full_snippets, max(0.0, scope_start - 5.0), min(scope_end, scope_start + 60.0))
    current_topic = get_micro_question_anchor(first_q_text)
    print(f"Seeded Initial Q&A @ {scope_start/60:.2f}m -> '{current_topic}'")

    verified.append({
        "timestamp_sec": scope_start,
        "label": "qa_boundary",
        "llm_chapter_title": f"Q&A: {current_topic}",
        "is_qa_segment": True
    })

    for i in range(len(candidates_df)):
        row = candidates_df.iloc[i]
        curr_cand = float(row["timestamp_sec"])
        print(f"\nEvaluating Micro Candidate at {curr_cand / 60:.2f} min (Active: {current_topic}):")

        win_start = max(scope_start, curr_cand - 10.0)
        curr_plus_30 = min(scope_end, curr_cand + 30.0)
        curr_plus_60 = min(scope_end, curr_cand + 60.0)

        pre_text = extract_text_window(full_snippets, win_start, curr_cand)
        post_text_1 = extract_text_window(full_snippets, curr_cand, curr_plus_30)

        inspection_text = f"[PRE-BOUNDARY (-10s to 0s)]\n{pre_text}\n\n[POST-BOUNDARY: FIRST 30s (0s to +30s)]\n{post_text_1}"

        if curr_plus_60 > curr_plus_30:
            post_text_2 = extract_text_window(full_snippets, curr_plus_30, curr_plus_60)
            inspection_text += f"\n\n[POST-BOUNDARY: SECOND 30s (+30s to +60s)]\n{post_text_2}"

        verdict = verify_micro_boundary_window(inspection_text, current_topic, episode_title)

        handoff = verdict.get("has_handoff_phrase_or_username", False)
        question = verdict.get("is_explicit_question_read", False)
        unrelated = verdict.get("is_topic_completely_unrelated", False)

        prong_score = int(handoff) + int(question) + int(unrelated)
        is_boundary = (prong_score >= 2)
        new_topic = verdict.get("window_question_subject", "Unknown Question")

        print(f"  ├─ Handoff Phrase: {handoff} | Explicit Question: {question} | Topic Disconnect: {unrelated}")
        print(f"  ├─ Total Prong Score: {prong_score}/3")
        print(f"  ├─ Reason: {verdict.get('reasoning')}")

        if is_boundary:
            print(f"  ✅ CONFIRMED: '{new_topic}'")
            verified.append(row.to_dict() | {
                "label": "qa_boundary",
                "llm_chapter_title": f"Q&A: {new_topic}",
                "is_qa_segment": True,
                "anchor_topic": new_topic
            })
            current_topic = new_topic
        else:
            print("  🚫 REJECTED (Did not meet 3-prong threshold)")
            rejected.append(row.to_dict())

    verified = finalize_pass2_summaries(verified, full_snippets, scope_end, episode_title, mode="micro")
    return pd.DataFrame(verified), pd.DataFrame(rejected)