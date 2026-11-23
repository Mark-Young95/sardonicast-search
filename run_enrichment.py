import os
import regex as re
import json
import numpy as np
import pandas as pd

# Comprehensive pattern for common podcast filler grunts and vocalizations
GRUNT_PATTERN = re.compile(
    r"\b("
    r"m+[-_ ]*h+m+|"          # mm-hmm, mh-hm, mhm, mm hmm
    r"u+h*[-_ ]*h+u*h*|"      # uh-huh, uh-uh, uh huh
    # r"u+[mh]+|"               # um, uh, umm, uhh, ummm
    # r"a+h+|"                  # ah, ahh, ahhh
    r"e+r+|"                  # er, err
    r"h+m+"                   # hm, hmm, hmmm
    # r"like"                 # like
    r")\b",
    re.IGNORECASE
)

def clean_grunts(text: str) -> str:
    # 1. Remove grunts
    cleaned = GRUNT_PATTERN.sub("", text)
    # 2. Collapse leftover consecutive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

# Define keyword patterns
KEYWORD_PATTERNS = {
    "qa": [
        re.compile(r"\bquestion[\s\W]*time\b", re.IGNORECASE),
        re.compile(r"\b(?:head[\s\W]+over[\s\W]+)?(?:to[\s\W]+the[\s\W]+)?suggestion[\s\W]*thread\b", re.IGNORECASE),
        re.compile(r"\b(?:sardonic[\s\W]+cast|sardonicast)[\s\W]*(?:community|subreddit)\b", re.IGNORECASE),
        re.compile(r"\bcast\W+(?:community|subreddit)\b", re.IGNORECASE),
        re.compile(fr"\b(?:on[\s\W]+|in[\s\W]+)the[\s\W]*(?:\w\s+){0,2}subreddit\b", re.IGNORECASE),
        re.compile(r"\b(?:ask|leave)[\s\W]+your[\s\W]+own[\s\W]+questions?\b", re.IGNORECASE),
        re.compile(r"\blet['’]s[\s\W]*do[\s\W]*(?:some|a[\s\W]+few)[\s\W]*questions(?:[\s\W]+then)?\b", re.IGNORECASE),
        re.compile(r"\bquestions[\s\W]*from\b", re.IGNORECASE),
        re.compile(r"\btime[\s\W]*for[\s\W]*questions\b", re.IGNORECASE),
        re.compile(r"\bhead[\s\W]*over[\s\W]*to[\s\W]*the\b", re.IGNORECASE),
        re.compile(r"\bthere['’]s[\s\W]+some[\s\W]+questions(?:[\s\W]+right[\s\W]+here)?\b", re.IGNORECASE),
        re.compile(r"\b(?:we['’]re[\s\W]+gonna[\s\W]+go[\s\W]+)?into[\s\W]+some[\s\W]+questions\b", re.IGNORECASE),
    ],
    "outro": [
        re.compile(r"\b(time[\s\W]+to[\s\W]+recommend[\s\W]+a[\s\W]+(?:film|movie)|what(?:[\s\W]+have)?[\s\W]+you[\s\W]+got)\b", re.IGNORECASE),
        re.compile(r"\b(i[\s\W]+guess[\s\W]+that['’]s[\s\W]+it|we['’]ve[\s\W]+done[\s\W]+it|it['’]s[\s\W]+me[\s\W]+is[\s\W]+it)\b", re.IGNORECASE),
        re.compile(r"\b(?:someone['’]s[\s\W]+|i[\s\W]+)got[\s\W]+to[\s\W]+recommend(?:[\s\W]+a[\s\W]+film|[\s\W]+a[\s\W]+movie)?\b", re.IGNORECASE),
        re.compile(r"\b((?:i[\s\W]+think[\s\W]+)?we[\s\W]+did[\s\W]+it|that[\s\W]+was[\s\W]+it|that['’]s[\s\W]+it)\b", re.IGNORECASE),
        re.compile(r"\b(that['’]s[\s\W]+it[\s\W]+for[\s\W]+questions|are[\s\W]+we[\s\W]+done[\s\W]+with[\s\W]+questions)\b", re.IGNORECASE),
        re.compile(r"\b(i[\s\W]+think[\s\W]+we[\s\W]+did[\s\W]+it|is[\s\W]+it[\s\W]+my[\s\W]+(?:wreck|rec)|it['’][\s\W]+my[\s\W]+(?:wreck|rec))\b", re.IGNORECASE),
        re.compile(r"\b(time[\s\W]+for[\s\W]+(?:the|a)(?:[\s\W]+film)?[\s\W]+recommendation|(?:i[\s\W]+guess[\s\W]+)?there['’]s[\s\W]+a[\s\W]+recommendation)\b", re.IGNORECASE),
        re.compile(r"\b(i(?:['’]ve)?[\s\W]+got[\s\W]+a(?:[\s\W]+film)?[\s\W]+recommendation|good[\s\W]+episode)\b", re.IGNORECASE),
        re.compile(r"\b((?:we|you)[\s\W]+have[\s\W]+a(?:[\s\W]+film)?[\s\W]+recommendation|there[\s\W]+we[\s\W]+are)\b", re.IGNORECASE),
        re.compile(r"\bthank[\s\W]+you|(?:thank[\s\W]+you|thanks)[\s\W]+for[\s\W]+(?:your|the)[\s\W]+questions\b", re.IGNORECASE),
        re.compile(r"\bwe[\s\W]+(?:have|got)[\s\W]+a[\s\W]+guest[\s\W]+joining\b", re.IGNORECASE),
        re.compile(r"\b(good[\s\W]+questions|it['’]s[\s\W]+(?:my|your)[\s\W]+turn(?:[\s\W]to[\s\W]recommend)?|about[\s\W]+wraps[\s\W]+things[\s\W]+up)\b", re.IGNORECASE),
        re.compile(r"\bthat[\s\W]+about[\s\W]+does[\s\W]+it\b", re.IGNORECASE),
        re.compile(r"\b(?:for|on|joining)\W+the\W+next\W+episode\b", re.IGNORECASE),
        re.compile(r"\bthat['’]s[\s\W]+the[\s\W]+end[\s\W]+of[\s\W]+our[\s\W]+show\b", re.IGNORECASE),
        re.compile(r"\byou[\s\W]+have[\s\W]+a[\s\W]+film[\s\W]+to[\s\W]+recommend\b", re.IGNORECASE)
    ],
    "transition": [
        re.compile(r"\bthe[\s\W]*oscars[\s\W]*happened\b", re.IGNORECASE),
        re.compile(r"\bgo[\s\W]+through[\s\W]+the[\s\W]+winners\b", re.IGNORECASE),
        re.compile(r"\bwe['’]re(?:[\s\W]*gonna|[\s\W]*going[\s\W]*to)[\s\W]*move[\s\W]*on\b", re.IGNORECASE),
        re.compile(r"\bwe['’]re[\s\W]+going[\s\W]+to[\s\W]+talk[\s\W]+about(?:[\s\W]+it)?\b", re.IGNORECASE),
        re.compile(r"\bquick[\s\W]+review\b", re.IGNORECASE),
        re.compile(r"\blet['’]s\W+do\b", re.IGNORECASE),
        re.compile(r"\bwhat\W+did\W+you\W+all\W+think\b", re.IGNORECASE),
        re.compile(r"\bdo(?:\W+\w+){0,4}\W+start\W+with\b", re.IGNORECASE),
        re.compile(r"\b(?:all[\s\W]+right[\s\W]+|alright[\s\W]+)?did[\s\W]+(?:you|we|you[\s\W]+guys)[\s\W]+want[\s\W]+to[\s\W]+move[\s\W]+(?:on|into)\b", re.IGNORECASE),
        re.compile(r"\b(?:all[\s\W]+right[\s\W]+|alright[\s\W]+|uh[\s\W]+)?we[\s\W]+also[\s\W]+(?:saw|watched|watch)\b", re.IGNORECASE),
        re.compile(r"\b(?:all[\s\W]+right|alright)[\s\W]+there['’]s[\s\W]+a[\s\W]+(?:movie|film)\b", re.IGNORECASE),
        re.compile(r"\bspeaking[\s\W]*of|\w+\s+and\s+\w+\s+(?:saw|watched)\s+(?:a|another)\s+(?:movie|film)\b", re.IGNORECASE),
        re.compile(r"\bnext[\s\W]*up\b", re.IGNORECASE),
        re.compile(r"\blet['’]s\W+move\W+on\b", re.IGNORECASE),
        re.compile(r"\bbefore\W+we(?:[\s\W]+\w+){0,4}\W+get\W+into\W+questions\b", re.IGNORECASE),
        re.compile(r"\bhow\W+you\W+say\W+it\W+is\W+it\b", re.IGNORECASE),
        re.compile(r"\bmove\W+on\W+to\W+the\W+next\W+(?:film|movie)\b", re.IGNORECASE),
        re.compile(r"\bsome[\s\W]*movie[\s\W]*news\b", re.IGNORECASE),
        re.compile(r"\bin[\s\W]*other[\s\W]*news\b", re.IGNORECASE),
        re.compile(r"\b(?:ok|okay)[\s\W]*um\b", re.IGNORECASE),
        re.compile(r"\bhow\W+shall\W+we\W+dive\W+in\b", re.IGNORECASE),
        re.compile(r"\bahead\W+of\W+us\W+today\b", re.IGNORECASE),
        re.compile(r"\bwe\W+both\W+saw\W+the\W+film\b", re.IGNORECASE),
        re.compile(r"\bwhat\W+are\W+we\W+talking\W+about\W+today\b", re.IGNORECASE),
        re.compile(r"\b(?:all[\s\W]+right|alright|well)[\s\W]*\w+\W+we(?:[\s\W]*all)?(?:[\s\W]*also)?[\s\W]*(?:watched|saw)\b", re.IGNORECASE),
        re.compile(r"\bwe(?:[\s\W]*all)?(?:[\s\W]*also)?[\s\W]*(?:watched|saw)[\s\W]*(?:a|another)(?:[\s\W]+\w+){0,2}[\s\W]*(?:movie|film)\b", re.IGNORECASE),
        re.compile(r"\b(?:movie|film)[\s\W]*we(?:[\s\W]*also)?[\s\W]*(?:watched|saw)\b", re.IGNORECASE),
        re.compile(r"\banother\W+film\W+that\W+we\W+saw\b", re.IGNORECASE),
        re.compile(r"\bspeaking[\s\W]+of(?:[\s\W]+\w+){0,6}[\s\W]+we(?:[\s\W]+all)?[\s\W]+saw\b", re.IGNORECASE),
        re.compile(r"\bi[\s\W]+had[\s\W]+a[\s\W]+recommendation[\s\W]+for[\s\W]+a[\s\W]+(?:movie|film)\b", re.IGNORECASE),
        re.compile(r"\bwe['’]re[\s\W]+going[\s\W]+to[\s\W]+spoil[\s\W]+the\s+\w+\s+out[\s\W]+of\b", re.IGNORECASE),
        re.compile(r"\b(?:all[\s\W]+right|alright)[\s\W]*well\b", re.IGNORECASE),
        re.compile(r"\bwell[\s\W]*(?:all[\s\W]+right|alright)\b", re.IGNORECASE),
        re.compile(r"\b(?:all[\s\W]+right|alright)[\s\W]*let['’]s[\s\W]*talk[\s\W]*about\b", re.IGNORECASE),
        re.compile(r"\bi[\s\W]+guess[\s\W]+we['’]ll[\s\W]+talk[\s\W]+about\b", re.IGNORECASE),
        re.compile(r"\byou(?:[\s\W]+guys)?[\s\W]*(?:want|wanted)[\s\W]*to[\s\W]*talk[\s\W]*about\b", re.IGNORECASE),
        re.compile(r"\bwe['’]re[\s\W]+gonna(?:[\s\W]+\w+){1,4}[\s\W]+talk[\s\W]+about\b", re.IGNORECASE),
        re.compile(r"\b(?:all[\s\W]+right|alright)(?:[\s\W]*I[\s\W]*guess)?[\s\W]*(?:um|uh)?\b", re.IGNORECASE),
        re.compile(r"\bmy[\s\W]*(?:rec/wreck)\b", re.IGNORECASE),
        re.compile(r"\bare[\s\W]+we[\s\W]+moving[\s\W]+forward[\s\W]+to[\s\W]+the[\s\W]+recommendation\b", re.IGNORECASE),
        re.compile(r"\b(?:it\W+was\W+)?my\W+recommendation(?:\W+was)?\b", re.IGNORECASE),
        re.compile(r"\bi[\s\W]+recommended\b", re.IGNORECASE),
        re.compile(r"\byou[\s\W]+recommended\b", re.IGNORECASE),
        re.compile(r"\brecommended\W+a\W+film\b", re.IGNORECASE),
        re.compile(r"\bwe\W+ready\W+for\W+the\b", re.IGNORECASE),
        re.compile(r"\ball\W+right\W+so\W+are\W+we\W+talking\b", re.IGNORECASE),
        re.compile(r"\bgo\W+for\W+it\W+talk\W+about\b", re.IGNORECASE),
        re.compile(r"\bstart\W+doing\W+the\W+movie\W+recommendation\W+discussion\b", re.IGNORECASE),
        re.compile(r"\bi(?:[\s\W]+just)?[\s\W]+(?:saw|watched)[\s\W]+(?:an|a)(?:[\s\W]+\w+){0,2}[\s\W]+(?:film|movie)(?:[\s\W]+recently)?\b", re.IGNORECASE),
        re.compile(r"\bthat['’]s[\s\W]+funny[\s\W]+(?:\w+\s+)?we['’]re[\s\W]+(?:\w+\s+)?talking[\s\W]+about\b", re.IGNORECASE),
        re.compile(r"\bwe['’]re\W+talking\W+about\b", re.IGNORECASE),
        re.compile(r"\btime\W+there\W+is\W+a\W+movie\b", re.IGNORECASE),
        re.compile(r"\b(?:all[\s\W]+right|alright)[\s\W]+(?:so)?[\s\W]+we[\s\W]+have[\s\W]+the[\s\W]+film[\s\W]+recommendation\b", re.IGNORECASE),
        re.compile(r"\bwe[\s\W]+have[\s\W]+a[\s\W]+recommendation[\s\W]+from\b", re.IGNORECASE),
        re.compile(r"\bthis[\s\W]+is[\s\W]+a(?:[\s\W]+bit[\s\W]+of[\s\W]+a)?[\s\W]+tangent\b", re.IGNORECASE),
        re.compile(r"\bfirst[\s\W]+we['’]ll[\s\W]+talk[\s\W]+about\b", re.IGNORECASE),
        re.compile(r"\bwe[\s\W]+(?:could|should)[\s\W]+start[\s\W]+talking[\s\W]+about\b", re.IGNORECASE),
        re.compile(r"\bso[\s\W]+there['’]s[\s\W]+some(?:[\s\W]+\w+){0,2}[\s\W]+news\b", re.IGNORECASE),
        re.compile(r"\b(?:all[\s\W]+right[\s\W]+|alright[\s\W]+)?there[\s\W]+was[\s\W]+a[\s\W]+film[\s\W]+recommendation\b", re.IGNORECASE),
        re.compile(r"\bhow[\s\W]+are[\s\W]+we[\s\W]+feeling[\s\W]+about[\s\W]+this[\s\W]+movie\b", re.IGNORECASE),
        re.compile(r"\blet['’]s[\s\W]+talk[\s\W]+about[\s\W]+it\b", re.IGNORECASE),
        re.compile(r"\btoday['’]s[\s\W]+recommendation\b", re.IGNORECASE),
        re.compile(r"\bare[\s\W]+we[\s\W]+going[\s\W]+to[\s\W]+mention\b", re.IGNORECASE),
        re.compile(r"\bi[\s\W]+guess[\s\W]+it['’]s[\s\W]+time\b", re.IGNORECASE),
        re.compile(r"\blast[\s\W]+episode\b", re.IGNORECASE),
        re.compile(r"\bgood[\s\W]+seg\b", re.IGNORECASE),
        re.compile(r"\bthat\W+was\W+a\W+great\W+conversation\b", re.IGNORECASE),
        re.compile(r"\bleads[\s\W]+into[\s\W]+my[\s\W]+(?:rec|wreck)\b", re.IGNORECASE),
        re.compile(r"\bthis[\s\W]+is[\s\W]+my[\s\W]+(?:rec|wreck)\b", re.IGNORECASE),
        re.compile(r"\b(?:film[\s\W]+|movie[\s\W]+)that[\s\W]+we[\s\W]+watched\b", re.IGNORECASE),
        re.compile(r"\bthere['’]s[\s\W]+a[\s\W]+(?:film|movie)[\s\W]+that[\s\W]+was[\s\W]+recommended[\s\W]+last[\s\W]+episode\b", re.IGNORECASE),
        re.compile(r"\bwe[\s\W]+should[\s\W]+start[\s\W]+off[\s\W]+with\b", re.IGNORECASE),
        re.compile(r"\bwhat[\s\W]+are[\s\W]+we[\s\W]+doing[\s\W]+next\b", re.IGNORECASE),
        re.compile(r"\b(?:all[\s\W]+right|alright)[\s\W]+what['’]s[\s\W]+next\b", re.IGNORECASE),
        re.compile(r"\bthis[\s\W]+is[\s\W]+the[\s\W]+film[\s\W]+discussion[\s\W]+part\b", re.IGNORECASE),
        re.compile(r"\btime[\s\W]+for[\s\W]+the[\s\W]+recommendation\b", re.IGNORECASE),
        re.compile(r"\bso[\s\W]+take[\s\W]+it[\s\W]+away\b", re.IGNORECASE),
        re.compile(r"\byou\W+guys\W+watched\W+it\b", re.IGNORECASE),
        re.compile(r"\bthere\W+was\W+a(?:\W+\w+){0,2}\W+remake\b", re.IGNORECASE),
        re.compile(r"\bthat\W+was\W+a\W+good\W+transition\b", re.IGNORECASE),
        re.compile(r"\bdirected\W+by\b", re.IGNORECASE),
        re.compile(r"\bsegway\b", re.IGNORECASE)
    ],
    "rating": [
        re.compile(r"\b(stars"
                r"|this[\s\W]+(?:would[\s\W]+be|is)(?:[\s\W]+like)?[\s\W]+a[\s\W]+(?:high|low)[\s\W]+(?:\d|one|two|three|four|five)"
                r"|(?:i\W+gave|i['’]m\W+giving)\W+(?:it|this|this\W+one)(?:\W+like)?\W+(?:a|an)\W+(?:\d|half|one|two|three|four|five|six|seven|eight|nine|ten)|stars?\W+(?:for|from)\W+me"
                r"|(?:stars|star)?(?:[\s\W]+at|[\s\W]+outta|[\s\W]+out[\s\W]+of)[\s\W]+(?:10|5|ten|five)"
                r"|(?:\d|one|two|three|four|five|six|seven|eight|nine|ten)(?:[\s\W]+star)?(?:s)?[\s\W]+out[\s\W]+of[\s\W]+(?:10|5|five|ten))\b", re.IGNORECASE)
    ],
    "filler": [
        re.compile(
            r"\b(m+[-_ ]*h+m+|u+h*[-_ ]*h+u*h*|e+r+|h+m+|um|uh|sure|everybody|anyway|yeah\W+yeah|alright|all[\s\W]+right|okay|ok|i[\s\W]+guess|well|perfect|awesome)\b",
            re.IGNORECASE
        )
    ],
    "guess": [
        # Match 'predict' words normally
        re.compile(r"\bpredict(?:ion|ions|ing|ed|s)?\b", re.IGNORECASE),
        
        # Match 'guess' words ONLY if they are NOT immediately preceded by "i "
        re.compile(r"(?<!\bi[\s\W]+)\bguess(?:es|ing|ed)?\b", re.IGNORECASE),

        # Joking about fake score
        re.compile(r"\b(top[\s\W]+level|no)\b", re.IGNORECASE)
    ],
    "spoiler": [
        re.compile(r"\b(?:if[\s\W]+you[\s\W]+don['’]t[\s\W]+want[\s\W]+to[\s\W]+be[\s\W]+spoiled)\b", re.IGNORECASE)
    ]
}

# Standalone high-priority 3x question pattern
THREE_QUESTIONS_PATTERN = re.compile(r"(?:\bquestions?\b[\s\S]*?){3}", re.IGNORECASE)



def count_keywords(text: str, patterns: list) -> int:
    if not isinstance(text, str): return 0
    return sum(len(p.findall(text)) for p in patterns)

def extract_titles_from_description(description: str) -> list:
    """Extracts movie/chapter titles from standard YouTube timestamp lines."""
    if not isinstance(description, str): return []
    titles = []
    # Looks for lines like "53:14 Tokyo Godfathers" or "1:29:30 Questions"
    for line in description.split('\n'):
        match = re.match(r'^\d{1,2}:\d{2}(?::\d{2})?\s+(.+)$', line.strip())
        if match:
            title = match.group(1).strip().lower()
            # Ignore generic chapters to focus on actual movie titles
            if title not in ["questions", "Q&A", "recommendation + outro", "recommendation", "outro"]:
                titles.append(title)
    return titles

def reconstruct_windowed_text(df: pd.DataFrame, cache_dir: str, window_sec: float = 60.0) -> pd.DataFrame:
    """
    Reads the JSON cache and reconstructs the sliding window text 
    for each timestamp present in the ML dataframe.
    """
    print("Reconstructing sliding window text from JSON caches...")
    text_mapping = []
    title_mentions = []

    for vid_id, group in df.groupby("video_id"):
        json_path = os.path.join(cache_dir, f"{vid_id}.json")
        
        if not os.path.exists(json_path):
            print(f"  ⚠️ Warning: No cache found for {vid_id}")
            text_mapping.extend([""] * len(group))
            continue
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        snippets = data.get("snippets", [])
        if not snippets:
            text_mapping.extend([""] * len(group))
            title_mentions.extend([0] * len(group))
            continue

        description = data.get("meta", {}).get("description", "")
        episode_titles = extract_titles_from_description(description)
            
        snippet_times = np.array([s["start_seconds"] for s in snippets], dtype=np.float64)
        snippet_texts_arr = np.array([s["text"] for s in snippets])
        
        # Ensure we process rows in the exact order they appear in the group
        for curr_time in group["timestamp_sec"]:
            left_mask = (snippet_times >= (curr_time - window_sec)) & (snippet_times < curr_time)
            right_mask = (snippet_times >= curr_time) & (snippet_times < (curr_time + window_sec))
            combined_mask = left_mask | right_mask
            
            combined_text = " ".join(snippet_texts_arr[combined_mask]).lower()
            text_mapping.append(combined_text)

            # NEW: Check if any of the episode's specific movie titles are in this chunk
            has_title = int(any(title in combined_text for title in episode_titles))
            title_mentions.append(has_title)
            
    # Assign reconstructed text back to dataframe
    df = df.sort_values(by=["video_id", "timestamp_sec"]).reset_index(drop=True)
    df["text"] = text_mapping
    #df["mentions_chapter_title"] = title_mentions
    return df

def enrich_transcript_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Starting enrichment on {len(df)} rows...")

    # ---------------------------------------------------------
    # NEW: IDEMPOTENCY CLEANUP
    # Detect if we are re-running on an already enriched dataset 
    # and drop the old columns to guarantee a completely fresh run.
    # ---------------------------------------------------------
    generated_cols = [
        "ep_duration_sec", "relative_position", "time_from_start_sec", "time_from_end_sec",
        "has_3x_questions", "word_count", "time_idx", "rating_upcoming_60s", "spoiler_upcoming_5m"
    ]
    for kw_type in KEYWORD_PATTERNS.keys():
        generated_cols.extend([f"{kw_type}_count", f"{kw_type}_density", f"{kw_type}_rolling_5min"])
        
    cols_to_drop = [col for col in generated_cols if col in df.columns]
    if cols_to_drop:
        print(f"  🧹 Found {len(cols_to_drop)} existing enriched columns. Dropping them for a fresh recalculation...")
        df = df.drop(columns=cols_to_drop)
    # ---------------------------------------------------------
    
    # 1. Temporal & Positional Features
    ep_durations = df.groupby("video_id")["timestamp_sec"].transform("max")
    df["ep_duration_sec"] = ep_durations
    
    # Safely calculate relative position
    df["relative_position"] = df["timestamp_sec"] / (df["ep_duration_sec"] + 1e-5)
    df["time_from_start_sec"] = df["timestamp_sec"]
    df["time_from_end_sec"] = df["ep_duration_sec"] - df["timestamp_sec"]

    # 2. Extract Keyword Counts & Densities
    df["has_3x_questions"] = df["text"].apply(
        lambda txt: 1 if bool(THREE_QUESTIONS_PATTERN.search(txt)) else 0
    )
    
    df["word_count"] = df["text"].apply(lambda txt: len(txt.split()))
    safe_wc = df["word_count"].replace(0, 1) # Prevent div by zero
    
    # 2a. Calculate Raw Counts First
    for kw_type, patterns in KEYWORD_PATTERNS.items():
        df[f"{kw_type}_count"] = df["text"].apply(lambda txt: count_keywords(txt, patterns))
        
    # 2b. The Guessing Nullifier (+/- 15 seconds)
    # Set up time index for backward roll
    df["time_idx"] = pd.to_timedelta(df["timestamp_sec"], unit="s")
    df = df.set_index("time_idx")
    guess_past = df.groupby("video_id")["guess_count"].rolling("15s").sum().reset_index(level=0, drop=True)
    df = df.reset_index()
    
    # Set up inverted time index for forward roll
    df["inv_time_sec"] = df["ep_duration_sec"] - df["timestamp_sec"]
    df["inv_time_idx"] = pd.to_timedelta(df["inv_time_sec"], unit="s")
    df_fwd = df.sort_values(["video_id", "inv_time_sec"]).set_index("inv_time_idx")
    
    guess_future = df_fwd.groupby("video_id")["guess_count"].rolling("15s").sum().reset_index(level=0, drop=True)
    df_fwd = df_fwd.reset_index().sort_values(["video_id", "timestamp_sec"])
    
    # If a guess happened within 15s (past or future), erase the rating count!
    guessing_zone_mask = (guess_past.values + guess_future.values) > 0
    df.loc[guessing_zone_mask, "rating_count"] = 0
    
    # Cleanup temporary columns
    df = df.drop(columns=["inv_time_sec", "inv_time_idx", "time_idx"])
    
    # 2c. Calculate Densities
    for kw_type in KEYWORD_PATTERNS.keys():
        # We don't necessarily need guess density/rolling features for ML, just the count for nullifying
        if kw_type != "guess": 
            df[f"{kw_type}_density"] = df[f"{kw_type}_count"] / safe_wc

    # 3. Rolling Local Context Features (5-Minute Window)
    # Setup a time-based index for pandas rolling window
    df["time_idx"] = pd.to_timedelta(df["timestamp_sec"], unit="s")
    df = df.set_index("time_idx")

    # Sum keyword occurrences in a 5 minute (300s) rolling window
    for kw_type in KEYWORD_PATTERNS.keys():
        col_name = f"{kw_type}_count"
        rolling_col = f"{kw_type}_rolling_5min"
        # Calculate rolling sum grouped by video
        df[rolling_col] = df.groupby("video_id")[col_name].rolling("300s").sum().reset_index(level=0, drop=True)



    # 4. Lookahead Features (The 60-Second Penalty Zone)
    # To look forward in time, we create an inverted time index (Time remaining in episode)
    df = df.reset_index() # Temporarily pull time_idx out
    df["inv_time_sec"] = df["ep_duration_sec"] - df["timestamp_sec"]
    df["inv_time_idx"] = pd.to_timedelta(df["inv_time_sec"], unit="s")
    
    # Sort backwards so we can roll "forward" seamlessly
    df_forward = df.sort_values(["video_id", "inv_time_sec"]).set_index("inv_time_idx")
    
    # Calculate the upcoming ratings in the next 60 seconds
    df_forward["rating_upcoming_60s"] = df_forward.groupby("video_id")["rating_count"].rolling("60s").sum().reset_index(level=0, drop=True)

    # NEW: 5-minute (300s) Spoiler Lookahead
    df_forward["spoiler_upcoming_5m"] = df_forward.groupby("video_id")["spoiler_count"].rolling("300s").sum().reset_index(level=0, drop=True)
    
    # Merge it safely back into our main dataframe in the correct chronological order
    df_forward = df_forward.reset_index().sort_values(["video_id", "timestamp_sec"])
    df["rating_upcoming_60s"] = df_forward["rating_upcoming_60s"].values
    df["spoiler_upcoming_5m"] = df_forward["spoiler_upcoming_5m"].values
    
    # Clean up the temporary inversion columns
    df = df.drop(columns=["inv_time_sec", "inv_time_idx", "time_idx"])



    # Cleanup index and drop temporary text column so XGBoost doesn't choke on strings
    df = df.reset_index(drop=True)
    df = df.drop(columns=["text"])
    
    return df

if __name__ == "__main__":
    ML_CSV_PATH = "boundary_112_five.csv"  # The file generated by build_dataset.py
    # use boundary_dataset_enriched_v5 for boundaries
    # boundary_112_five for 5-sec, 114-records
    # use boundary_qa.csv for qa
    # use boundary_qa_five for qa
    CACHE_DIR = "transcript_cache"        # Directory holding your .json files
    OUTPUT_PATH = "boundary_112_enriched_five.csv"
    # boundary_112_enriched_five for 5-sec, 114-record
    # boundary_qa_enriched_five for 5-sec, 30-record qa
    
    # 1. Load the raw ML primitives
    df = pd.read_csv(ML_CSV_PATH)
    
    # 2. Reconstruct the windowed text used during vector encoding
    df = reconstruct_windowed_text(df, CACHE_DIR, window_sec=60.0)
    
    # 3. Calculate keyword densities and rolling context features
    enriched_df = enrich_transcript_dataset(df)
    
    # 4. Save
    enriched_df.to_csv(OUTPUT_PATH, index=False)
    print(f"🎉 Success! Enriched dataset saved to {OUTPUT_PATH}")