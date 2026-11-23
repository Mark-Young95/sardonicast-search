import json
import re
from typing import List, Dict, Any

def preprocess_transcript(
    raw_items: List[Dict[str, Any]], 
    max_words_per_unit: int = 12, 
    min_pause_gap: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Splits raw unpunctuated YouTube transcript fragments into tight, precise 
    semantic units based on audio pause gaps and strict word count caps.
    """
    units = []
    current_words = []
    current_start = None
    last_end_time = 0.0

    for item in raw_items:
        text = str(item.get("text", "")).strip()
        start = float(item.get("start_seconds", 0.0))
        duration = float(item.get("duration", 0.0))
        end = start + duration

        # Omit music/noise markers
        if re.match(r"^\[.*\]$", text) or not text:
            continue

        # Detect audio pause gap between actual speech events
        # (Handling potential timestamp overlaps in raw YouTube JSON)
        time_gap = start - last_end_time if last_end_time > 0 else 0.0

        # Split condition 1: Speech pause detected (> min_pause_gap)
        # Split condition 2: Current buffer exceeds word cap
        if current_words and (time_gap >= min_pause_gap or len(current_words) >= max_words_per_unit):
            unit_text = " ".join(current_words).strip()
            if unit_text:
                units.append({
                    "text": unit_text,
                    "start_seconds": round(current_start, 2),
                    "end_seconds": round(last_end_time, 2)
                })
            current_words = []
            current_start = None

        if current_start is None:
            current_start = start

        # Append individual words with estimated word-level timestamps
        words_in_item = text.split()
        current_words.extend(words_in_item)
        
        # Track end time progression
        if end > last_end_time:
            last_end_time = end

    # Flush remaining words in buffer
    if current_words and current_start is not None:
        unit_text = " ".join(current_words).strip()
        if unit_text:
            units.append({
                "text": unit_text,
                "start_seconds": round(current_start, 2),
                "end_seconds": round(last_end_time, 2)
            })

    return units

if __name__ == "__main__":
    # Example usage on raw JSON structure
    # sample_data = [
    #     {"text": "[Music]", "start_seconds": 0.0, "duration": 7.28},
    #     {"text": "3 2", "start_seconds": 4.68, "duration": 5.72},
    #     {"text": "1 where's my", "start_seconds": 7.28, "duration": 7.16},
    #     {"text": "copy of Dune?", "start_seconds": 14.44, "duration": 3.00}
    # ]
    # sample_data = [
    #     {"text": "[Music]", "start_seconds": 0.0, "duration": 7.28, "timestamp": "00:00:00", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=0s"},
    #     {"text": "3 2", "start_seconds": 4.68, "duration": 5.72, "timestamp": "00:00:04", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=4s"},
    #     {"text": "1 where's my", "start_seconds": 7.28, "duration": 7.16, "timestamp": "00:00:07", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=7s"},
    #     {"text": "snare that was a quote from Eminem Who", "start_seconds": 10.4, "duration": 5.28, "timestamp": "00:00:10", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=10s"},
    #     {"text": "is Loosely", "start_seconds": 14.44, "duration": 4.32, "timestamp": "00:00:14", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=14s"},
    #     {"text": "tied to the film Venom the last dance in", "start_seconds": 15.68, "duration": 4.6, "timestamp": "00:00:15", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=15s"},
    #     {"text": "the sense that he made a song for the", "start_seconds": 18.76, "duration": 4.24, "timestamp": "00:00:18", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=18s"},
    #     {"text": "first one but I feel like I've already", "start_seconds": 20.28, "duration": 5.12, "timestamp": "00:00:20", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=20s"},
    #     {"text": "done knock knock let the devil in or", "start_seconds": 23.0, "duration": 5.2, "timestamp": "00:00:23", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=23s"},
    #     {"text": "that's an awfully hot coffee pot as an", "start_seconds": 25.4, "duration": 5.52, "timestamp": "00:00:25", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=25s"},
    #     {"text": "intro so I don't know yeah you have I'm", "start_seconds": 28.2, "duration": 4.08, "timestamp": "00:00:28", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=28s"},
    #     {"text": "Adam from your movie sex this is", "start_seconds": 30.92, "duration": 4.64, "timestamp": "00:00:30", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=30s"},
    #     {"text": "sardonic cast um I'm Alex from I and we", "start_seconds": 32.28, "duration": 7.16, "timestamp": "00:00:32", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=32s"},
    #     {"text": "are Venom no you like that no I don't I", "start_seconds": 35.56, "duration": 6.08, "timestamp": "00:00:35", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=35s"},
    #     {"text": "hate it I actually hate it a lot no I'm", "start_seconds": 39.44, "duration": 3.92, "timestamp": "00:00:39", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=39s"},
    #     {"text": "really I'm sold on The Voice", "start_seconds": 41.64, "duration": 4.96, "timestamp": "00:00:41", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=41s"},
    #     {"text": "now I'm", "start_seconds": 43.36, "duration": 3.24, "timestamp": "00:00:43", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=43s"},
    #     {"text": "funny yeah What If Venom literally What", "start_seconds": 47.16, "duration": 5.16, "timestamp": "00:00:47", "youtube_link": "https://youtu.be/_bvT65wMeoE?t=47s"}
    # ]
    sample_data = [
        {"text": "[Music]", "start_seconds": 0.0, "duration": 7.28},
        {"text": "3 2", "start_seconds": 4.68, "duration": 5.72},
        {"text": "1 where's my", "start_seconds": 7.28, "duration": 7.16},
        {"text": "snare that was a quote from Eminem Who", "start_seconds": 10.4, "duration": 5.28},
        {"text": "is Loosely", "start_seconds": 14.44, "duration": 4.32},
        {"text": "tied to the film Venom the last dance in", "start_seconds": 15.68, "duration": 4.6},
        {"text": "the sense that he made a song for the", "start_seconds": 18.76, "duration": 4.24},
        {"text": "first one but I feel like I've already", "start_seconds": 20.28, "duration": 5.12},
        {"text": "done knock knock let the devil in or", "start_seconds": 23.0, "duration": 5.2},
        {"text": "that's an awfully hot coffee pot as an", "start_seconds": 25.4, "duration": 5.52},
        {"text": "intro so I don't know yeah you have I'm", "start_seconds": 28.2, "duration": 4.08},
        {"text": "Adam from your movie sex this is", "start_seconds": 30.92, "duration": 4.64},
        {"text": "sardonic cast um I'm Alex from I and we", "start_seconds": 32.28, "duration": 7.16},
        {"text": "are Venom no you like that no I don't I", "start_seconds": 35.56, "duration": 6.08},
        {"text": "hate it I actually hate it a lot no I'm", "start_seconds": 39.44, "duration": 3.92},
        {"text": "really I'm sold on The Voice", "start_seconds": 41.64, "duration": 4.96},
        {"text": "now I'm", "start_seconds": 43.36, "duration": 3.24},
        {"text": "funny yeah What If Venom literally What", "start_seconds": 47.16, "duration": 5.16}
    ]
    # _bvT65wMeoE
    processed = preprocess_transcript(sample_data)
    print(json.dumps(processed, indent=2))