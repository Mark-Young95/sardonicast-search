# enrich_dataset.py
import pandas as pd
import numpy as np

# Keyword lists derived from your notes
QA_PHRASES = [
    "question time", "suggestion thread", "sardonicast community", 
    "on the subreddit", "let's do some questions", "questions from"
]

TRANSITION_PHRASES = [
    "we're going to move on", "speaking of", "saw a film", "next up",
    "some movie news", "alright um", "okay um", "we all saw a movie",
    "we saw", "alright well", "well alright", "we also saw"
]

RATING_PHRASES = [
    "stars", "outta ten", "out of ten", "i'm giving it a", "out of five"
]

OUTRO_PHRASES = [
    "i guess that's it", "we've done it", "that was it", "that's it",
    "are we done with questions", "i think we did it", "is it my rec",
    "time for a recommendation", "thanks for listening", "patreon", "thank you", "it's your turn"
]

def extract_phrase_features(text_block: str) -> dict:
    text = text_block.lower()
    return {
        "has_qa_kw": int(any(p in text for p in QA_PHRASES)),
        "has_transition_kw": int(any(p in text for p in TRANSITION_PHRASES)),
        "has_rating_kw": int(any(p in text for p in RATING_PHRASES)),
        "has_outro_kw": int(any(p in text for p in OUTRO_PHRASES))
    }