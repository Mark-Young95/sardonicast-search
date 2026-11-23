from typing import List, Optional
from pydantic import BaseModel, Field

class TransitionCandidate(BaseModel):
    timestamp: str = Field(description="Timestamp in HH:MM:SS format where a topic shift occurs.")
    topic_hint: str = Field(description="Short phrase indicating what is starting (e.g., 'Review of The Piano', 'Community Q&A').")

class DetectedTransitions(BaseModel):
    transitions: List[TransitionCandidate]

class Chapter(BaseModel):
    title: str = Field(description="Clear title of the segment (e.g. 'Review: The Piano (1993)').")
    start_timestamp: str = Field(description="Start time of this chapter in HH:MM:SS.")
    end_timestamp: str = Field(description="End time of this chapter in HH:MM:SS.")
    movies_discussed: List[str] = Field(default=[], description="Exact film titles discussed.")
    transition_quote: str = Field(description="The exact 3-7 word phrase spoken when hosts switch to this topic.")
    summary: str = Field(description="Concise 2-3 sentence summary of the discussion.")

class EpisodeChapters(BaseModel):
    chapters: List[Chapter]