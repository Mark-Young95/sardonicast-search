import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import litellm
import sys
from pathlib import Path

# Add the parent directory (project root) to Python's search path
sys.path.append(str(Path(__file__).resolve().parent.parent))
load_dotenv()

# Initialize OpenAI client (or set base_url for Ollama/LiteLLM)
groq_api_key = os.getenv("GROQ_API_KEY")
print("groq_api_key found: ", groq_api_key[:6])
#client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)
MOD_LL="groq/llama-3.3-70b-versatile"
#gpt-4o-mini
MOD_OA="openai/gpt-4o-mini"

from chapter_segmenter_schema import EpisodeChapters

REFINED_SYSTEM_PROMPT = """
You are an expert podcast editor and transcription analyzer for 'Sardonicast'.
Your task is to review a continuous transcript with timestamp markers and extract every single major discussion topic, movie review, news topic, and listener Q&A block.

Structure of a Sardonicast Episode:
1. INTRO: Opening quotes, banter, housekeeping.
2. RECENT MOVIES / NEWS: Discussion of 0-2 newly released movies or film news/Oscars.
3. EPISODE FILM RECOMMENDATION: Comprehensive review of the movie recommended in the PREVIOUS episode. (Crucial!)
4. LISTENER Q&A: The crew answers questions from the community.
5. NEXT EPISODE RECOMMENDATION: A host picks/recommends a film for the next episode.
6. OUTRO: Final thoughts, social links, sign-off.

Extraction Rules:
- DO NOT miss any movie title being discussed. Pay attention when hosts transition topics (e.g., "Moving on to...", "My recommendation was...", "Next up...").
- For any movie discussion, list the exact movie title in `movies_discussed`.
- Timestamps must correspond accurately to when the transition happens in the text.
"""

def extract_chapters_full_context(
    chunks: List[Dict[str, Any]], 
    model_name: str = "gpt-4o-mini"
) -> List[Dict[str, Any]]:
    """
    Feeds continuous transcript blocks with full text (no truncation) to the LLM.
    """
    if not chunks:
        return []

    # Build FULL continuous transcript text with timestamp cues
    # Format: [HH:MM:SS] Full chunk text...
    transcript_blocks = []
    for c in chunks:
        transcript_blocks.append(f"[{c['timestamp']}] {c['text']}")
    
    full_transcript_text = "\n".join(transcript_blocks)

    try:
        # response = litellm.completion(
        #     model=MOD_LL,
        #     messages=[
        #         {"role": "system", "content": REFINED_SYSTEM_PROMPT},
        #         {
        #             "role": "user", 
        #             "content": f"Analyze this full podcast transcript and extract all chapters, movie reviews, and Q&A segments:\n\n{full_transcript_text}"
        #         }
        #     ],
        #     response_format=EpisodeChapters,
        #     temperature=0.1  # Low temperature for precise extraction
        # )

        # raw_json_string = response.choices[0].message.content
        # parsed_data = EpisodeChapters.model_validate_json(raw_json_string)
        # return [ch.model_dump() for ch in parsed_data.chapters]



        # OpenAI
        response = litellm.completion(
            model=MOD_OA,
            messages=[
                {"role": "system", "content": REFINED_SYSTEM_PROMPT},
                {
                    "role": "user", 
                    "content": f"Analyze this full podcast transcript and extract all chapters, movie reviews, and Q&A segments:\n\n{full_transcript_text}"
                }
            ],
            response_format=EpisodeChapters,
            temperature=0.1  # Low temperature for precise extraction
        )

        raw_json_string = response.choices[0].message.content
        parsed_data = EpisodeChapters.model_validate_json(raw_json_string)
        return [ch.model_dump() for ch in parsed_data.chapters]

    except Exception as e:
        print(f"Error extracting chapters: {e}")
        return []


if __name__ == "__main__":
    from transcript_fetcher import fetch_youtube_transcript
    from chunker import create_sliding_window_chunks
    print("groq_api_key found: ", groq_api_key[:6])

    TEST_VIDEO_ID = "9OZAYwBtTHo"
    print("Fetching transcript and creating chunks...")
    snippets = fetch_youtube_transcript(TEST_VIDEO_ID)
    chunks = create_sliding_window_chunks(snippets)

    print(f"Segmenting {len(chunks)} chunks into high-level chapters with {MOD_OA}...")
    chapters = extract_chapters_full_context(chunks)

    print(f"\nSuccessfully extracted {len(chapters)} Chapters:\n")
    for ch in chapters:
        print(f"📌 [{ch['start_timestamp']}] {ch['title']}")
        print(f"   Summary: {ch['summary']}\n")