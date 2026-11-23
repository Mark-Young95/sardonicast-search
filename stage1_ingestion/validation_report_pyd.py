import litellm
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from stage1_ingestion.chapter_segmenter_schema import EpisodeChapters
#from stage1_ingestion.chapter_segmenter import patch_missing_chapters
#from stage1_ingestion.chapter_segmenter import PASS2_SYSTEM_PROMPT

MOD_OA="openai/gpt-4o-mini"

class ValidationReport(BaseModel):
    is_valid: bool = Field(description="True if all major topics/movies from title are present in chapters.")
    missing_topics: List[str] = Field(default=[], description="List of movie titles or topics present in YouTube title/metadata but missing in chapters.")
    feedback: str = Field(description="Brief explanation of why validation passed or failed.")

VALIDATOR_SYSTEM_PROMPT = """
You are a Quality Control Auditor for podcast chapter segmentation.
Compare the YouTube Video Title and Metadata against the extracted Chapters list.

Check:
1. Is every main movie review listed in the YouTube Title represented in its own dedicated chapter?
2. Are there any obviously missing segments?

If any movie from the title is missing from the chapter breakdown, mark `is_valid = False` and list them in `missing_topics`.
"""




PATCH_PROMPT = """
You are a podcast chapter repair assistant for 'Sardonicast'.
Your previous chapter breakdown missed one or more key topics that were explicitly in the YouTube Title/Metadata.

YouTube Title: "{video_title}"
Upload Date: "{upload_date}"

MISSING TOPICS THAT MUST BE ADDED:
{missing_topics_text}

CURRENT INCOMPLETE CHAPTERS:
{current_chapters_text}

FULL EPISODE OUTLINE:
{sparse_transcript}

INSTRUCTIONS:
1. Insert new chapter(s) for the missing topic(s) into their EXACT chronological position within the current chapters.
2. Re-adjust the start and end timestamps so that all chapters remain NON-OVERLAPPING and MONOTONICALLY INCREASING.
3. For each new chapter, ensure you provide an exact, distinct `transition_quote` spoken by a host.
"""





# PASS 2 PROMPT: Final consolidation & structured summary
PASS2_SYSTEM_PROMPT = """
You are an expert podcast editor for 'Sardonicast'.
You are given candidate transition timestamps and a chronological outline of an episode.
Consolidate these into clean, NON-OVERLAPPING, strictly CHRONOLOGICAL episode chapters.

CRITICAL RULES FOR TIMESTAMPS AND ORDER:
1. Chapters MUST be in strict chronological order. Chapter start times MUST increase monotonically (e.g., 00:05:00 -> 00:20:00 -> 00:55:00). NEVER jump backwards in time.
2. The first chapter starts near 00:00:00 .
3. The final chapter MUST end at the very end of the episode outline.

Rules:
1. Ensure the introduction, news (if any), main movie review(s) of recently released movies, recommended film review, listener Q&A, and outro are cleanly separated.
2. Provide the exact start and end timestamps (HH:MM:SS).
3. List all movie titles mentioned in `movies_discussed`, even for the Q&A chapter.
4. While preserving all the movies listed in Chapter.movies_discussed, to construct the Chapter.title, use the YouTube video title and video publish date as references to pick the best choice from the list of movie titles mentioned (but preserve the full list of all movies mentioned in Chapter.movies_discussed).
5. Intro typically starts with music ('[music]') and a countdown '3 2 1'.
6. Outro typically contains a film recommendation for the next episode, include the film title in the list of movie titles mentioned, and also pick the best choice to include in Chapter.title. Outro timestamp should always end on the length of the YouTube video (the end of the video is the end of the outro)

CONTENT COVERAGE RULES:
1. YouTube Title & Upload Date are primary anchors. Ensure ALL major movie reviews mentioned in the title are allocated their own distinct chapter.
2. Do NOT lump movie discussions together into the Intro unless they are brief warm-up chatter. If a movie discussion lasts several minutes, split it into its own chapter.
3. Every listener Q&A segment and final recommendation MUST have its own chapter.

RULES FOR `transition_quote`:
1. Provide a VERY SPECIFIC, unique 4 to 10 word verbatim quote spoken by a host at the EXACT start of the segment.
2. Do NOT use generic phrases like "welcome back" or "next movie". Pick distinctive phrases to allow exact string matching.
"""







### When validate_and_correct_chapters() flags that a topic (like The Invite) is missing from the chapter breakdown, 
# patch_missing_chapters() sends the existing chapters along with the specific missing topic back to Pass 2 with a 
# strict instruction to insert the missing chapter in its correct chronological spot.

def patch_missing_chapters(
    current_chapters: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    missing_topics: List[str],
    video_title: str,
    upload_date: str
) -> List[Dict[str, Any]]:
    """
    Re-runs Pass 2 with explicit targeted instructions to insert missing topics.
    """
    print(f"🔧 Repairing chapters to insert missing topics: {missing_topics}")

    missing_topics_text = "\n".join([f"- {topic}" for topic in missing_topics])
    current_chapters_text = "\n".join([
        f"- [{ch['start_timestamp']} - {ch['end_timestamp']}] {ch['title']}" 
        for ch in current_chapters
    ])
    sparse_transcript = "\n".join([f"[{c['timestamp']}] {c['text'][:100]}..." for c in chunks])

    prompt = PATCH_PROMPT.format(
        video_title=video_title,
        upload_date=upload_date,
        missing_topics_text=missing_topics_text,
        current_chapters_text=current_chapters_text,
        sparse_transcript=sparse_transcript
    )

    try:
        response = litellm.completion(
            model=MOD_OA,
            messages=[
                {"role": "system", "content": PASS2_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format=EpisodeChapters,
            temperature=0.1
        )
        
        parsed = EpisodeChapters.model_validate_json(response.choices[0].message.content)
        repaired_chapters = [ch.model_dump() for ch in parsed.chapters]
        print(f"✅ Successfully patched! New chapter count: {len(repaired_chapters)}")
        return repaired_chapters

    except Exception as e:
        print(f"❌ Patching failed with error: {e}. Falling back to original chapters.")
        return current_chapters




def validate_and_correct_chapters(
    chapters: List[Dict[str, Any]], 
    chunks: List[Dict[str, Any]], 
    video_title: str, 
    upload_date: str
) -> List[Dict[str, Any]]:
    """Validates chapter coverage against metadata and re-runs segmentation if topics are missing."""
    
    chapter_summary = "\n".join([f"- [{ch['start_timestamp']}] {ch['title']} (Movies: {ch.get('movies_discussed', [])})" for ch in chapters])
    
    prompt = f"""
YouTube Video Title: "{video_title}"
Upload Date: "{upload_date}"

Extracted Chapters:
{chapter_summary}
"""

    try:
        response = litellm.completion(
            model=MOD_OA,
            messages=[
                {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format=ValidationReport,
            temperature=0.0
        )
        
        report = ValidationReport.model_validate_json(response.choices[0].message.content)
        
        if not report.is_valid:
            print(f"⚠️ Validation Failed: Missing topics detected -> {report.missing_topics}")
            print(f"Feedback: {report.feedback}")
            print("🔄 Running targeted re-segmentation patch...")
            
            # Re-run Pass 2 with explicit missing topic correction directive
            return patch_missing_chapters(chapters, chunks, report.missing_topics, video_title, upload_date)
            
        print("✅ Validation Passed: All major topics accounted for.")
        return chapters

    except Exception as e:
        print(f"Validation step error: {e}")
        return chapters