import sys
import re
import json
from typing import List, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import urllib.request

def fetch_youtube_metadata(video_id: str) -> Dict[str, Any]:
    """
    Fetches video title, publish date, and full description directly from YouTube page metadata.
    Returns: {"title": str, "upload_date": str, "description": str}
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    metadata = {"title": "", "upload_date": "", "description": ""}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            
            # Extract title from meta tag
            title_match = re.search(r'<meta property="og:title" content="([^"]+)">', html)
            if title_match:
                metadata["title"] = title_match.group(1).replace(" - YouTube", "")
                
            # Extract publish date from item-prop or schema meta tag
            date_match = re.search(r'<meta itemprop="uploadDate" content="([^"]+)">', html) or \
                         re.search(r'<meta itemprop="datePublished" content="([^"]+)">', html)
            if date_match:
                # ISO date format like "2024-03-15T00:00:00-07:00" -> extract "2024-03-15"
                metadata["upload_date"] = date_match.group(1).split("T")[0]

            # 3. Extract Description from ytInitialPlayerResponse
            desc_match = re.search(r'"shortDescription":"(.*?)","isCrawlable"', html)
            if desc_match:
                # Unescape escaped unicode & backslashes (e.g. \n, \u0026)
                raw_desc = desc_match.group(1)
                metadata["description"] = json.loads(f'"{raw_desc}"')
            else:
                # Fallback to og:description meta tag
                og_desc_match = re.search(r'<meta property="og:description" content="([^"]*)">', html)
                if og_desc_match:
                    metadata["description"] = og_desc_match.group(1)

    except Exception as e:
        print(f"Warning: Could not fetch metadata for video {video_id}: {e}")

    return metadata

def format_timestamp(seconds: float) -> str:
    """Converts seconds into HH:MM:SS format."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def fetch_youtube_transcript(video_id: str, languages: List[str] = ['en']) -> List[Dict[str, Any]]:
    """
    Fetches the transcript for a given YouTube video ID.
    Returns a list of dictionaries with text, start time (sec), duration, and formatted timestamp.
    """
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=languages)

        formatted_transcript = []
        for entry in transcript:
            start_sec = entry.start
            duration = entry.duration
            text = entry.text.replace('\n', ' ').strip()
            
            # Skip empty lines
            if not text:
                continue
                
            formatted_transcript.append({
                "text": text,
                "start_seconds": round(start_sec, 2),
                "duration": round(duration, 2),
                "timestamp": format_timestamp(start_sec),
                "youtube_link": f"https://youtu.be/{video_id}?t={int(start_sec)}s"
            })
            
        return formatted_transcript

    except TranscriptsDisabled:
        print(f"Error: Transcripts are disabled for video ID: {video_id}")
        return []
    except NoTranscriptFound:
        print(f"Error: No transcript found in languages {languages} for video ID: {video_id}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

if __name__ == "__main__":
    # Test on a Sardonicast Video ID (e.g., 'Sardonicast #1' or sample video ID)
    # Extract video_id from URL: https://www.youtube.com/watch?v=EXAMPLE_ID
    TEST_VIDEO_ID = "9OZAYwBtTHo"  # Replace with a real Sardonicast YouTube Video ID
    BAD_TEST_VIDEO_ID = "invalid_id_123"
    
    print(f"Fetching transcript for video ID: {TEST_VIDEO_ID}...")
    transcript_data = fetch_youtube_transcript(TEST_VIDEO_ID)
    metadata = fetch_youtube_metadata(TEST_VIDEO_ID)
    print(f"Metadata: {metadata}")
    
    if transcript_data:
        print(f"\nSuccessfully fetched {len(transcript_data)} transcript snippets!")
        print("\n--- First 5 Raw Snippets ---")
        for line in transcript_data[:5]:
            print(f"[{line['timestamp']}] ({line['start_seconds']}s): {line['text']}")
            print(f"Link: {line['youtube_link']}\n")