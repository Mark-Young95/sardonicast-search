import re
import json
import requests
import os
import time
import random
import http.cookiejar
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, IpBlocked

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

CACHE_DIR = Path(__file__).resolve().parent / "data" / "transcripts_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Path to your exported browser cookies
COOKIES_FILE = Path(__file__).resolve().parent / "cookies.txt"

def create_authenticated_session() -> requests.Session:
    """Creates a requests.Session loaded with cookies from cookies.txt if available."""
    session = requests.Session()
    if COOKIES_FILE.exists():
        cj = http.cookiejar.MozillaCookieJar(str(COOKIES_FILE))
        try:
            cj.load(ignore_discard=True, ignore_expires=True)
            session.cookies = cj
        except Exception as e:
            print(f"  ⚠️ Failed to load cookies.txt: {e}")
    else:
        print("  ⚠️ Warning: cookies.txt not found. Attempting unauthenticated request...")
    return session

def fetch_youtube_metadata(video_id: str, max_retries: int = 3) -> Dict[str, Any]:
    """Fetches video title, upload date, and full description directly from YouTube metadata with retries."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    metadata = {"video_id": video_id, "title": "", "upload_date": "", "description": ""}
    
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8')
                
                # Extract Title
                title_match = re.search(r'<meta property="og:title" content="([^"]+)">', html)
                if title_match:
                    metadata["title"] = title_match.group(1).replace(" - YouTube", "")
                    
                # Extract Upload Date
                date_match = re.search(r'<meta itemprop="uploadDate" content="([^"]+)">', html) or \
                             re.search(r'<meta itemprop="datePublished" content="([^"]+)">', html)
                if date_match:
                    metadata["upload_date"] = date_match.group(1).split("T")[0]

                # Extract Description
                desc_match = re.search(r'"shortDescription":"(.*?)","isCrawlable"', html)
                if desc_match:
                    raw_desc = desc_match.group(1)
                    metadata["description"] = json.loads(f'"{raw_desc}"')
                else:
                    og_desc_match = re.search(r'<meta property="og:description" content="([^"]*)">', html)
                    if og_desc_match:
                        metadata["description"] = og_desc_match.group(1)

            # If successful, return the metadata and exit the retry loop
            return metadata

        except Exception as e:
            print(f"  ⚠️ Warning: Error fetching metadata for {video_id} (Attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                sleep_time = random.uniform(3.0, 7.0)
                print(f"     Sleeping for {sleep_time:.1f} seconds before retrying...")
                time.sleep(sleep_time)
            else:
                print(f"  ❌ Failed to fetch metadata for {video_id} after {max_retries} attempts.")
                
    return metadata

def parse_description_timestamps(description_text: str) -> List[Dict[str, Any]]:
    """Extracts ground-truth timestamps and chapter titles from YouTube description text."""
    chapters = []
    pattern = r'(?:\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?)\s*[-–:]?\s*(.+)'
    
    for line in description_text.splitlines():
        match = re.search(pattern, line.strip())
        if match:
            time_str, title = match.group(1), match.group(2).strip()
            
            parts = [int(p) for p in time_str.split(':')]
            if len(parts) == 3:
                start_sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                start_sec = parts[0] * 60 + parts[1]
            else:
                continue
                
            chapters.append({
                "start_seconds": float(start_sec),
                "timestamp_str": time_str,
                "title": title
            })
            
    chapters = sorted(chapters, key=lambda x: x["start_seconds"])
    for i in range(len(chapters) - 1):
        chapters[i]["end_seconds"] = chapters[i + 1]["start_seconds"]
        
    return chapters

def format_timestamp(seconds: float) -> str:
    """Converts seconds into HH:MM:SS format."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def fetch_and_clean_transcript(video_id: str, max_retries: int = 3, languages: List[str] = ['en']) -> List[Dict[str, Any]]:
    """
    Fetches raw transcript snippets via youtube_transcript_api, with disk caching and retry delays.
    """
    
    cache_file = CACHE_DIR / f"{video_id}.json"

    # Load from cache if it exists
    if cache_file.exists():
        print(f"pulling from cache: {video_id}.json")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Verify cookies file exists
    cookies_path_str = str(COOKIES_FILE) if COOKIES_FILE.exists() else None
    if not cookies_path_str:
        print("  ⚠️ Warning: cookies.txt not found. Attempting unauthenticated request...")

    # Fetch from YouTube with throttling & backoff
    for attempt in range(1, max_retries + 1):
        print(f"cache not found: {video_id}")
        #proxy_config = get_next_proxy_config()

        try:
            # Random delay (5-7 seconds) before outbound call
            time.sleep(random.uniform(5.0, 7.0))

            #ytt_api = YouTubeTranscriptApi() #YouTubeTranscriptApi(proxy_config=proxy_config) if proxy_config else YouTubeTranscriptApi()

            session = create_authenticated_session()
            api = YouTubeTranscriptApi(http_client=session)
            
            transcript = api.fetch(video_id, languages=languages)

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
            # Save to cache
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(formatted_transcript, f, indent=2)

            return formatted_transcript

        except (TranscriptsDisabled, NoTranscriptFound):
            print(f"  ⚠️ Transcript unavailable for {video_id}.")
            return []
        except (IpBlocked, Exception) as e:
            print(f"  ⚠️ Attempt {attempt}/{max_retries} failed for {video_id}: {e}")
            if attempt == max_retries:
                print(f"  ❌ Failed all {max_retries} attempts for {video_id}.")
                return []
    return []

# Example usage check
if __name__ == "__main__":
    test_video_ids = ["9OZAYwBtTHo", "nmKCUFGq0hA", "WCA8Jt_qrG4"] #, "FXv2XUshrNE", "5rL7bBwOuio", "Ojz0sskxFBo", "kdlzHIbwQfs", "0OYydixtVA8", "AExOwJa58no", 
    # "egz-Hx4iLDw", "xVFrlN1aQIE", "7qxWAC4A0yk", "6jGjbEm99CI", "AnNI-4ihV9c", "HPVFbp_CBFA", "aQBFixFqv6M", "leOu1G1faBE", "7nlPtTUtHrw", "8nP1uXKmwSY", "6QxmbwpFRF4", 
    # "nMwkk5jozAw", "YO9wkuvTsJE", "9MTD8kOpGhk", "xtSnzvJFyYU", "4IP8hfMPzas", "0Seji1U6-4g", "V2YtTeafjvE", "Z1rsMoca8Jo", "50a38cwyvdM", "bsK6U6Lpbic", 
    # "9JIveRHOVVQ", "aGBDLj4LWEY", "kjtJIsUpoSQ", "9TVhA0iKI-8", "JzS6NDPTuuM", "7bSKhAzFusU", "ut4ZvbEnzuU", "R6oPFvxIoJk", "MwdXyumoXiE", "Dj4Mg9FeE-M", 
    # "9nkBXAyaJUQ", "yfGkuJBY2SE", "sqyyO9kIQV8", "krAxPEEpLTc", "1V1yTmH44XI", "cKIXdtntMac", "lRCDT1iMdRg", "s5_aI6T9OIQ", "cSkEMK0Z3Eo", "URfbzdgxFmE"]

    for vid_id in test_video_ids:
        TEST_VIDEO_ID = vid_id
        print(f"Fetching metadata for video ID: {TEST_VIDEO_ID}...")
        # Extract Title and Upload Date
        meta = fetch_youtube_metadata(TEST_VIDEO_ID)
        print(f"Fetched Metadata -> Title: '{meta['title']}', Date: '{meta['upload_date']}'")
        print(f"Fetched Metadata -> Description: '{meta['description']}'")

        parsed = parse_description_timestamps(meta['description'])
        print(f"Parsed {len(parsed)} Ground Truth Chapters from Description:")
        for ch in parsed:
            print(f"  [{ch['timestamp_str']}] {ch['title']} (Ends: {ch.get('end_seconds', 'N/A')}s)")