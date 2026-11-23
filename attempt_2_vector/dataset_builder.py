import re
import json
import requests
import os
import time
import random
import sys
import itertools
import http.cookiejar
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, IpBlocked
#from youtube_transcript_api.proxies import WebshareProxyConfig
from youtube_transcript_api.proxies import GenericProxyConfig

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# Points to <project_root>/transcript_cache/ (one level above attempt_2_vector/)
CACHE_DIR = Path(__file__).resolve().parent.parent / "transcript_cache"
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

def load_proxy_pool(file_path: str = "proxies.txt") -> List[str]:
    """Loads proxy lines from proxies.txt and formats them into http://user:pass@ip:port/ URLs."""
    # Look for proxies.txt in current working directory or relative to script
    target_path = Path(file_path)
    if not target_path.exists():
        target_path = Path(__file__).resolve().parent / file_path

    if not target_path.exists():
        print(f"⚠️ Warning: Proxy file '{file_path}' not found. Network requests will run directly without proxies.")
        return []

    formatted_proxies = []
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) == 4:
                ip, port, user, pwd = parts
                url = f"http://{user}:{pwd}@{ip}:{port}/"
                formatted_proxies.append(url)
            elif len(parts) == 2:
                ip, port = parts
                url = f"http://{ip}:{port}/"
                formatted_proxies.append(url)

    print(f"✅ Loaded {len(formatted_proxies)} proxies from {target_path.name}")
    return formatted_proxies

# Initialize infinite proxy rotator
PROXY_LIST = load_proxy_pool("proxies.txt")
PROXY_CYCLE = itertools.cycle(PROXY_LIST) if PROXY_LIST else None


def get_next_proxy_config() -> GenericProxyConfig | None:
    """Retrieves the next proxy from the pool and returns a GenericProxyConfig object."""
    if not PROXY_CYCLE:
        return None

    proxy_url = next(PROXY_CYCLE)
    return GenericProxyConfig(
        http_url=proxy_url,
        https_url=proxy_url
    )

def fetch_metadata_batch_official(video_ids: list[str]) -> Dict[str, dict]:
    """
    Fetches title, publish date, and description for up to 50 video IDs
    in a SINGLE official API call (Cost: 1 quota unit).
    Returns a dictionary mapping video_id -> metadata dict.
    """
    if not API_KEY:
        print("⚠️ Warning: YOUTUBE_API_KEY not found in environment.")
        return {}

    results = {}
    
    # Process in chunks of 50 (YouTube Data API limit per request)
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        ids_str = ",".join(chunk)
        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={ids_str}&key={API_KEY}"
        
        try:
            response = requests.get(url).json()
            for item in response.get("items", []):
                snippet = item["snippet"]
                vid_id = item["id"]
                results[vid_id] = {
                    "video_id": vid_id,
                    "title": snippet.get("title", ""),
                    "upload_date": snippet.get("publishedAt", "").split("T")[0],
                    "description": snippet.get("description", "")
                }
        except Exception as e:
            print(f"⚠️ Error in batch API metadata fetch for batch starting at index {i}: {e}")

    return results

def fetch_youtube_metadata(video_id: str) -> Dict[str, Any]:
    """Fetches video title, upload date, and full description directly from YouTube metadata."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    metadata = {"video_id": video_id, "title": "", "upload_date": "", "description": ""}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
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

    except Exception as e:
        print(f"Warning: Error fetching metadata for {video_id}: {e}")
        
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


def detect_transcript_gaps(snippets: List[Dict[str, Any]], min_gap_seconds: float = 2.0) -> List[Dict[str, Any]]:
    """Identifies silent gaps between transcr snippets that signal topic shifts."""
    gap_anchors = []
    for i in range(len(snippets) - 1):
        curr_end = snippets[i]["start_seconds"] + snippets[i].get("duration", 0.0)
        next_start = snippets[i + 1]["start_seconds"]
        gap = next_start - curr_end
        if gap >= min_gap_seconds:
            gap_anchors.append({
                "start_seconds": next_start,
                "gap_duration": round(gap, 2)
            })
    return gap_anchors

def label_transcript_chunks(
    transcr_snippets: List[Dict[str, Any]], 
    ground_truth_chapters: List[Dict[str, Any]],
    chunk_window_sec: float = 30.0
) -> List[Dict[str, Any]]:
    """Groups transcr into N-second windows and tags boundary windows matching ground truth."""
    if not transcr_snippets or not ground_truth_chapters:
        return []

    total_duration = transcr_snippets[-1]["start_seconds"] + transcr_snippets[-1].get("duration", 5.0)
    # Avoid modifying the original ground_truth_chapters in place
    chapters = [dict(ch) for ch in ground_truth_chapters]
    chapters[-1]["end_seconds"] = total_duration

    dataset_samples = []
    current_time = 0.0
    
    while current_time < total_duration:
        window_end = current_time + chunk_window_sec
        
        window_snippets = [
            s["text"] for s in transcr_snippets 
            if current_time <= s["start_seconds"] < window_end
        ]
        window_text = " ".join(window_snippets).strip()
        
        matching_chapter = "OTHER / INTRO"
        is_boundary_chunk = False
        
        for ch in chapters:
            if ch["start_seconds"] <= current_time < ch.get("end_seconds", total_duration):
                matching_chapter = ch["title"]
                
            if current_time <= ch["start_seconds"] < window_end:
                is_boundary_chunk = True

        if window_text:
            dataset_samples.append({
                "window_start_sec": current_time,
                "window_end_sec": window_end,
                "total_episode_duration": total_duration,
                "text": window_text,
                "chapter_label": matching_chapter,
                "is_transition_boundary": int(is_boundary_chunk)
            })
            
        current_time += chunk_window_sec

    return dataset_samples








# Example usage check
if __name__ == "__main__":
    test_video_ids = ["krAxPEEpLTc","sqyyO9kIQV8","yfGkuJBY2SE","9nkBXAyaJUQ","Dj4Mg9FeE-M"] #["cKIXdtntMac", "YO9wkuvTsJE", "6LHCYUmb-xc"] #["9OZAYwBtTHo", "nmKCUFGq0hA", "WCA8Jt_qrG4"] #, "FXv2XUshrNE", "5rL7bBwOuio", "Ojz0sskxFBo", "kdlzHIbwQfs", "0OYydixtVA8", "AExOwJa58no", 
    # "egz-Hx4iLDw", "xVFrlN1aQIE", "7qxWAC4A0yk", "6jGjbEm99CI", "AnNI-4ihV9c", "HPVFbp_CBFA", "aQBFixFqv6M", "leOu1G1faBE", "7nlPtTUtHrw", "8nP1uXKmwSY", "6QxmbwpFRF4", 
    # "nMwkk5jozAw", "YO9wkuvTsJE", "9MTD8kOpGhk", "xtSnzvJFyYU", "4IP8hfMPzas", "0Seji1U6-4g", "V2YtTeafjvE", "Z1rsMoca8Jo", "50a38cwyvdM", "bsK6U6Lpbic", 
    # "9JIveRHOVVQ", "aGBDLj4LWEY", "kjtJIsUpoSQ", "9TVhA0iKI-8", "JzS6NDPTuuM", "7bSKhAzFusU", "ut4ZvbEnzuU", "R6oPFvxIoJk", "MwdXyumoXiE", "Dj4Mg9FeE-M", 
    # "9nkBXAyaJUQ", "yfGkuJBY2SE", "sqyyO9kIQV8", "krAxPEEpLTc", "1V1yTmH44XI", "cKIXdtntMac", "lRCDT1iMdRg", "s5_aI6T9OIQ", "cSkEMK0Z3Eo", "URfbzdgxFmE"]

    # 1. Batch-fetch official metadata first (falls back to scraping per video if YOUTUBE_API_KEY is unset)
    batch_meta = fetch_metadata_batch_official(test_video_ids)

    for vid_id in test_video_ids:
        print(f"\nProcessing video ID: {vid_id}...")
        cache_file = CACHE_DIR / f"{vid_id}.json"

        # 2. Get metadata (from batch API or fallback scraper)
        meta = batch_meta.get(vid_id) or fetch_youtube_metadata(vid_id)
        print(f"  ├─ Title: '{meta['title']}' ({meta['upload_date']})")

        parsed = parse_description_timestamps(meta["description"])
        print(f"  ├─ Parsed {len(parsed)} Ground Truth Chapters from Description")

        # 3. Fetch transcript snippets (note: fetch_and_clean_transcript saves a raw list,
        #    so we wrap it with "meta" + "snippets" right after)
        snippets = fetch_and_clean_transcript(vid_id)
        if isinstance(snippets, dict) and "snippets" in snippets:
            snippets = snippets["snippets"]

        payload = {
            "meta": meta,
            "snippets": snippets,
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"  └─ ✅ Saved {len(snippets)} snippets + metadata to: {cache_file}")