from typing import List, Dict, Any

def create_sliding_window_chunks(
    transcript_snippets: List[Dict[str, Any]], 
    window_duration_sec: int = 60, 
    overlap_duration_sec: int = 20
) -> List[Dict[str, Any]]:
    """
    Groups raw transcript entries into overlapping time-based chunks.
    
    :param transcript_snippets: List of dicts from transcript_fetcher.py
    :param window_duration_sec: Target duration of each chunk in seconds (default: 60s)
    :param overlap_duration_sec: Overlap between consecutive chunks in seconds (default: 20s)
    :return: List of structured chunk dictionaries
    """
    if not transcript_snippets:
        return []

    chunks = []
    step_size = window_duration_sec - overlap_duration_sec
    
    # Determine the total video length from the last transcript snippet
    last_snippet = transcript_snippets[-1]
    total_duration = last_snippet['start_seconds'] + last_snippet['duration']
    
    current_start = 0.0
    chunk_index = 0

    while current_start < total_duration:
        current_end = current_start + window_duration_sec
        
        # Collect all text snippets that fall within this time window
        matching_snippets = [
            s for s in transcript_snippets 
            if s['start_seconds'] >= current_start and s['start_seconds'] < current_end
        ]
        
        if matching_snippets:
            # Join individual lines into a coherent paragraph
            chunk_text = " ".join([s['text'] for s in matching_snippets])
            
            first_snippet = matching_snippets[0]
            start_time = first_snippet['start_seconds']
            start_timestamp = first_snippet['timestamp']
            youtube_link = first_snippet['youtube_link']
            
            chunks.append({
                "chunk_id": f"chunk_{chunk_index}",
                "start_seconds": start_time,
                "end_seconds": round(matching_snippets[-1]['start_seconds'] + matching_snippets[-1]['duration'], 2),
                "timestamp": start_timestamp,
                "text": chunk_text,
                "youtube_link": youtube_link
            })
            chunk_index += 1

        # Advance window by step size (window - overlap)
        current_start += step_size

    return chunks


if __name__ == "__main__":
    # Import our transcript fetcher from the same package
    from transcript_fetcher import fetch_youtube_transcript

    TEST_VIDEO_ID = "9OZAYwBtTHo"  # Replace with a real Sardonicast YouTube Video ID
    BAD_TEST_VIDEO_ID = "invalid_id_123"
    print(f"Fetching raw transcript for video: {TEST_VIDEO_ID}...")
    raw_snippets = fetch_youtube_transcript(TEST_VIDEO_ID)

    if raw_snippets:
        print(f"Raw Snippets: {len(raw_snippets)}")
        chunks = create_sliding_window_chunks(raw_snippets, window_duration_sec=60, overlap_duration_sec=20)
        print(f"Created {len(chunks)} overlapping chunks!\n")

        print("--- Sample Chunk 0 ---")
        print(f"Time: [{chunks[0]['timestamp']}] ({chunks[0]['start_seconds']}s - {chunks[0]['end_seconds']}s)")
        print(f"Link: {chunks[0]['youtube_link']}")
        print(f"Text Preview: {chunks[0]['text'][:200]}...\n")

        if len(chunks) > 1:
            print("--- Sample Chunk 1 (Notice the overlapping text from Chunk 0) ---")
            print(f"Time: [{chunks[1]['timestamp']}] ({chunks[1]['start_seconds']}s - {chunks[1]['end_seconds']}s)")
            print(f"Text Preview: {chunks[1]['text'][:200]}...")
        if len(chunks) > 2:
            for num in range(2, len(chunks)):
                print(f"--- Sample Chunk {num} (Notice the overlapping text from Chunk 0) ---")
                print(f"Time: [{chunks[num]['timestamp']}] ({chunks[num]['start_seconds']}s - {chunks[num]['end_seconds']}s)")
                print(f"Text Preview: {chunks[num]['text'][:50]}...")