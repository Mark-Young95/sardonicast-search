import json
from coherence import CoherenceAnalyzer

if __name__ == "__main__":
    test_video_ids = ['_bvT65wMeoE', 'tVG-PoH-n80']

    # Initialize analyzer once outside the loop to avoid reloading model weights
    analyzer = CoherenceAnalyzer(model_name='BAAI/bge-m3')

    for vid_id in test_video_ids:
        print(f"\n==========================================")
        print(f"Analyzing video ID: {vid_id}")
        print(f"==========================================")
        
        input_file = f"data/transcripts_cache/{vid_id}.json"
        
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_transcript = json.load(f)

        # Updated unpacking to handle all 4 return values from analyze_episode
        raw_sims, scaled_sims, sentences, embeddings = analyzer.analyze_episode(raw_transcript)
        
        # Cull candidates down to primary chapters
        primary_chapters = analyzer.cull_candidates(
            scaled_sims=scaled_sims,
            sentences=sentences,
            embeddings=embeddings,
            min_chapter_duration_sec=300.0  # Enforce minimum 5 minutes per chapter
        )
        
        print(f"\n--- Primary Chapters Detected ({len(primary_chapters)}) ---")
        for ch in primary_chapters:
            time_str = ch.get("timestamp_formatted", "00:00")
            snippet = ch.get("text", "")[:70]
            ch_type = ch.get("type", "Boundary")
            print(f"[{time_str}] ({ch_type}) -> {snippet}...")

        # Export Plotly HTML graph using the scaled similarities
        analyzer.export_interactive_plot(
            similarities=scaled_sims, 
            eval_sentences=sentences, 
            boundaries=primary_chapters, 
            output_html=f"coherence_plot_{vid_id}.html"
        )