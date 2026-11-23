import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks
import plotly.graph_objects as go

from preprocess import preprocess_transcript

def format_seconds(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}" if hrs > 0 else f"{mins:02d}:{secs:02d}"

# Keyphrase anchors for structural transitions
QA_KEYPHRASES = ["sardonicast community", "suggestion thread", "head over to", "questions from", "question time"]
RECOMMENDATION_KEYPHRASES = ["recommendation for next", "that does it for questions", "thanks for questions", "next episode", "time for recommendation", "my recommendation is", "next episode's recommendation"]

class CoherenceAnalyzer:
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        self.model = SentenceTransformer(model_name)

    def analyze_episode(
        self, 
        raw_transcript: List[Dict[str, Any]], 
        block_size: int = 6
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
        sentences = preprocess_transcript(raw_transcript)
        texts = [s['text'] for s in sentences]
        
        print(f"Encoding {len(sentences)} sentence units...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=64)

        raw_similarities = []
        total_sents = len(sentences)

        # Compute similarity using clamped edges (prevents edge drop-offs)
        for i in range(total_sents):
            left_start = max(0, i - block_size)
            left_end = i
            
            right_start = i
            right_end = min(total_sents, i + block_size)

            if left_start == left_end:
                left_ctx = embeddings[0].reshape(1, -1)
            else:
                left_ctx = np.mean(embeddings[left_start:left_end], axis=0).reshape(1, -1)

            if right_start == right_end:
                right_ctx = embeddings[-1].reshape(1, -1)
            else:
                right_ctx = np.mean(embeddings[right_start:right_end], axis=0).reshape(1, -1)

            sim = cosine_similarity(left_ctx, right_ctx)[0][0]
            raw_similarities.append(sim)

        sims_arr = np.array(raw_similarities)

        # Contrast Scaling: Expand range between min and max to boost vertical variance
        min_s, max_s = np.min(sims_arr), np.max(sims_arr)
        scaled_sims = (sims_arr - min_s) / (max_s - min_s + 1e-8) if max_s > min_s else sims_arr

        return sims_arr, scaled_sims, sentences, embeddings

    def cull_candidates(
        self, 
        scaled_sims: np.ndarray, 
        sentences: List[Dict[str, Any]],
        embeddings: np.ndarray,
        min_chapter_duration_sec: float = 300.0 # Minimum 5 minutes per chapter
    ) -> List[Dict[str, Any]]:
        dissimilarity = 1.0 - scaled_sims
        
        # 1. Find peaks in dissimilarity (valleys in similarity)
        peaks, properties = find_peaks(dissimilarity, distance=15, prominence=0.10)

        raw_candidates = []
        for idx, peak_i in enumerate(peaks):
            sent = sentences[peak_i]
            raw_candidates.append({
                "index": int(peak_i),
                "timestamp_seconds": float(sent["start_seconds"]),
                "timestamp_formatted": format_seconds(sent["start_seconds"]),
                "text": sent["text"],
                "score": float(dissimilarity[peak_i])
            })

        # 2. Check for Structural Keyphrases (Q&A / Recommendation)
        phrase_anchors = []
        for idx, sent in enumerate(sentences):
            text_lower = sent["text"].lower()
            if any(k in text_lower for k in QA_KEYPHRASES) or any(k in text_lower for k in RECOMMENDATION_KEYPHRASES):
                phrase_anchors.append({
                    "index": idx,
                    "timestamp_seconds": float(sent["start_seconds"]),
                    "timestamp_formatted": format_seconds(sent["start_seconds"]),
                    "text": sent["text"],
                    "score": 1.0 # High priority anchor
                })

        # Combine candidates and anchors
        all_candidates = sorted(raw_candidates + phrase_anchors, key=lambda x: x["timestamp_seconds"])

        # 3. Time-based merging & Long-context validation
        final_chapters = []
        if sentences:
            # Force Chapter 1 at the very start
            final_chapters.append({
                "timestamp_seconds": sentences[0]["start_seconds"],
                "timestamp_formatted": format_seconds(sentences[0]["start_seconds"]),
                "text": sentences[0]["text"],
                "type": "Start"
            })

        last_added_time = 0.0
        for cand in all_candidates:
            c_time = cand["timestamp_seconds"]
            if (c_time - last_added_time) < min_chapter_duration_sec:
                continue # Skip candidates that are too close to the previous chapter boundary

            # Long-context verification (Compare 2 minutes before vs 2 minutes after)
            c_idx = cand["index"]
            window = 12
            pre_start, pre_end = max(0, c_idx - window), c_idx
            post_start, post_end = c_idx, min(len(embeddings), c_idx + window)

            if pre_start < pre_end and post_start < post_end:
                vec_pre = np.mean(embeddings[pre_start:pre_end], axis=0).reshape(1, -1)
                vec_post = np.mean(embeddings[post_start:post_end], axis=0).reshape(1, -1)
                long_sim = cosine_similarity(vec_pre, vec_post)[0][0]

                # Keep candidate if structural anchor or long-context shift is verified
                if cand.get("score") == 1.0 or long_sim < 0.70:
                    final_chapters.append(cand)
                    last_added_time = c_time

        return final_chapters

    def export_interactive_plot(
        self, 
        similarities: np.ndarray, 
        eval_sentences: List[Dict[str, Any]], 
        boundaries: List[Dict[str, Any]],
        output_html: str = "coherence_plot.html"
    ):
        timestamps = [s["start_seconds"] for s in eval_sentences]
        time_formatted = [format_seconds(s["start_seconds"]) for s in eval_sentences]
        
        # Explicit hover formatting for Plotly
        hover_texts = [
            f"<b>Time:</b> {time_formatted[i]} ({s['start_seconds']:.1f}s)<br>"
            f"<b>Text:</b> {s['text']}<br>"
            f"<b>Similarity:</b> {similarities[i]:.4f}"
            for i, s in enumerate(eval_sentences)
        ]

        fig = go.Figure()

        # Cosine similarity line
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=similarities, 
            mode='lines', 
            name='Semantic Coherence',
            line=dict(color='#1f77b4', width=2), 
            text=hover_texts,
            hoverinfo='text'
        ))

        # Render detected boundaries as vertical green dashed lines
        for b in boundaries:
            fig.add_vline(
                x=b["timestamp_seconds"], 
                line_width=2, 
                line_dash="dash", 
                line_color="green",
                annotation_text=f"Chapter Split: {b['timestamp_formatted']}", 
                annotation_position="top left"
            )

        fig.update_layout(
            title="Sardonicast Semantic Coherence Graph",
            xaxis_title="Episode Timeline (Seconds)",
            yaxis_title="Cosine Similarity",
            hovermode="x unified",
            template="plotly_white",
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            )
        )

        fig.write_html(output_html)
        print(f"\nSaved interactive graph to {output_html}.")