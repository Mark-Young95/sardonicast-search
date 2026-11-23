import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Load lightweight SentenceTransformer model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_window_features(
    dataset_samples: List[Dict[str, Any]], 
    gap_anchors: List[Dict[str, Any]],
    video_title: str,
    upload_date: str
) -> List[Dict[str, Any]]:
    """
    Computes embedding shifts, metadata cross-references, and structural cues for every window.
    """
    texts = [sample["text"] for sample in dataset_samples]
    embeddings = embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    
    # Pre-calculate embedding for the YouTube Video Title
    title_embedding = embed_model.encode([video_title], show_progress_bar=False, convert_to_numpy=True)[0] if video_title else None

    qa_triggers = ["question", "questions", "head over to", "suggestion thread", "community"]
    outro_triggers = ["recommendation", "next episode", "that does it", "wrap up", "patreon"]
    
    features = []
    
    for i in range(len(dataset_samples)):
        sample = dataset_samples[i]
        text_lower = sample["text"].lower()
        curr_emb = embeddings[i]
        
        # 1. Semantic Embedding Similarity Shift with PREVIOUS window
        if i > 0:
            sim_prev = float(np.dot(embeddings[i - 1], curr_emb) / (np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(curr_emb)))
        else:
            sim_prev = 1.0

        # 2. Semantic Embedding Similarity Shift with NEXT window
        if i < len(dataset_samples) - 1:
            sim_next = float(np.dot(curr_emb, embeddings[i + 1]) / (np.linalg.norm(curr_emb) * np.linalg.norm(embeddings[i + 1])))
        else:
            sim_next = 1.0

        # 3. Similarity between current window text and YouTube Video Title
        if title_embedding is not None:
            title_sim = float(np.dot(curr_emb, title_embedding) / (np.linalg.norm(curr_emb) * np.linalg.norm(title_embedding)))
        else:
            title_sim = 0.0

        # 4. Check for Audio/Silence Gap
        has_silence_gap = any(
            sample["window_start_sec"] <= g["start_seconds"] < sample["window_end_sec"]
            for g in gap_anchors
        )

        # 5. Temporal Position Feature (relative position in the video: 0.0 to 1.0)
        relative_pos = sample["window_start_sec"] / sample["total_episode_duration"] if sample.get("total_episode_duration") else 0.0

        features.append({
            "window_start_sec": sample["window_start_sec"],
            "window_end_sec": sample["window_end_sec"],
            "relative_position": relative_pos,
            "sim_with_prev": sim_prev,
            "sim_with_next": sim_next,
            "sim_drop": (1.0 - sim_prev) + (1.0 - sim_next),
            "title_similarity": title_sim,
            "qa_trigger_count": sum(1 for kw in qa_triggers if kw in text_lower),
            "outro_trigger_count": sum(1 for kw in outro_triggers if kw in text_lower),
            "has_silence_gap": int(has_silence_gap),
            "video_title": video_title,
            "upload_date": upload_date,
            "chapter_label": sample["chapter_label"],
            "is_transition_boundary": sample["is_transition_boundary"]
        })
        
    return features