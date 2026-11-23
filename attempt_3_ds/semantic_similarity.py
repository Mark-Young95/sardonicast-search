import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device Count:", torch.cuda.device_count())
    print("Device Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is NOT available. Check PyTorch installation build.")

def compute_coherence_scores(sentences, window_size=4):
    """
    Computes semantic similarity drop across a sliding window of sentences.
    """
    model = SentenceTransformer('BAAI/bge-m3')
    
    # Generate embeddings for each sentence
    texts = [s['text'] for s in sentences]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    similarity_scores = []
    
    # Compute similarity across sliding windows
    for i in range(window_size, len(sentences) - window_size):
        left_window = np.mean(embeddings[i - window_size:i], axis=0).reshape(1, -1)
        right_window = np.mean(embeddings[i:i + window_size], axis=0).reshape(1, -1)
        
        sim = cosine_similarity(left_window, right_window)[0][0]
        similarity_scores.append((i, sim))
        
    return similarity_scores

def extract_peaks(similarity_scores, threshold=0.35):
    """
    Identifies topic boundary indices based on significant drops in similarity.
    """
    boundary_indices = []
    scores_only = [s[1] for s in similarity_scores]
    
    for idx in range(1, len(scores_only) - 1):
        # Local minimum check (trough in similarity = peak in topic shift)
        if scores_only[idx] < scores_only[idx - 1] and scores_only[idx] < scores_only[idx + 1]:
            depth = (scores_only[idx - 1] - scores_only[idx]) + (scores_only[idx + 1] - scores_only[idx])
            if depth > threshold:
                sentence_idx = similarity_scores[idx][0]
                boundary_indices.append(sentence_idx)
                
    return boundary_indices