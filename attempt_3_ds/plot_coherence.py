import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess_transcript

def analyze_and_plot_episode(json_path: str, ground_truth_timestamps: list[float] = None):
    # 1. Load and preprocess
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    sentences = preprocess_transcript(raw_data)
    texts = [s['text'] for s in sentences]
    timestamps = [s['start_seconds'] for s in sentences]

    print(f"Loaded {len(sentences)} preprocessed sentences.")

    # 2. Encode sentences
    model = SentenceTransformer('BAAI/bge-m3')
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    # 3. Compute sliding window coherence
    window_size = 5  # Adjust window size to tune smoothing
    similarities = []
    eval_timestamps = []

    for i in range(window_size, len(sentences) - window_size):
        left_context = np.mean(embeddings[i - window_size:i], axis=0).reshape(1, -1)
        right_context = np.mean(embeddings[i:i + window_size], axis=0).reshape(1, -1)
        
        sim = cosine_similarity(left_context, right_context)[0][0]
        similarities.append(sim)
        eval_timestamps.append(timestamps[i])

    # 4. Plot results
    plt.figure(figsize=(14, 6))
    plt.plot(np.array(eval_timestamps) / 60.0, similarities, label='Semantic Similarity (Sliding Window)', color='#1f77b4', alpha=0.8)
    
    # Plot Ground Truth description lines if available
    if ground_truth_timestamps:
        for idx, gt in enumerate(ground_truth_timestamps):
            plt.axvline(x=gt / 60.0, color='red', linestyle='--', alpha=0.7, 
                        label='GT Chapter Boundary' if idx == 0 else "")

    plt.title('Semantic Coherence & Topic Boundary Valleys')
    plt.xlabel('Episode Time (Minutes)')
    plt.ylabel('Cosine Similarity')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace with path to one of your cached transcript JSON files
    analyze_and_plot_episode("data/transcripts_cache/_bvT65wMeoE.json", ground_truth_timestamps=[0.0, 3194.0, 5370.0, 7435.0])
    # 0:00 Venom: The Last Dance
# 53:14 Tokyo Godfathers
# 1:29:30 Questions
# 2:03:55 Recommendation + outro
# [0.0, 3194.0, 5370.0, 7435.0]
    #pass