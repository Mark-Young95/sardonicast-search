import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load index and vectors
data_dir = Path("data")
with open(data_dir / "processed_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load(data_dir / "embeddings.npy")
model = SentenceTransformer('all-MiniLM-L6-v2')

def search(query_text: str, top_k: int = 3):
    # 1. Embed query into 384-dim vector
    query_vec = model.encode([query_text], convert_to_numpy=True)

    # 2. Compute Cosine Similarity against all transcript vectors
    # Normalize vectors for fast dot product
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_query = query_vec / np.linalg.norm(query_vec)

    scores = np.dot(norm_embeddings, norm_query.T).squeeze()

    # 3. Get top K indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    print(f"\n🔍 Search Query: '{query_text}'")
    print("=" * 60)
    for idx in top_indices:
        match = chunks[idx]
        score = scores[idx]
        print(f"Score: {score:.4f} | [{match['timestamp']}] ({match['video_id']})")
        print(f"Text: \"{match['text']}\"")
        print(f"Link: {match['youtube_link']}")
        print("-" * 60)

# Run test queries
search("deep discussion on the advance of AI")
search("spider-man videogame")