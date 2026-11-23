import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "data" / "transcripts_cache"
OUTPUT_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load pre-trained lightweight embedding model
# 'all-MiniLM-L6-v2' is fast, free, runs locally on CPU, and yields great search accuracy
print("⚡ Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
model = SentenceTransformer('all-MiniLM-L6-v2')


def load_all_cached_chunks():
    """Loads all individual transcript snippets from local JSON cache files."""
    all_chunks = []
    
    json_files = list(CACHE_DIR.glob("*.json"))
    print(f"📂 Found {len(json_files)} transcript JSON files in cache.")

    for file_path in json_files:
        video_id = file_path.stem
        with open(file_path, "r", encoding="utf-8") as f:
            snippets = json.load(f)

        for item in snippets:
            text = item.get("text", "").strip()
            if not text:
                continue

            all_chunks.append({
                "video_id": video_id,
                "text": text,
                "start_seconds": item.get("start_seconds"),
                "timestamp": item.get("timestamp"),
                "youtube_link": item.get("youtube_link")
            })

    return all_chunks


def main():
    # 2. Gather text snippets
    chunks = load_all_cached_chunks()
    if not chunks:
        print("❌ No transcript chunks found. Make sure data/transcripts_cache/ contains JSON files.")
        return

    print(f"📊 Total transcript snippets loaded: {len(chunks)}")

    # Extract text strings for embedding input
    texts_to_embed = [item["text"] for item in chunks]

    # 3. Generate Embeddings (Inference)
    print("🧠 Generating vector embeddings (this takes ~10-30 seconds on CPU)...")
    embeddings = model.encode(
        texts_to_embed, 
        batch_size=64, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )

    print(f"✅ Generated matrix shape: {embeddings.shape} (Chunks x Vector Dimensions)")

    # 4. Save metadata and vectors to disk
    metadata_file = OUTPUT_DIR / "processed_chunks.json"
    vectors_file = OUTPUT_DIR / "embeddings.npy"

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    np.save(vectors_file, embeddings)

    print(f"🎉 Saved chunk metadata to: {metadata_file}")
    print(f"🎉 Saved dense embeddings to: {vectors_file}")


if __name__ == "__main__":
    main()