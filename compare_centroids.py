# load from .npz

# import numpy as np

# data = np.load("speaker_centroids.npz")
# print("Saved speakers:", data.files)

# adam_centroid = data["adam"]      # Shape: (256,)
# alex_centroid = data["alex"]

# # Cosine similarity between two unit-normalized vectors is simply the dot product
# sim = np.dot(adam_centroid, alex_centroid)
# print(f"Similarity: {sim:.4f}")





# load from .json

# import json
# import numpy as np

# with open("speaker_centroids.json", "r") as f:
#     data = json.load(f)

# # Reconstruct numpy array
# centroids = {spk: np.array(vec, dtype=np.float32) for spk, vec in data.items()}