import numpy as np

data = np.load("speaker_centroids.npz")
speakers = list(data.files)

print("--- Cosine Similarity Matrix ---")
header = f"{'':15}" + "".join(f"{s:>15}" for s in speakers)
print(header)

for s1 in speakers:
    row = f"{s1:15}"
    for s2 in speakers:
        sim = float(np.dot(data[s1], data[s2]))
        row += f"{sim:>15.4f}"
    print(row)