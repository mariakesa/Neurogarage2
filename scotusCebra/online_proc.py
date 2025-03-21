import numpy as np
import pandas as pd
import umap
from sklearn.cluster import SpectralClustering
import json

# Load embeddings
output_path = "/home/maria/Downloads/scotus_embeddings.npy"
embeddings = np.load(output_path)[:5000]  # Reduce dataset size for faster processing

# Apply UMAP to reduce to 3D
umap_model = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
umap_result_ = umap_model.fit_transform(embeddings)

# Perform Spectral Clustering
num_clusters = 5
clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
cluster_labels = clustering.fit_predict(umap_result_)

# Load text summaries
texts = pd.read_csv("/home/maria/Neurogarage2/scotusCebra/scotus_with_summaries_ordered_full.csv", header=0)
df_cases = pd.read_csv("/home/maria/Neurogarage2/scotusCebra/scotus_with_summaries_ordered_full.csv", header=0).loc[:4999]
texts = df_cases["summary"]
case_names = df_cases["case_name"]  # Load case names

# Format text into a box shape
def format_text(text, line_length=50, max_lines=20):
    if isinstance(text, float) or text is None:  
        return "No summary available"

    words = text.split()
    lines, current_line = [], ""

    for word in words:
        if len(current_line) + len(word) + 1 <= line_length:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
        if len(lines) >= max_lines:
            break

    if current_line and len(lines) < max_lines:
        lines.append(current_line)

    return "<br>".join(lines)

# Apply text formatting
texts = [format_text(text) for text in texts]

# Save UMAP results, cluster labels, and text summaries to JSON
data = []
for i in range(len(umap_result_)):
    data.append({
        "x": float(umap_result_[i, 0]),
        "y": float(umap_result_[i, 1]),
        "z": float(umap_result_[i, 2]),
        "cluster": int(cluster_labels[i]),
        "case_name": case_names[i],  # Include case name
        "text_summary": texts[i]
    })

with open("scotus_umap_data.json", "w") as f:
    json.dump(data, f)

print("✅ Data saved to scotus_umap_data.json")
