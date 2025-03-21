import numpy as np
import pandas as pd
import umap
from sklearn.cluster import SpectralClustering
import json

# Load text summaries
texts = pd.read_csv("/home/maria/Neurogarage2/scotusCebra/scotus_with_summaries_ordered_full.csv", header=0)["summary"]
df_cases = pd.read_csv("/home/maria/Neurogarage2/scotusCebra/scotus_with_summaries_ordered_full.csv", header=0)
texts = df_cases["summary"]
case_names = df_cases["case_name"]  # Load case names

# Function to format text into a box shape
def format_text_box(text, line_length=50, max_lines=20):
    """
    Format text into a box shape by breaking it every `line_length` characters.
    Limits the number of lines to `max_lines` to avoid overly long text.
    """
    if pd.isna(text):  # Handle missing values
        return "No summary available"

    text = text.strip()  # Remove leading/trailing spaces
    words = text.split()  # Split into words
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= line_length:
            current_line += (" " if current_line else "") + word  # Append word to current line
        else:
            lines.append(current_line)  # Store the full line
            current_line = word  # Start a new line
        if len(lines) >= max_lines:  # Stop if max lines reached
            break

    if current_line and len(lines) < max_lines:
        lines.append(current_line)  # Add the last line if space allows

    return "<br>".join(lines)  # Join lines with HTML line breaks

# Apply formatting to text summaries
texts = texts.apply(lambda x: format_text_box(x, line_length=50, max_lines=20))

# Load embeddings
output_path = "/home/maria/Downloads/scotus_embeddings.npy"
embeddings = np.load(output_path)  # Ensure the same length as texts

# Apply UMAP for nonlinear dimensionality reduction to 3D
umap_model = umap.UMAP(n_components=3, n_neighbors=15, min_dist=1.0, metric='euclidean', random_state=42)
umap_result_ = umap_model.fit_transform(embeddings)

# Perform Spectral Clustering
num_clusters = 10  # Adjust as needed
clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
cluster_labels = clustering.fit_predict(umap_result_)

# Convert to JSON-friendly format
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
# Save JSON file
json_output_path = "scotus_umap_data2.json"
with open(json_output_path, "w") as f:
    json.dump(data, f)

print(f"✅ JSON data saved to {json_output_path}")
