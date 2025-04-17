import numpy as np
import plotly.express as px
import pandas as pd
import umap
from sklearn.cluster import SpectralClustering

# Load texts correctly (Exclude header to avoid an extra row)
texts = pd.read_csv("/home/maria/Neurogarage2/scotusCebra/scotus_with_summaries_ordered_full.csv", header=0)['summary']#.loc[:4999, "summary"]

# Function to format text into a box shape by breaking lines at fixed intervals
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
output_path = 'scotus_embeddings_pca750.npy'
embeddings = np.load(output_path)#[:5000]  # Ensure the same length as texts

# Apply UMAP for nonlinear dimensionality reduction to 3D
umap_model = umap.UMAP(n_components=3, n_neighbors=40, min_dist=0.0, metric='cosine', random_state=42)
umap_result_ = umap_model.fit_transform(embeddings)

# Perform Spectral Clustering
#num_clusters = 10  # Adjust as needed
#clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
#cluster_labels = clustering.fit_predict(umap_result_)

cluster_labels=np.load('vqvae_clusters.npy')

# Convert to DataFrame
df = pd.DataFrame(umap_result_, columns=["UMAP1", "UMAP2", "UMAP3"])
df["Cluster"] = cluster_labels.astype(str)  # Convert to string for categorical coloring
df["Text Summary"] = texts.values  # Assign formatted text

# Plot interactive 3D scatter plot with formatted text on hover
fig = px.scatter_3d(df, x="UMAP1", y="UMAP2", z="UMAP3", opacity=0.7, 
                     title="3D UMAP Visualization of SCOTUS Embeddings (Spectral Clustering)",
                     color=df["Cluster"],
                     hover_data={"Text Summary": True})  # Display formatted summaries on hover

fig.update_traces(marker=dict(size=3, line=dict(width=0.5, color="black")))
fig.show()
