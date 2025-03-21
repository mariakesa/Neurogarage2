import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import nltk
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

# Ensure NLTK sentence tokenizer is available
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
torch.set_float32_matmul_precision('high')

# Load OpenAI API Key
load_dotenv()

# Load SCOTUS dataset
data_path = "/home/maria/Downloads/archive(1)/all_opinions.csv"
df = pd.read_csv(data_path)#.loc[:100]

# Extract texts (ensure non-null)
texts = df["text"].fillna("")

# Load Legal-BERT model
#tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
#model = BertModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base").to(device)
print('BOOM!', device)
model.eval()  # Set model to evaluation mode


def embed_sentences(sentences, batch_size=16):
    """Embed a batch of sentences using BERT on GPU."""
    model.to(device)  # Ensure the model is moved to GPU

    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=8000)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move tokenizer outputs to GPU

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)

#num clusters 50, lebalbert model
def summarize_text(text, num_clusters=10):
    """Summarize long text by selecting representative sentences via clustering, preserving order."""
    sentences = sent_tokenize(text)  # Split into sentences

    # If fewer sentences than clusters, return original text
    if len(sentences) < num_clusters:
        return text  

    embeddings = embed_sentences(sentences)  # Embed all sentences

    # Remove duplicate embeddings to avoid clustering issues
    unique_embeddings, unique_indices = np.unique(embeddings, axis=0, return_index=True)
    unique_sentences = [sentences[i] for i in sorted(unique_indices)]  # Maintain order

    # Adjust num_clusters if fewer unique embeddings exist
    actual_clusters = min(num_clusters, len(unique_embeddings))
    
    # If we still have too few sentences, return the full text
    if actual_clusters < 2:
        return text

    # K-Means Clustering
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    kmeans.fit(unique_embeddings)

    # Assign sentences to clusters
    labels = kmeans.predict(unique_embeddings)

    # Select the closest sentence to the centroid in each cluster
    cluster_centers = kmeans.cluster_centers_
    selected_sentences = []
    selected_indices = []

    for cluster_id in range(actual_clusters):
        cluster_members = np.where(labels == cluster_id)[0]
        if len(cluster_members) == 0:
            continue  # Skip empty clusters (shouldn't normally happen)

        # Find the sentence closest to the cluster center
        closest_idx = cluster_members[np.argmin(
            np.linalg.norm(unique_embeddings[cluster_members] - cluster_centers[cluster_id], axis=1)
        )]

        selected_indices.append(closest_idx)

    # Sort selected indices to preserve sentence order
    selected_indices = sorted(selected_indices)
    summary_sentences = [unique_sentences[i] for i in selected_indices]

    return " ".join(summary_sentences)  # Return summary as ordered text

# Generate summaries for all cases in parallel
df["summary"] = [summarize_text(text) for text in tqdm(df["text"].fillna(""), desc="Summarizing cases")]

# Save summaries
output_path = "/home/maria/Neurogarage2/scotusCebra/scotus_with_summaries_ordered_full.csv"
df.to_csv(output_path, index=False)

print(f"Summarization complete! Saved to {output_path}")
