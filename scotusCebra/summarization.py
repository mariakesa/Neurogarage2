import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import time
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import openai
import nltk

# Ensure NLTK sentence tokenizer is available
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# Load OpenAI API Key
load_dotenv()
openai.api_key = os.getenv("OPENAIKEY")

# Load SCOTUS dataset
data_path = "/home/maria/Downloads/archive(1)/all_opinions.csv"
df = pd.read_csv(data_path)

# Extract texts (ensure non-null)
texts = df["text"].fillna("")

# Load Legal-BERT model
tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = BertModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
model.eval()  # Set model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def embed_sentences(sentences, batch_size=16):
    """Embed a batch of sentences using Legal-BERT on GPU."""
    embeddings = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)

def summarize_text(text, num_clusters=50):
    """Summarize long text by selecting representative sentences via clustering."""
    sentences = sent_tokenize(text)  # Split into sentences

    # If fewer sentences than clusters, return original text
    if len(sentences) < num_clusters:
        return text  

    embeddings = embed_sentences(sentences)  # Batch process all sentences

    # Remove duplicate embeddings to avoid clustering issues
    unique_embeddings, unique_indices = np.unique(embeddings, axis=0, return_index=True)
    unique_sentences = [sentences[i] for i in unique_indices]

    # Adjust num_clusters if fewer unique embeddings exist
    actual_clusters = min(num_clusters, len(unique_embeddings))
    
    # If we still have too few sentences, return the full text
    if actual_clusters < 2:
        return text

    # K-Means Clustering
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    kmeans.fit(unique_embeddings)

    # Select closest sentences to cluster centroids
    cluster_indices = np.argmin(np.linalg.norm(unique_embeddings[:, np.newaxis] - kmeans.cluster_centers_, axis=2), axis=0)
    summary_sentences = [unique_sentences[idx] for idx in cluster_indices]

    return " ".join(summary_sentences)  # Return as a summary text

# Generate summaries for all cases in parallel
df["summary"] = [summarize_text(text) for text in tqdm(df["text"].fillna(""), desc="Summarizing cases")]

# Save summaries
output_path = "/home/maria/Neurogarage2/scotusCebra/scotus_with_summaries.csv"
df.to_csv(output_path, index=False)

print(f"Summarization complete! Saved to {output_path}")
