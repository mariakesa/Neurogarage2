import os
import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Load OpenAI API Key
load_dotenv()
openai.api_key = os.getenv("OPENAIKEY")

# Load SCOTUS dataset
data_path = "/home/maria/Downloads/archive(1)/all_opinions.csv"
df = pd.read_csv(data_path)

# Extract texts (ensure non-null)
texts = df["text"].fillna("")

# OpenAI embedding model
MODEL_NAME = "text-embedding-3-large"

# Function to get OpenAI embeddings
def get_embedding(text, model=MODEL_NAME):
    """Fetch embeddings from OpenAI API for a given text."""
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Fetch embeddings in batches
batch_size = 20  # Adjust based on rate limits
embeddings = []

for i in tqdm(range(0, len(texts), batch_size), desc="Fetching Embeddings"):
    batch = texts[i : i + batch_size].tolist()
    
    try:
        response = openai.embeddings.create(
            input=batch,
            model=MODEL_NAME
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    except Exception as e:
        print(f"Error at batch {i}: {e}")
        time.sleep(5)  # Wait and retry if API fails
        continue

# Convert to NumPy array
embeddings_array = np.array(embeddings)

# Save to .npy file
output_path = "/home/maria/Downloads/scotus_embeddings.npy"
np.save(output_path, embeddings_array)

print(f"Embeddings saved at {output_path}, shape: {embeddings_array.shape}")
