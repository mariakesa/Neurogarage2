import torch
import numpy as np
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path

# ==== Config ====
SEED = 1337
LAYER_INDEX = 6
LAMBDA = 5.0
MAX_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.95
LATENT_DIR = "latents"

PROMPT = "The nature of intelligence lies in"

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load GPT-2 ====
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", output_hidden_states=True, return_dict_in_generate=True)
model.eval().to(device)

# ==== Utility ====
def load_latent_vector(name):
    if name == "none":
        return None
    path = Path(LATENT_DIR) / f"{name}.npy"
    return torch.tensor(np.load(path), dtype=torch.float32).to(device)

# ==== Injection and Generation ====
def generate_with_latent(prompt, latent_vector=None):
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Inject latent vector at LAYER_INDEX
    if latent_vector is not None:
        with torch.no_grad():
            outputs = model.transformer(**inputs)
            residual = outputs.hidden_states[LAYER_INDEX][0]
            residual[-1] += LAMBDA * latent_vector
            logits = model.lm_head(residual.unsqueeze(0))
            next_token_id = torch.argmax(logits[0, -1]).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id.unsqueeze(0))], dim=1)

    # Generate full continuation
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    scores = outputs.scores
    return text, scores

# ==== Compare Latents ====
LATENTS = {
    "none": None,
    "maria": "freedom_latent_maria_layer6",
    "shakespeare": "shakespeare_cpca_layer6_cpc1",
}

results = {}

for label, latent_name in LATENTS.items():
    vec = load_latent_vector(latent_name) if latent_name else None
    print(f"\n🧠 Generating with: {label}")
    text, scores = generate_with_latent(PROMPT, vec)
    results[label] = {
        "text": text,
        "scores": scores
    }
    print(f"\n{text}\n")

# ==== Save Results ====
with open("steering_comparison_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("✅ All generations complete. Results saved to steering_comparison_results.pkl.")
