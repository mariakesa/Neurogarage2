import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
from pathlib import Path

class EmbeddingDataset:
    def __init__(self, path: str, layer_index: int = 1):
        """Initialize with full GPT-2 embeddings and extract a specific layer."""
        self.full = np.load(path)  # expects shape [n, 3072]
        self.layer_index = layer_index
        self.layer = self.full[:, 768 * layer_index: 768 * (layer_index + 1)]  # extract 768-dim layer
        self.normalized = None

    def normalize(self, scaler: StandardScaler = None):
        """Standardize the dataset."""
        if scaler is None:
            scaler = StandardScaler()
            self.normalized = scaler.fit_transform(self.layer)
            return scaler
        else:
            self.normalized = scaler.transform(self.layer)
            return scaler

class LatentExtractor:
    def __init__(self, foreground: EmbeddingDataset, background: EmbeddingDataset, alpha: float = 1.0):
        self.X_fg = foreground.normalized
        self.X_bg = background.normalized
        self.alpha = alpha
        self.components = None
        self.eigenvalues = None

    def run_cpca(self):
        C_fg = np.cov(self.X_fg, rowvar=False)
        C_bg = np.cov(self.X_bg, rowvar=False)
        C_c = C_fg - self.alpha * C_bg
        eigvals, eigvecs = eigh(C_c)
        idx = np.argsort(eigvals)[::-1]
        self.eigenvalues = eigvals[idx]
        self.components = eigvecs[:, idx]  # shape [768, n_components]
        return self.components

    def save_top_component(self, filename: str):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        np.save(filename, self.components[:, 0])

def run_analysis_pipeline(
    maria_path="maria_gpt2.npy",
    shakespeare_path="shakespeare_gpt2.npy",
    layer_index=1,
    alpha=1.0,
    output_dir="latents"
):
    maria = EmbeddingDataset(maria_path, layer_index)
    shakespeare = EmbeddingDataset(shakespeare_path, layer_index)

    # Normalize Maria first to reuse scaler
    scaler = maria.normalize()
    shakespeare.normalize(scaler)

    # Maria vs Shakespeare → "freedom_latent"
    print("→ Extracting freedom_latent_maria...")
    maria_cpca = LatentExtractor(maria, shakespeare, alpha)
    maria_cpca.run_cpca()
    maria_cpca.save_top_component(f"{output_dir}/freedom_latent_maria_layer6.npy")

    # Now reverse
    print("→ Extracting shakespeare_cpca...")
    shakespeare_cpca = LatentExtractor(shakespeare, maria, alpha)
    shakespeare_cpca.run_cpca()
    shakespeare_cpca.save_top_component(f"{output_dir}/shakespeare_cpca_layer6_cpc1.npy")

    print("✅ Latents saved!")

# Optional: Uncomment to run directly
if __name__ == "__main__":
    run_analysis_pipeline()
