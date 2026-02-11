from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn

ART_DIR = Path("/Users/hanatjendrawasi/Desktop/DSC180B-Q2/dpsgd/query2/artifacts")
MODEL_PATH = ART_DIR / "dpvae_query2.pt"
OUT_PATH = ART_DIR / "synthetic_query2.parquet"

N_SYNTH = 50000  # choose how many synthetic rows to generate
SEED = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DPVAE(nn.Module):
    def __init__(self, num_dim: int, cat_sizes: list[int], latent_dim=32, hidden=256):
        super().__init__()
        self.num_dim = num_dim
        self.cat_sizes = cat_sizes

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.num_head = nn.Sequential(nn.Linear(hidden, num_dim), nn.Sigmoid())
        self.cat_heads = nn.ModuleList([nn.Linear(hidden, k) for k in cat_sizes])

    def decode(self, z):
        h = self.decoder(z)
        x_num_hat = self.num_head(h)
        cat_logits = [head(h) for head in self.cat_heads]
        return x_num_hat, cat_logits


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    meta = ckpt["meta"]
    cat_sizes = ckpt["cat_sizes"]
    num_dim = ckpt["num_dim"]

    model = DPVAE(
        num_dim=num_dim,
        cat_sizes=cat_sizes,
        latent_dim=ckpt["latent_dim"],
        hidden=ckpt["hidden"],
    ).to(device)

    # load only matching decoder weights from full model
    sd = ckpt["state_dict"]
    model_sd = model.state_dict()
    for k in list(model_sd.keys()):
        if k in sd:
            model_sd[k] = sd[k]
    model.load_state_dict(model_sd, strict=False)
    model.eval()

    # Sample z ~ N(0,1)
    with torch.no_grad():
        z = torch.randn(N_SYNTH, ckpt["latent_dim"], device=device)
        x_num_hat, cat_logits = model.decode(z)

        x_num = x_num_hat.cpu().numpy().astype(np.float32)  # in [0,1]
        x_cat = []
        for logits in cat_logits:
            probs = torch.softmax(logits, dim=1)
            idx = torch.multinomial(probs, num_samples=1).squeeze(1)
            x_cat.append(idx.cpu().numpy().astype(np.int64))
        x_cat = np.stack(x_cat, axis=1)

    # Decode categoricals back to strings
    cat_cols = meta["cat_cols"]
    vocabs = meta["vocabs"]

    out = {}
    for j, c in enumerate(cat_cols):
        itos = vocabs[c]["itos"]
        out[c] = [itos[i] for i in x_cat[:, j]]

    # Convert numeric back to approximate raw scale (inverse of log1p_minmax)
    # x in [0,1] => raw = expm1(x * log1p(clip_max))
    clip = meta["clip_p99_raw"]
    num_cols = meta["num_cols"]  # ["nrs","sent_bytes","received_bytes","delta_pos","delta_neg"]

    def inv_unit_to_raw(u, clip_max):
        u = np.clip(u, 0.0, 1.0)
        return np.expm1(u * np.log1p(clip_max))

    raw = {}
    for j, name in enumerate(num_cols):
        raw[name] = inv_unit_to_raw(x_num[:, j], float(clip[name]))

    # Reconstruct delta_bytes from pos/neg
    delta_bytes = raw["delta_pos"] - raw["delta_neg"]

    df = pd.DataFrame({
        "nrs": raw["nrs"],
        "sent_bytes": raw["sent_bytes"],
        "received_bytes": raw["received_bytes"],
        "delta_bytes": delta_bytes,
        **out,
    })

    # Your bucketed training table uses vendor_bucket/model_bucket names.
    # Keep consistent:
    df = df.rename(columns={"vendor_bucket": "vendor_bucket", "model_bucket": "model_bucket"})

    df.to_parquet(OUT_PATH, index=False)
    print("Saved synthetic parquet:", OUT_PATH)
    print(df.head(10))


if __name__ == "__main__":
    main()

