from pathlib import Path
import json
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from opacus import PrivacyEngine

ART_DIR = Path("/Users/hanatjendrawasi/Desktop/DSC180B-Q2/dpsgd/query2/artifacts")
X_NUM_PATH = ART_DIR / "X_num.npy"
X_CAT_PATH = ART_DIR / "X_cat.npy"
META_PATH  = ART_DIR / "meta.json"

MODEL_PATH = ART_DIR / "dpvae_query2.pt"
TRAIN_LOG  = ART_DIR / "train_log.jsonl"

# -------------------------
# DP + TRAIN CONFIG
# -------------------------
SEED = 7
BATCH_SIZE = 512
EPOCHS = 20

LR = 1e-3
WEIGHT_DECAY = 0.0

# DP knobs
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 1.2   # increase for more privacy, decrease for better utility
DELTA = 1e-5

# VAE knobs
LATENT_DIM = 32
HIDDEN = 256
BETA_KL = 0.01  # KL weight (small is usually better for tabular)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TabDataset(Dataset):
    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray):
        self.x_num = torch.from_numpy(x_num).float()
        self.x_cat = torch.from_numpy(x_cat).long()

    def __len__(self):
        return self.x_num.shape[0]

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx]


class DPVAE(nn.Module):
    def __init__(self, num_dim: int, cat_sizes: list[int], latent_dim=32, hidden=256):
        super().__init__()
        self.num_dim = num_dim
        self.cat_sizes = cat_sizes
        self.cat_dim_total = sum(cat_sizes)

        # encoder takes: [num + onehot(cat)]
        enc_in = num_dim + self.cat_dim_total
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        # decoder outputs:
        # - num in [0,1] via sigmoid
        # - logits per categorical column
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.num_head = nn.Sequential(nn.Linear(hidden, num_dim), nn.Sigmoid())
        self.cat_heads = nn.ModuleList([nn.Linear(hidden, k) for k in cat_sizes])

    def onehot_cats(self, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat: [B, D_cat], each entry is index
        parts = []
        for j, k in enumerate(self.cat_sizes):
            parts.append(torch.nn.functional.one_hot(x_cat[:, j], num_classes=k).float())
        return torch.cat(parts, dim=1)

    def encode(self, x_num, x_cat):
        x = torch.cat([x_num, self.onehot_cats(x_cat)], dim=1)
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-10, max=10)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        x_num_hat = self.num_head(h)
        cat_logits = [head(h) for head in self.cat_heads]
        return x_num_hat, cat_logits

    def forward(self, x_num, x_cat):
        mu, logvar = self.encode(x_num, x_cat)
        z = self.reparam(mu, logvar)
        x_num_hat, cat_logits = self.decode(z)
        return x_num_hat, cat_logits, mu, logvar


def vae_loss(x_num, x_cat, x_num_hat, cat_logits, mu, logvar, beta_kl=0.01):
    # numeric: MSE in [0,1]
    loss_num = torch.nn.functional.mse_loss(x_num_hat, x_num, reduction="mean")

    # categorical: CE per column
    loss_cat = 0.0
    for j, logits in enumerate(cat_logits):
        loss_cat = loss_cat + torch.nn.functional.cross_entropy(logits, x_cat[:, j], reduction="mean")

    # KL
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return loss_num + loss_cat + beta_kl * kl, float(loss_num.item()), float(loss_cat.item()), float(kl.item())


def main():
    set_seed(SEED)

    meta = json.loads(META_PATH.read_text())
    x_num = np.load(X_NUM_PATH)
    x_cat = np.load(X_CAT_PATH)

    cat_cols = meta["cat_cols"]
    # reconstruct cat sizes from itos lengths
    cat_sizes = [len(meta["vocabs"][c]["itos"]) for c in cat_cols]

    ds = TabDataset(x_num, x_cat)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = DPVAE(num_dim=x_num.shape[1], cat_sizes=cat_sizes, latent_dim=LATENT_DIM, hidden=HIDDEN).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    privacy_engine = PrivacyEngine()
    model, optim, dl = privacy_engine.make_private(
        module=model,
        optimizer=optim,
        data_loader=dl,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM,
    )

    print("Training with DP-SGD (Opacus)")
    print("N:", len(ds), "batch:", BATCH_SIZE, "epochs:", EPOCHS)
    print("noise_multiplier:", NOISE_MULTIPLIER, "max_grad_norm:", MAX_GRAD_NORM, "delta:", DELTA)

    TRAIN_LOG.write_text("")  # reset

    steps = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = {"loss": 0.0, "num": 0.0, "cat": 0.0, "kl": 0.0}
        n_batches = 0

        for xb_num, xb_cat in dl:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)

            optim.zero_grad(set_to_none=True)
            x_num_hat, cat_logits, mu, logvar = model(xb_num, xb_cat)

            loss, ln, lc, lkl = vae_loss(xb_num, xb_cat, x_num_hat, cat_logits, mu, logvar, beta_kl=BETA_KL)
            loss.backward()
            optim.step()

            running["loss"] += float(loss.item())
            running["num"] += ln
            running["cat"] += lc
            running["kl"] += lkl
            n_batches += 1
            steps += 1

        # privacy accounting
        eps = privacy_engine.get_epsilon(delta=DELTA)

        row = {
            "epoch": epoch,
            "loss": running["loss"] / n_batches,
            "loss_num": running["num"] / n_batches,
            "loss_cat": running["cat"] / n_batches,
            "kl": running["kl"] / n_batches,
            "epsilon": float(eps),
            "delta": float(DELTA),
            "noise_multiplier": float(NOISE_MULTIPLIER),
            "max_grad_norm": float(MAX_GRAD_NORM),
        }
        with open(TRAIN_LOG, "a") as f:
            f.write(json.dumps(row) + "\n")

        print(f"epoch {epoch:02d} | loss {row['loss']:.4f} (num {row['loss_num']:.4f}, cat {row['loss_cat']:.4f}, kl {row['kl']:.4f}) | eps {row['epsilon']:.2f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": meta,
            "cat_sizes": cat_sizes,
            "num_dim": x_num.shape[1],
            "latent_dim": LATENT_DIM,
            "hidden": HIDDEN,
        },
        MODEL_PATH,
    )
    print("Saved model:", MODEL_PATH)
    print("Saved log:", TRAIN_LOG)


if __name__ == "__main__":
    main()

