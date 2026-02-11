#!/usr/bin/env python3
"""
dp_vae_train_v2.py

DP-VAE (with DP-SGD via Opacus) for Intel telemetry.

Pipeline:
1) Read event-level `real_data` + device-level `sysinfo` from real_data.duckdb
2) Aggregate to one-row-per-guid (`train_guid`)
3) Preprocess: OneHot categorical + MinMax scale numerics
4) Train VAE under DP-SGD using Opacus (epsilon accountant)
5) Sample synthetic device rows and decode back to original feature space

Outputs (in same directory as this script):
- dpvae_train_guid.parquet
- dpvae_preprocess.joblib
- dpvae_model.pt
- dpvae_synth.parquet
- dpvae_synth.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

from opacus import PrivacyEngine


# ---------------------------
# CONFIG
# ---------------------------

@dataclass
class Config:
    base_dir: Path = Path(__file__).resolve().parent
    duckdb_path: Path = None  # set in main

    # outputs
    train_parquet: str = "dpvae_train_guid.parquet"
    preprocess_path: str = "dpvae_preprocess.joblib"
    model_path: str = "dpvae_model.pt"
    synth_parquet: str = "dpvae_synth.parquet"
    synth_csv: str = "dpvae_synth.csv"

    # debug
    max_rows_for_debug: int | None = None  # e.g., 20000 for quick tests, else None

    # training
    seed: int = 42
    batch_size: int = 1024
    epochs: int = 20
    lr: float = 1e-3
    hidden_dim: int = 128
    latent_dim: int = 16

    # DP params
    target_epsilon: float = 8.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float | None = None  # set to a float to use fixed noise mode

    # sampling
    n_synth: int = 50000

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# Robust unwrapping for Opacus wrappers
# ---------------------------

def unwrap_model(m: nn.Module) -> nn.Module:
    """
    Unwrap Opacus (GradSampleModule) and other wrappers until we reach the original model.
    Opacus typically stores wrapped model in `._module`.
    Some wrappers store in `.module`.
    """
    seen = set()
    while True:
        mid = id(m)
        if mid in seen:
            break
        seen.add(mid)

        if hasattr(m, "_module"):
            m = getattr(m, "_module")
            continue
        if hasattr(m, "module"):
            m = getattr(m, "module")
            continue
        break
    return m


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# DuckDB aggregation to one-row-per-guid
# ---------------------------

AGG_SQL = """
CREATE OR REPLACE TABLE train_guid AS
WITH
net AS (
  SELECT
    guid,
    SUM(
      CASE WHEN input_description = 'OS:NETWORK INTERFACE::BYTES RECEIVED/SEC::'
           THEN avg_bytes_sec::DOUBLE * nr_samples::DOUBLE * 5.0
           ELSE 0 END
    ) AS received_bytes,
    SUM(
      CASE WHEN input_description = 'OS:NETWORK INTERFACE::BYTES SENT/SEC::'
           THEN avg_bytes_sec::DOUBLE * nr_samples::DOUBLE * 5.0
           ELSE 0 END
    ) AS sent_bytes,
    SUM(nr_samples)::BIGINT AS net_nr_samples
  FROM real_data
  WHERE record_type = 'network'
  GROUP BY 1
),
fr AS (
  SELECT
    guid,
    SUM(duration_sec)::DOUBLE AS frgnd_total_duration_sec,
    SUM(nrs)::BIGINT AS frgnd_total_nrs,
    COUNT(*)::BIGINT AS frgnd_rows
  FROM real_data
  WHERE record_type = 'foreground_usage'
  GROUP BY 1
),
uw AS (
  SELECT
    guid,
    SUM(duration_ms)::BIGINT AS userwait_total_duration_ms,
    COUNT(*)::BIGINT AS userwait_rows
  FROM real_data
  WHERE record_type = 'userwait'
  GROUP BY 1
),
guids AS (
  SELECT DISTINCT guid FROM real_data
)
SELECT
  g.guid::TEXT AS guid,

  s.os::TEXT AS os,
  s.chassistype::TEXT AS chassistype,
  s.vendor::TEXT AS vendor,
  s.model::TEXT AS model,
  TRY_CAST(s.ram AS DOUBLE) AS ram,
  TRY_CAST(s.number_of_cores AS BIGINT) AS number_of_cores,

  COALESCE(net.received_bytes, 0.0) AS received_bytes,
  COALESCE(net.sent_bytes, 0.0) AS sent_bytes,
  COALESCE(net.net_nr_samples, 0) AS net_nr_samples,

  COALESCE(fr.frgnd_total_duration_sec, 0.0) AS frgnd_total_duration_sec,
  COALESCE(fr.frgnd_total_nrs, 0) AS frgnd_total_nrs,
  COALESCE(fr.frgnd_rows, 0) AS frgnd_rows,

  COALESCE(uw.userwait_total_duration_ms, 0) AS userwait_total_duration_ms,
  COALESCE(uw.userwait_rows, 0) AS userwait_rows
FROM guids g
LEFT JOIN sysinfo s ON s.guid = g.guid
LEFT JOIN net ON net.guid = g.guid
LEFT JOIN fr  ON fr.guid  = g.guid
LEFT JOIN uw  ON uw.guid  = g.guid
;
"""


def build_training_table(con: duckdb.DuckDBPyConnection, cfg: Config) -> pd.DataFrame:
    con.execute(AGG_SQL)
    if cfg.max_rows_for_debug is not None:
        df = con.execute(f"SELECT * FROM train_guid LIMIT {cfg.max_rows_for_debug};").df()
    else:
        df = con.execute("SELECT * FROM train_guid;").df()

    out_path = cfg.base_dir / cfg.train_parquet
    df.to_parquet(out_path, index=False)
    print(f"[OK] Wrote aggregated training table: {out_path}  shape={df.shape}")
    return df


# ---------------------------
# Preprocessing
# ---------------------------

def make_preprocessor(cat_cols, num_cols) -> ColumnTransformer:
    cat_pipe = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    num_pipe = Pipeline(steps=[
        ("scaler", MinMaxScaler())
    ])
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop"
    )


# ---------------------------
# VAE Model
# ---------------------------

class VAE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
            nn.Sigmoid(),  # for [0,1] scaled data
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x, x_hat, mu, logvar):
    recon = F.binary_cross_entropy(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon.detach(), kl.detach()


# ---------------------------
# Training DP-VAE with Opacus
# ---------------------------

def train_dp_vae(X_train: np.ndarray, cfg: Config):
    set_seed(cfg.seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_train_t), batch_size=cfg.batch_size, shuffle=True)

    in_dim = X_train.shape[1]
    base_model = VAE(in_dim=in_dim, hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim).to(cfg.device)
    optimizer = torch.optim.Adam(base_model.parameters(), lr=cfg.lr)

    privacy_engine = PrivacyEngine()

    if cfg.noise_multiplier is None:
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=base_model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=cfg.target_epsilon,
            target_delta=cfg.delta,
            epochs=cfg.epochs,
            max_grad_norm=cfg.max_grad_norm,
        )
        print(f"[DP] Using target ε={cfg.target_epsilon}, δ={cfg.delta}, max_grad_norm={cfg.max_grad_norm}")
    else:
        model, optimizer, loader = privacy_engine.make_private(
            module=base_model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=cfg.noise_multiplier,
            max_grad_norm=cfg.max_grad_norm,
        )
        print(f"[DP] Using fixed noise_multiplier={cfg.noise_multiplier}, δ={cfg.delta}, max_grad_norm={cfg.max_grad_norm}")

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        losses, recons, kls = [], [], []
        for (xb,) in loader:
            xb = xb.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            x_hat, mu, logvar = model(xb)
            loss, recon, kl = vae_loss(xb, x_hat, mu, logvar)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            recons.append(recon.item())
            kls.append(kl.item())

        try:
            eps = privacy_engine.get_epsilon(cfg.delta)
            eps_str = f"{eps:.3f}"
        except Exception:
            eps_str = "N/A"

        print(
            f"Epoch {epoch:02d}/{cfg.epochs}  "
            f"loss={np.mean(losses):.4f}  recon={np.mean(recons):.4f}  kl={np.mean(kls):.4f}  "
            f"ε(δ={cfg.delta})={eps_str}"
        )

    return model, privacy_engine


# ---------------------------
# Sampling + inverse transform
# ---------------------------

def sample_synthetic(model: nn.Module, n: int, cfg: Config) -> np.ndarray:
    """
    Sample in latent space and decode using the UNWRAPPED VAE.
    This avoids calling .decode on Opacus wrappers.
    """
    model.eval()
    inner = unwrap_model(model).to(cfg.device)
    inner.eval()

    with torch.no_grad():
        z = torch.randn((n, cfg.latent_dim), device=cfg.device)
        x_hat = inner.decode(z)
        return x_hat.cpu().numpy()


def inverse_transform_to_dataframe(X_synth: np.ndarray, pre: ColumnTransformer, cat_cols, num_cols) -> pd.DataFrame:
    ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["ohe"]
    scaler: MinMaxScaler = pre.named_transformers_["num"].named_steps["scaler"]

    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    n_cat = len(cat_feature_names)

    X_cat = X_synth[:, :n_cat]
    X_num = X_synth[:, n_cat:]

    X_num_inv = scaler.inverse_transform(X_num)

    decoded = {}
    start = 0
    for i, col in enumerate(cat_cols):
        cats = ohe.categories_[i]
        k = len(cats)
        block = X_cat[:, start:start + k]
        idx = np.argmax(block, axis=1)
        decoded[col] = cats[idx]
        start += k

    df_cat = pd.DataFrame(decoded)
    df_num = pd.DataFrame(X_num_inv, columns=num_cols)
    return pd.concat([df_cat, df_num], axis=1)


# ---------------------------
# MAIN
# ---------------------------

def main():
    cfg = Config()
    cfg.duckdb_path = cfg.base_dir / "real_data.duckdb"

    print(f"[INFO] Base directory: {cfg.base_dir}")
    print(f"[INFO] DuckDB file:     {cfg.duckdb_path}")

    if not cfg.duckdb_path.exists():
        raise FileNotFoundError(f"Could not find {cfg.duckdb_path}. Put this script next to real_data.duckdb.")

    con = duckdb.connect(str(cfg.duckdb_path), read_only=False)

    # Build / load training data
    df = build_training_table(con, cfg)

    # Start without high-cardinality `model` (it will blow up one-hot)
    cat_cols = ["os", "chassistype", "vendor"]

    num_cols = [
        "ram",
        "number_of_cores",
        "received_bytes",
        "sent_bytes",
        "net_nr_samples",
        "frgnd_total_duration_sec",
        "frgnd_total_nrs",
        "frgnd_rows",
        "userwait_total_duration_ms",
        "userwait_rows",
    ]

    work = df.copy()
    for c in cat_cols:
        work[c] = work[c].fillna("UNKNOWN").astype(str)
    for c in num_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)

    train_df, _ = train_test_split(work, test_size=0.1, random_state=cfg.seed)

    pre = make_preprocessor(cat_cols, num_cols)
    X_train = pre.fit_transform(train_df[cat_cols + num_cols])
    print(f"[OK] Preprocessed X_train: {X_train.shape}")

    joblib.dump(pre, cfg.base_dir / cfg.preprocess_path)
    print(f"[OK] Saved preprocessor: {cfg.base_dir / cfg.preprocess_path}")

    # Train
    model, pe = train_dp_vae(X_train, cfg)

    # Save *unwrapped* model weights
    inner = unwrap_model(model)
    torch.save(inner.state_dict(), cfg.base_dir / cfg.model_path)
    print(f"[OK] Saved model: {cfg.base_dir / cfg.model_path}")

    # Sample synthetic
    X_synth = sample_synthetic(model, cfg.n_synth, cfg)
    print(f"[OK] Sampled synthetic (transformed): {X_synth.shape}")

    # Decode back
    synth_df = inverse_transform_to_dataframe(X_synth, pre, cat_cols, num_cols)

    # Clip invalid negatives
    nonneg_cols = num_cols
    for c in nonneg_cols:
        synth_df[c] = pd.to_numeric(synth_df[c], errors="coerce").fillna(0.0)
        synth_df[c] = synth_df[c].clip(lower=0)

    synth_df.insert(0, "guid", [f"SYNTH_{i:08d}" for i in range(len(synth_df))])

    out_parquet = cfg.base_dir / cfg.synth_parquet
    out_csv = cfg.base_dir / cfg.synth_csv

    synth_df.to_parquet(out_parquet, index=False)
    print(f"[OK] Wrote synthetic parquet: {out_parquet}  shape={synth_df.shape}")

    synth_df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote synthetic CSV:     {out_csv}")

    try:
        eps = pe.get_epsilon(cfg.delta)
        print(f"[DP] Final privacy: ε={eps:.3f} at δ={cfg.delta}")
    except Exception as e:
        print(f"[DP] Could not compute epsilon automatically: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()

