#!/usr/bin/env python3
"""
dp_vae_train_hurdle.py

Improved DP-VAE for sparse + heavy-tailed tabular telemetry.

Key fixes vs v1:
1) Bucket categoricals to top-K + Other
2) Hurdle representation for sparse numerics:
   - x_is_zero (Bernoulli)
   - x_log = log1p(x) for x>0 else 0
3) Scale numeric (including indicators) to [0,1]
4) Decode categoricals by sampling from block probabilities (not argmax)
5) Decode zeros via Bernoulli sampling, then decode positive values (expm1)

Outputs:
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

    train_parquet: str = "dpvae_train_guid.parquet"
    preprocess_path: str = "dpvae_preprocess.joblib"
    model_path: str = "dpvae_model.pt"
    synth_parquet: str = "dpvae_synth.parquet"
    synth_csv: str = "dpvae_synth.csv"

    seed: int = 42
    batch_size: int = 1024
    epochs: int = 25
    lr: float = 1e-3
    hidden_dim: int = 256
    latent_dim: int = 32

    # DP
    target_epsilon: float = 8.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float | None = None  # leave None to target epsilon

    n_synth: int = 50000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Category bucketing
    os_topk: int = 3
    chassis_topk: int = 4
    vendor_topk: int = 10


# ---------------------------
# Helpers
# ---------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(m: nn.Module) -> nn.Module:
    seen = set()
    while True:
        mid = id(m)
        if mid in seen:
            break
        seen.add(mid)
        if hasattr(m, "_module"):
            m = m._module
            continue
        if hasattr(m, "module"):
            m = m.module
            continue
        break
    return m


def topk_bucket(s: pd.Series, k: int, other: str) -> pd.Series:
    s = s.fillna("UNKNOWN").astype(str)
    top = set(s.value_counts().head(k).index)
    return s.where(s.isin(top), other)


# ---------------------------
# DuckDB aggregation
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
    df = con.execute("SELECT * FROM train_guid;").df()
    out_path = cfg.base_dir / cfg.train_parquet
    df.to_parquet(out_path, index=False)
    print(f"[OK] Wrote aggregated training table: {out_path}  shape={df.shape}")
    return df


# ---------------------------
# Preprocessing: hurdle features
# ---------------------------

def add_hurdle_features(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    For each x in cols:
      - x_is_zero in {0,1}
      - x_log = log1p(x) for x>0 else 0
    Returns updated df + lists (indicator_cols, log_cols).
    """
    out = df.copy()
    indicator_cols = []
    log_cols = []
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        is_zero = (x <= 0).astype(float)
        out[f"{c}__is_zero"] = is_zero
        out[f"{c}__log"] = np.log1p(x.clip(lower=0))
        # If x is zero, force log to 0 (clean hurdle)
        out.loc[x <= 0, f"{c}__log"] = 0.0
        indicator_cols.append(f"{c}__is_zero")
        log_cols.append(f"{c}__log")
    return out, indicator_cols, log_cols


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
# Model
# ---------------------------

class VAE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.enc1 = nn.Linear(in_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, in_dim)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return torch.sigmoid(self.out(h))  # [0,1] outputs

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x, x_hat, mu, logvar, beta: float = 0.5):
    """
    beta < 1 encourages better reconstruction; beta > 1 encourages more latent usage.
    For your collapse issue, beta=0.5 often helps keep recon decent under DP.
    """
    recon = F.binary_cross_entropy(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon.detach(), kl.detach()


# ---------------------------
# DP Training
# ---------------------------

def train_dp_vae(X_train: np.ndarray, cfg: Config):
    set_seed(cfg.seed)
    X_t = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=cfg.batch_size, shuffle=True)

    in_dim = X_train.shape[1]
    base_model = VAE(in_dim, cfg.hidden_dim, cfg.latent_dim).to(cfg.device)
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
            loss, recon, kl = vae_loss(xb, x_hat, mu, logvar, beta=0.5)
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
            f"Epoch {epoch:02d}/{cfg.epochs} "
            f"loss={np.mean(losses):.4f} recon={np.mean(recons):.4f} kl={np.mean(kls):.4f} "
            f"ε(δ={cfg.delta})={eps_str}"
        )

    return model, privacy_engine


# ---------------------------
# Decoding (sampling) helpers
# ---------------------------

def sample_categorical_block(block: np.ndarray, cats: np.ndarray) -> np.ndarray:
    """
    Sample categories from a one-hot block by treating values as unnormalized probs.
    """
    p = np.clip(block, 1e-8, None)
    p = p / p.sum(axis=1, keepdims=True)
    idx = np.array([np.random.choice(len(cats), p=row) for row in p])
    return cats[idx]


def sample_bernoulli(prob_one: np.ndarray) -> np.ndarray:
    """
    prob_one in [0,1], returns 0/1 samples.
    """
    return (np.random.rand(len(prob_one)) < prob_one).astype(int)


def sample_synthetic_transformed(model: nn.Module, n: int, cfg: Config) -> np.ndarray:
    model.eval()
    inner = unwrap_model(model).to(cfg.device)
    inner.eval()
    with torch.no_grad():
        z = torch.randn((n, cfg.latent_dim), device=cfg.device)
        x_hat = inner.decode(z)
        return x_hat.cpu().numpy()


def decode_synth(
    X_synth: np.ndarray,
    pre: ColumnTransformer,
    cat_cols: list[str],
    num_cols: list[str],
    hurdle_map: dict[str, tuple[str, str]],  # base -> (is_zero_col, log_col)
) -> pd.DataFrame:
    """
    Convert transformed [0,1] features back into columns:
    - Categoricals sampled
    - Numeric scaler inverse_transform
    - Hurdle: sample is_zero, then expm1(log) for positives, else 0
    """
    ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["ohe"]
    scaler: MinMaxScaler = pre.named_transformers_["num"].named_steps["scaler"]

    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    n_cat = len(cat_feature_names)

    X_cat = X_synth[:, :n_cat]
    X_num = X_synth[:, n_cat:]

    # inverse scale numerics (including indicators/logs)
    X_num_inv = scaler.inverse_transform(X_num)
    num_df = pd.DataFrame(X_num_inv, columns=num_cols)

    # decode categoricals by sampling
    decoded = {}
    start = 0
    for i, col in enumerate(cat_cols):
        cats = ohe.categories_[i]
        k = len(cats)
        block = X_cat[:, start:start + k]
        decoded[col] = sample_categorical_block(block, cats)
        start += k
    cat_df = pd.DataFrame(decoded)

    # rebuild original base numeric columns from hurdle features
    out = pd.concat([cat_df], axis=1)

    for base, (is_zero_col, log_col) in hurdle_map.items():
        # Interpret is_zero as probability of zero.
        # Our model outputs continuous values; after inverse scaling it may not be in [0,1],
        # so we clamp it.
        p_zero = np.clip(num_df[is_zero_col].to_numpy(), 0.0, 1.0)
        zero_sample = sample_bernoulli(p_zero)  # 1 means zero

        # log column is log1p(x)
        logv = num_df[log_col].to_numpy()
        pos = np.expm1(np.clip(logv, 0.0, None))

        x = np.where(zero_sample == 1, 0.0, pos)
        out[base] = x

    # non-hurdle numerics (ram, number_of_cores)
    out["ram"] = pd.to_numeric(num_df["ram"], errors="coerce").fillna(0.0).clip(lower=0)
    out["number_of_cores"] = pd.to_numeric(num_df["number_of_cores"], errors="coerce").fillna(0.0).round().clip(lower=0)

    return out


# ---------------------------
# MAIN
# ---------------------------

def main():
    cfg = Config()
    cfg.duckdb_path = cfg.base_dir / "real_data.duckdb"

    print(f"[INFO] Base directory: {cfg.base_dir}")
    print(f"[INFO] DuckDB file:     {cfg.duckdb_path}")

    con = duckdb.connect(str(cfg.duckdb_path), read_only=False)

    # 1) Build device-level data
    df = build_training_table(con, cfg)

    # 2) Bucket categoricals (this helps a lot)
    work = df.copy()
    work["os"] = topk_bucket(work["os"], cfg.os_topk, "OtherOS")
    work["chassistype"] = topk_bucket(work["chassistype"], cfg.chassis_topk, "OtherChassis")
    work["vendor"] = topk_bucket(work["vendor"], cfg.vendor_topk, "OtherVendor")

    # 3) Clean base numerics
    work["ram"] = pd.to_numeric(work["ram"], errors="coerce").fillna(0.0).clip(lower=0)
    work["number_of_cores"] = pd.to_numeric(work["number_of_cores"], errors="coerce").fillna(0.0).clip(lower=0)

    # 4) Hurdle features for sparse / heavy-tailed cols
    sparse_cols = [
        "received_bytes", "sent_bytes", "net_nr_samples",
        "frgnd_total_duration_sec", "frgnd_total_nrs", "frgnd_rows",
        "userwait_total_duration_ms", "userwait_rows",
    ]
    work, ind_cols, log_cols = add_hurdle_features(work, sparse_cols)

    # Map from base -> (is_zero, log)
    hurdle_map = {c: (f"{c}__is_zero", f"{c}__log") for c in sparse_cols}

    # 5) Preprocess
    cat_cols = ["os", "chassistype", "vendor"]
    num_cols = ["ram", "number_of_cores"] + ind_cols + log_cols

    train_df, _ = train_test_split(work, test_size=0.1, random_state=cfg.seed)

    pre = make_preprocessor(cat_cols, num_cols)
    X_train = pre.fit_transform(train_df[cat_cols + num_cols])

    joblib.dump(pre, cfg.base_dir / cfg.preprocess_path)
    print(f"[OK] Saved preprocessor: {cfg.base_dir / cfg.preprocess_path}")
    print(f"[OK] Preprocessed X_train: {X_train.shape}")

    # 6) Train DP-VAE
    model, pe = train_dp_vae(X_train, cfg)

    # Save unwrapped model
    inner = unwrap_model(model)
    torch.save(inner.state_dict(), cfg.base_dir / cfg.model_path)
    print(f"[OK] Saved model: {cfg.base_dir / cfg.model_path}")

    # 7) Sample synthetic in transformed space
    X_synth = sample_synthetic_transformed(model, cfg.n_synth, cfg)
    print(f"[OK] Sampled synthetic (transformed): {X_synth.shape}")

    # 8) Decode back to original columns
    synth_df = decode_synth(X_synth, pre, cat_cols, num_cols, hurdle_map)
    synth_df.insert(0, "guid", [f"SYNTH_{i:08d}" for i in range(len(synth_df))])

    # 9) Save
    out_parquet = cfg.base_dir / cfg.synth_parquet
    out_csv = cfg.base_dir / cfg.synth_csv

    synth_df.to_parquet(out_parquet, index=False)
    synth_df.to_csv(out_csv, index=False)

    print(f"[OK] Wrote synthetic parquet: {out_parquet}  shape={synth_df.shape}")
    print(f"[OK] Wrote synthetic CSV:     {out_csv}")

    try:
        eps = pe.get_epsilon(cfg.delta)
        print(f"[DP] Final privacy: ε={eps:.3f} at δ={cfg.delta}")
    except Exception as e:
        print(f"[DP] Could not compute epsilon automatically: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()

