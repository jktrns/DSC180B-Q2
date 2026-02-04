from pathlib import Path
import json
import numpy as np
import pandas as pd

# -------------------------
# INPUT / OUTPUT PATHS
# -------------------------
DATA_PATH = Path("/Users/hanatjendrawasi/Desktop/DSC180B-Q2/preprocessing/query2/train_query2_bucketed.parquet")
OUT_DIR = Path("/Users/hanatjendrawasi/Desktop/DSC180B-Q2/dpsgd/query2/artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NUM = OUT_DIR / "X_num.npy"      # float32, shape [N, D_num] in [0,1]
OUT_CAT = OUT_DIR / "X_cat.npy"      # int64,   shape [N, D_cat] (category indices)
OUT_META = OUT_DIR / "meta.json"     # vocab + clip bounds + columns

# -------------------------
# COLUMNS
# -------------------------
CAT_COLS = ["chassistype", "os", "vendor_bucket", "model_bucket"]
NUM_COLS = ["nrs", "sent_bytes", "received_bytes", "delta_bytes"]

# -------------------------
# HELPERS
# -------------------------
def clean_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def p99_clip_bounds(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return 1.0
    return float(np.percentile(x, 99))

def log1p_minmax_to_unit(x: np.ndarray, clip_max: float) -> np.ndarray:
    # clip to [0, clip_max], then log1p, then divide by log1p(clip_max) to get [0,1]
    x = np.clip(x, 0.0, clip_max)
    denom = np.log1p(clip_max) if clip_max > 0 else 1.0
    return (np.log1p(x) / denom).astype(np.float32)

def make_vocab(values: pd.Series):
    # reserve 0 for "Other" and keep stable ordering
    vals = values.dropna().astype(str)
    uniq = sorted(set(vals.tolist()))
    if "Other" in uniq:
        uniq.remove("Other")
    itos = ["Other"] + uniq
    stoi = {v: i for i, v in enumerate(itos)}
    return {"itos": itos, "stoi": stoi}

def map_to_index(values: pd.Series, stoi: dict) -> np.ndarray:
    return values.astype(str).map(lambda v: stoi.get(v, stoi["Other"])).astype(np.int64).to_numpy()

def main():
    df = pd.read_parquet(DATA_PATH)

    # Clean
    for c in CAT_COLS:
        df[c] = clean_str(df[c])

    # ---------
    # NUMERIC: freeze clip bounds using p99 in RAW space
    # ---------
    # For delta, we split into pos/neg magnitudes (much easier to model)
    df["delta_pos"] = np.clip(df["delta_bytes"].to_numpy(dtype=np.float64), 0, None)
    df["delta_neg"] = np.clip((-df["delta_bytes"]).to_numpy(dtype=np.float64), 0, None)

    num_defs = {
        "nrs": df["nrs"],
        "sent_bytes": df["sent_bytes"],
        "received_bytes": df["received_bytes"],
        "delta_pos": df["delta_pos"],
        "delta_neg": df["delta_neg"],
    }

    clip = {k: p99_clip_bounds(v) for k, v in num_defs.items()}

    X_num = np.stack([
        log1p_minmax_to_unit(df["nrs"].to_numpy(dtype=np.float64), clip["nrs"]),
        log1p_minmax_to_unit(df["sent_bytes"].to_numpy(dtype=np.float64), clip["sent_bytes"]),
        log1p_minmax_to_unit(df["received_bytes"].to_numpy(dtype=np.float64), clip["received_bytes"]),
        log1p_minmax_to_unit(df["delta_pos"].to_numpy(dtype=np.float64), clip["delta_pos"]),
        log1p_minmax_to_unit(df["delta_neg"].to_numpy(dtype=np.float64), clip["delta_neg"]),
    ], axis=1).astype(np.float32)

    # ---------
    # CATEGORICAL: vocab per column, map to indices
    # ---------
    vocabs = {}
    X_cat_cols = []
    for c in CAT_COLS:
        vocab = make_vocab(df[c])
        vocabs[c] = {"itos": vocab["itos"]}  # store itos only; stoi can be reconstructed
        stoi = {v: i for i, v in enumerate(vocab["itos"])}
        X_cat_cols.append(map_to_index(df[c], stoi))

    X_cat = np.stack(X_cat_cols, axis=1).astype(np.int64)

    # Save
    np.save(OUT_NUM, X_num)
    np.save(OUT_CAT, X_cat)

    meta = {
        "data_path": str(DATA_PATH),
        "cat_cols": CAT_COLS,
        "num_cols": ["nrs","sent_bytes","received_bytes","delta_pos","delta_neg"],
        "clip_p99_raw": clip,
        "vocabs": vocabs,
        "n_rows": int(len(df)),
    }
    OUT_META.write_text(json.dumps(meta, indent=2))

    print("Saved:", OUT_NUM)
    print("Saved:", OUT_CAT)
    print("Saved:", OUT_META)
    print("X_num:", X_num.shape, "range:", float(X_num.min()), float(X_num.max()))
    print("X_cat:", X_cat.shape)

if __name__ == "__main__":
    main()

