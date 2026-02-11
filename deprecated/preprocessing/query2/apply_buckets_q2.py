# preprocessing/query2/apply_buckets_query2.py
#
# Applies frozen bucket mappings to Query2 training data and writes a bucketed parquet.
#
# Inputs:
#   preprocessing/query2/train_query2_one_row_per_guid.parquet
#   preprocessing/query2/query2_vendor_map.json
#   preprocessing/query2/query2_model_map.json
#
# Output:
#   preprocessing/query2/train_query2_bucketed.parquet
#
# Run:
#   python preprocessing/query2/apply_buckets_query2.py

from pathlib import Path
import json
import pandas as pd


# ---- Paths (match your current setup) ----
BASE = Path("/Users/hanatjendrawasi/Desktop/DSC180B-Q2/preprocessing/query2")

DATA_PATH = BASE / "train_query2_one_row_per_guid.parquet"
VENDOR_MAP_PATH = BASE / "query2_vendor_map.json"
MODEL_MAP_PATH = BASE / "query2_model_map.json"

OUT_PATH = BASE / "train_query2_bucketed.parquet"


def _clean_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing mapping file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    # Basic checks
    required = {"guid", "vendor", "model", "chassistype", "os"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")

    # Clean strings
    for c in ["vendor", "model", "chassistype", "os"]:
        df[c] = _clean_str(df[c])

    vendor_map = _load_json(VENDOR_MAP_PATH)
    model_map = _load_json(MODEL_MAP_PATH)

    # Apply frozen mappings (anything unseen -> "Other")
    df["vendor_bucket"] = df["vendor"].map(vendor_map).fillna("Other")
    df["model_bucket"] = df["model"].map(model_map).fillna("Other")

    # Optional: keep original + bucketed
    # If you want to drop originals later for training, do it in the training script.
    keep_cols = [
        "guid",
        "nrs",
        "received_bytes",
        "sent_bytes",
        "delta_bytes",
        "chassistype",
        "os",
        "vendor_bucket",
        "model_bucket",
    ]
    missing2 = [c for c in keep_cols if c not in df.columns]
    if missing2:
        raise ValueError(f"Expected columns not found: {missing2}")

    out = df[keep_cols].copy()

    # Quick sanity report
    print("\n=== OUTPUT SHAPE ===")
    print("rows:", len(out))
    print("distinct guids:", out["guid"].nunique())

    print("\n=== BUCKET VALUE COUNTS (top 15) ===")
    print("\nvendor_bucket:")
    print(out["vendor_bucket"].value_counts().head(15))
    print("\nmodel_bucket:")
    print(out["model_bucket"].value_counts().head(15))

    print("\n% Other")
    print("vendor_bucket:", (out["vendor_bucket"] == "Other").mean())
    print("model_bucket:", (out["model_bucket"] == "Other").mean())

    out.to_parquet(OUT_PATH, index=False)
    print("\nSaved:", OUT_PATH)


if __name__ == "__main__":
    main()

