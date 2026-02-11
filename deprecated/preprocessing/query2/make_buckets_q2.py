# preprocessing/query2/make_buckets_query2.py
# Creates frozen category bucket mappings for Query 2.
# - vendor: keep vendors with >= 200 distinct guids, else "Other"
# - model:  keep models  with >= 500 distinct guids, else "Other"
#
# Outputs (JSON):
#   preprocessing/mappings/query2_vendor_map.json
#   preprocessing/mappings/query2_model_map.json
#
# Run from repo root:
#   python preprocessing/query2/make_buckets_query2.py

from pathlib import Path
import json
import pandas as pd


# -------------------------
# Config
# -------------------------
VENDOR_MIN_GUIDS = 200
MODEL_MIN_GUIDS = 500

# Adjust if your parquet lives elsewhere
DATA_PATH = Path("/Users/hanatjendrawasi/Desktop/DSC180B-Q2/preprocessing/query2/train_query2_one_row_per_guid.parquet")

OUT_DIR = Path("/Users/hanatjendrawasi/Desktop/DSC180B-Q2/preprocessing/query2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VENDOR_MAP_PATH = OUT_DIR / "query2_vendor_map.json"
MODEL_MAP_PATH = OUT_DIR / "query2_model_map.json"


def _clean_str(s: pd.Series) -> pd.Series:
    # normalize categorical strings to avoid fake-unique values due to whitespace
    return s.astype(str).str.strip()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. "
            "Update DATA_PATH to point to your one-row-per-guid parquet."
        )

    df = pd.read_parquet(DATA_PATH)

    # Basic checks
    required = {"guid", "vendor", "model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Clean strings
    df["vendor"] = _clean_str(df["vendor"])
    df["model"] = _clean_str(df["model"])

    # Distinct guid counts per category (THIS is the correct logic)
    vendor_counts = df.groupby("vendor")["guid"].nunique().sort_values(ascending=False)
    model_counts = df.groupby("model")["guid"].nunique().sort_values(ascending=False)

    keep_vendor = set(vendor_counts[vendor_counts >= VENDOR_MIN_GUIDS].index)
    # Frozen maps: every observed category -> itself or "Other"
    vendor_map = {v: (v if v in keep_vendor else "Other") for v in vendor_counts.index}
    
    MODEL_TOPK = 100
    keep_model = set(model_counts.head(MODEL_TOPK).index)
    model_map = {m: (m if m in keep_model else "Other") for m in model_counts.index}

    # Save mappings
    with open(VENDOR_MAP_PATH, "w") as f:
        json.dump(vendor_map, f, indent=2, sort_keys=True)

    with open(MODEL_MAP_PATH, "w") as f:
        json.dump(model_map, f, indent=2, sort_keys=True)

    # Report summary
    print("\n=== INPUT ===")
    print("rows:", len(df))
    print("distinct guids:", df["guid"].nunique())

    print("\n=== VENDOR COUNTS (top 15) ===")
    print(vendor_counts.head(15))

    print("\n=== MODEL COUNTS (top 15) ===")
    print(model_counts.head(15))

    print("\n=== BUCKETING SUMMARY ===")
    print(f"vendor unique: {len(vendor_counts)} | kept (>= {VENDOR_MIN_GUIDS}): {len(keep_vendor)}")
    print(f"model  unique: {len(model_counts)}  | kept (>= {MODEL_MIN_GUIDS}): {len(keep_model)}")

    vendor_other_rate = (df["vendor"].map(vendor_map) == "Other").mean()
    model_other_rate = (df["model"].map(model_map) == "Other").mean()

    print(f"vendor % mapped to Other: {vendor_other_rate:.3%}")
    print(f"model  % mapped to Other: {model_other_rate:.3%}")

    print("\n=== OUTPUT FILES ===")
    print("saved:", VENDOR_MAP_PATH)
    print("saved:", MODEL_MAP_PATH)


if __name__ == "__main__":
    main()

