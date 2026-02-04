from pathlib import Path
import json
import numpy as np
import pandas as pd

# INPUT
IN_PATH = Path("/Users/hanatjendrawasi/Desktop/DSC180B-Q2/preprocessing/query2/train_query2_one_row_per_guid.parquet")

# FROZEN MAPS (from your make_buckets script)
MAP_DIR = Path("/Users/hanatjendrawasi/Desktop/DSC180B-Q2/preprocessing/query2")
VENDOR_MAP_PATH = MAP_DIR / "query2_vendor_map.json"
MODEL_MAP_PATH  = MAP_DIR / "query2_model_map.json"

# OUTPUT
OUT_PATH = Path("/Users/hanatjendrawasi/Desktop/DSC180B-Q2/preprocessing/query2/train_query2_bucketed.parquet")

def clean_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def main():
    df = pd.read_parquet(IN_PATH)

    vendor_map = json.loads(VENDOR_MAP_PATH.read_text())
    model_map  = json.loads(MODEL_MAP_PATH.read_text())

    df["vendor"] = clean_str(df["vendor"])
    df["model"]  = clean_str(df["model"])
    df["chassistype"] = clean_str(df["chassistype"])
    df["os"] = clean_str(df["os"])

    # Apply frozen buckets (unknown/new categories -> Other)
    df["vendor_bucket"] = df["vendor"].map(lambda v: vendor_map.get(v, "Other"))
    df["model_bucket"]  = df["model"].map(lambda m: model_map.get(m, "Other"))

    # sanity
    print("rows:", len(df))
    print("vendor_bucket %Other:", (df["vendor_bucket"] == "Other").mean())
    print("model_bucket  %Other:", (df["model_bucket"] == "Other").mean())
    print("vendor_bucket top10:\n", df["vendor_bucket"].value_counts().head(10))
    print("model_bucket top10:\n", df["model_bucket"].value_counts().head(10))

    df.to_parquet(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()

