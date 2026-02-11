from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA = Path("/Users/hanatjendrawasi/Desktop/DSC180B-Q2/preprocessing/query2/train_query2_bucketed.parquet")
OUT  = Path("/Users/hanatjendrawasi/Desktop/DSC180B-Q2/preprocessing/query2/query2_clip_bounds.json")

def p99(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.percentile(x, 99))

def main():
    df = pd.read_parquet(DATA)

    df["delta_pos"] = np.clip(df["delta_bytes"].astype(float).to_numpy(), 0, None)
    df["delta_neg"] = np.clip((-df["delta_bytes"].astype(float)).to_numpy(), 0, None)

    clip = {
        "nrs": p99(df["nrs"]),
        "sent_bytes": p99(df["sent_bytes"]),
        "received_bytes": p99(df["received_bytes"]),
        "delta_pos": p99(df["delta_pos"]),
        "delta_neg": p99(df["delta_neg"]),
    }

    OUT.write_text(json.dumps(clip, indent=2))
    print("Saved clip bounds:", OUT)
    print(clip)

if __name__ == "__main__":
    main()

