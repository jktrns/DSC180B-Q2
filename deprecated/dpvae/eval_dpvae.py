import pandas as pd
import numpy as np

real = pd.read_parquet("dpvae_train_guid.parquet")
synth = pd.read_parquet("dpvae_synth.parquet")

cat_cols = ["os", "chassistype", "vendor"]
num_cols = [
    "ram","number_of_cores","received_bytes","sent_bytes","net_nr_samples",
    "frgnd_total_duration_sec","frgnd_total_nrs","frgnd_rows",
    "userwait_total_duration_ms","userwait_rows"
]

# ---- CATEGORICAL: top proportions ----
for c in cat_cols:
    print("\n====", c, "TOP 10 ====")
    r = real[c].fillna("UNKNOWN").value_counts(normalize=True).head(10)
    s = synth[c].fillna("UNKNOWN").value_counts(normalize=True).head(10)
    out = pd.concat([r.rename("real"), s.rename("synth")], axis=1).fillna(0)
    print(out)

# ---- NUMERIC: summary stats + rough “shape” ----
def summarize(x):
    x = pd.to_numeric(x, errors="coerce").fillna(0)
    return pd.Series({
        "mean": x.mean(),
        "p50": x.quantile(0.50),
        "p90": x.quantile(0.90),
        "p99": x.quantile(0.99),
        "max": x.max(),
        "zeros_%": (x == 0).mean(),
    })

print("\n\n==== NUMERIC SUMMARY ====")
rows = []
for c in num_cols:
    r = summarize(real[c]).rename(lambda k: f"real_{k}")
    s = summarize(synth[c]).rename(lambda k: f"synth_{k}")
    rows.append(pd.concat([pd.Series({"col": c}), r, s]))
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))

