import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA = "/Users/hanatjendrawasi/Desktop/DSC180B-Q2/training/train_query2_one_row_per_guid.parquet"
df = pd.read_parquet(DATA)

print("\n=== SHAPE ===")
print("rows:", len(df))
print("distinct guids:", df["guid"].nunique())

# -----------------------
# 1) Missingness + sanity
# -----------------------
cols = ["nrs","received_bytes","sent_bytes","delta_bytes","chassistype","vendor","model","ram","os","number_of_cores"]
print("\n=== NULL RATES ===")
print((df[cols].isna().mean() * 100).sort_values(ascending=False))

print("\n=== BASIC CHECKS ===")
print("sent>received rate:", (df["sent_bytes"] > df["received_bytes"]).mean())
print("min/max nrs:", df["nrs"].min(), df["nrs"].max())

# -----------------------
# 2) Distribution for clipping (bytes are heavy-tailed)
# -----------------------
def pct_report(x, name):
    x = x.replace([np.inf,-np.inf], np.nan).dropna().astype(float)
    p = np.percentile(x, [0,1,5,25,50,75,95,99,100])
    print(f"\n=== {name} percentiles ===")
    print("min/p1/p5/p25/p50/p75/p95/p99/max:", p)
    return p

pct_sent = pct_report(df["sent_bytes"], "sent_bytes")
pct_recv = pct_report(df["received_bytes"], "received_bytes")
pct_delta = pct_report(df["delta_bytes"], "delta_bytes")

# Plots (use log1p because bytes are huge)
for col in ["sent_bytes","received_bytes","delta_bytes"]:
    plt.figure()
    plt.hist(np.log1p(df[col].clip(lower=0)), bins=100)
    plt.title(f"log1p({col}) histogram")
    plt.xlabel(f"log1p({col})")
    plt.ylabel("count")
    plt.savefig(f"log_hist_{col}.png", dpi=200, bbox_inches="tight")
    plt.close()

# -----------------------
# 3) Categorical tail heaviness (for bucketing)
# -----------------------
def top_cats(col, topn=20):
    vc = df[col].astype(str).value_counts(dropna=False)
    print(f"\n=== {col} unique:", df[col].nunique(dropna=False), "===")
    print(vc.head(topn))
    return vc

top_cats("vendor", 20)
top_cats("model", 20)
top_cats("chassistype", 20)
top_cats("os", 20)

# Distinct guid count per category (more stable than row count)
for col in ["vendor","model","chassistype","os"]:
    g = df.groupby(col)["guid"].nunique().sort_values(ascending=False)
    print(f"\n=== distinct guids by {col} (top 20) ===")
    print(g.head(20))

# -----------------------
# 4) Choose clipping bounds (recommend p99 on log-scale targets)
# -----------------------
print("\nSUGGESTED CLIP BOUNDS (raw space):")
print("sent_bytes clip ~ p99:", pct_sent[7])
print("received_bytes clip ~ p99:", pct_recv[7])
print("delta_bytes clip ~ p99:", pct_delta[7])

