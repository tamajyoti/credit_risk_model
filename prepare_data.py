"""
Part 1: Data Engineering Pipeline to analyze data and create training data set
 
Creates the training_set.csv with one row per customer,
Based on the model analysis of pre-trained model.joblib which shows 9 features it creates the 9
used in the model building and also creates some features for future model iterations.

"""
 
import re
import warnings
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
warnings.filterwarnings("ignore")
 
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
ARTIFACTS   = BASE_DIR / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# 1. LOAD DATA

print("STEP 1: Loading data")
 
tx     = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["txn_timestamp"])
labels = pd.read_csv(DATA_DIR / "labels.csv")
 
print(f"Transactions : {tx.shape[0]} rows x {tx.shape[1]} cols")
print(f"Labels       : {labels.shape[0]} rows x {labels.shape[1]} cols")

# 2. EXPLORE THE DATA

print("STEP 2: Exploring data for data quality")
# check for nulls
nulls = tx.isnull().sum()
print(f"Nulls: {'none' if not nulls.any() else nulls[nulls > 0]}")

# check for duplicates
dupe_rows = tx.duplicated().sum()
dupe_ids  = tx["transaction_id"].duplicated().sum()
print(f"Duplicate rows:            {dupe_rows}")
print(f"Duplicate transaction IDs: {dupe_ids}")
if dupe_rows:
    tx = tx.drop_duplicates()
    print("Dropped duplicate rows")

# Checking for data consistency: debits should be negative, credits positive
mismatched = tx[
    ((tx["txn_type"] == "debit")  & (tx["amount"] > 0)) |
    ((tx["txn_type"] == "credit") & (tx["amount"] < 0))
]
print(f"  Amount/type mismatches:    {len(mismatched)}")
if len(mismatched):
    # flip the sign in case there is mismatched signs
    # can be removed or highlighted too based on business considerations
    tx.loc[mismatched.index, "amount"] = -tx.loc[mismatched.index, "amount"]
    print("Signs corrected to match txn_type")

print(f"Amount range: [{tx['amount'].min():.2f}, {tx['amount'].max():.2f}]")
print(f"Date range:   {tx['txn_timestamp'].min()} -> {tx['txn_timestamp'].max()}")

# check nulls duplicates and default value for labels data
print("\n Labels Data")
lbl_nulls = labels.isnull().sum()
print(f"Nulls: {'none' if not lbl_nulls.any() else lbl_nulls[lbl_nulls > 0]}")
print(f"Duplicate customer IDs: {labels['customer_id'].duplicated().sum()}")
print(f"Default rate: {labels['defaulted_within_90d'].mean():.1%}")


# find non mapped data either from transactions side or labels side
txn_custs  = set(tx["customer_id"])
lbl_custs  = set(labels["customer_id"])
only_txn   = txn_custs - lbl_custs
only_label = lbl_custs - txn_custs
print(f"\n Join coverage for data on both txns and lbls")
print(f"  In both:               {len(txn_custs & lbl_custs)}")
if only_txn:
    print(f"Txn-only (dropped):    {only_txn}")
if only_label:
    print(f"Label-only (dropped):  {only_label}")

# since we are creating training data after analysis removing non mapped ones as they will not have either
# dependent label output or independent features


# 3. CLEANING OF DESCRIPTION TEXT

print("STEP 3: Text processing on description column")
def clean_text(s: str) -> str:
    """Lowercase, strip punctuation/digits, collapse whitespace."""
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+",       " ", s).strip()
    return s
 
tx["clean_desc"] = tx["description"].fillna("").apply(clean_text)
 
print("Sample cleaned descriptions:")
for _, row in tx[["description", "clean_desc"]].drop_duplicates().head(7).iterrows():
    print(f"    '{row['description']}' -> '{row['clean_desc']}'")
 

# 4. EXTRACTION OF FEATURES
print("STEP 4: Feature engineering")
agg = (
    tx.groupby("customer_id")
    .agg(
        txn_count    = ("transaction_id", "count"),
        total_debit  = ("amount", lambda x: x[x < 0].sum()),
        total_credit = ("amount", lambda x: x[x > 0].sum()),
        avg_amount   = ("amount", "mean"),
        all_desc     = ("clean_desc", lambda x: " ".join(x)),
    )
    .reset_index()
)

# ── Model-expected keyword flags (must match model training exactly) ──
# since we dont have enough data to train and .joblib also shows number of features and not the name
# looked into the app.py added in the project to check which features are used in model
MODEL_KEYWORDS = ["rent", "netflix", "tesco", "payroll", "bonus"]
for kw in MODEL_KEYWORDS:
    agg[f"kw_{kw}"] = agg["all_desc"].str.contains(rf"\b{kw}\b").astype(int)

# some extra engineered feature based on the txn data
# net cashflow shows as solvency signal
agg["net_cashflow"] = agg["total_credit"] + agg["total_debit"] 
# debit credit ratio shows how much is savings close to 1 or >1 is risky as either in debt or entire money in expended
agg["debit_to_credit_ratio"] = np.where(
    agg["total_credit"] > 0,
    agg["total_debit"].abs() / agg["total_credit"],
    np.nan,
) 

# max single debit added as a expense feature to understand unexpected shocks like large rent component to overall income
max_debit = (
    tx[tx["amount"] < 0]
    .groupby("customer_id")["amount"]
    .min()
    .abs()
    .rename("max_single_debit")
)
agg = agg.merge(max_debit, on="customer_id", how="left")
agg["max_single_debit"] = agg["max_single_debit"].fillna(0)
agg["has_large_single_debit"] = (
    (agg["max_single_debit"] / agg["total_credit"].replace(0, np.nan)) > 0.5
).astype(int)

# Intermediate text column deleted
agg = agg.drop(columns=["all_desc"])

print("  Features created:")
for col in agg.columns[1:]:
    print(f"    - {col}")


# 5. BUILD THE TRAINING DATASET

print("STEP 5: Build training dataset")

df = agg.merge(labels, on="customer_id", how="inner")
 
cols = (["customer_id"]
        + [c for c in df.columns if c not in ("customer_id", "defaulted_within_90d")]
        + ["defaulted_within_90d"])
df = df[cols]
 
print(f"Final shape: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"Default rate: {df['defaulted_within_90d'].mean():.1%}")
 
df.to_csv(ARTIFACTS / "training_set.csv", index=False)
print(f"\n  Saved in artifacts/training_set.csv")

# 6. DATA SCALING AND STORAGE OF MEAN AND STD
print("STEP 6: Fit and save StandardScaler")

import json
import joblib as _joblib
from sklearn.preprocessing import StandardScaler
 
FLOAT_COLS = ["txn_count", "total_debit", "total_credit", "avg_amount"]
 
scaler = StandardScaler()
scaler.fit(df[FLOAT_COLS])

_joblib.dump(scaler, ARTIFACTS / "scaler.joblib")
print("Saved artifacts/scaler.joblib")

scaler_meta = {
    "float_cols": FLOAT_COLS,
    "mean":  {col: round(float(m), 6) for col, m in zip(FLOAT_COLS, scaler.mean_)},
    "std":   {col: round(float(s), 6) for col, s in zip(FLOAT_COLS, scaler.scale_)},
    "note": (
        "Fitted on training_set.csv to be applied at inference time as: "
        "x_scaled = (x - mean) / std for each float feature. "
        "Keyword flags (kw_*) are passed through unchanged."
    ),
}
with open(ARTIFACTS / "scaler_params.json", "w") as f:
    json.dump(scaler_meta, f, indent=2)
print("Saved artifacts/scaler_params.json")
 
print("  Scaler parameters:")
print(f"  {'feature':<16} {'mean':>12} {'std':>12}")
print(f"  {'-'*42}")
for col in FLOAT_COLS:
    print(f"  {col:<16} {scaler_meta['mean'][col]:>12.6f} {scaler_meta['std'][col]:>12.6f}")

# 7. DATA VISUALIZATIONS FOR EDA

print("STEP 7: EDA visualisations")

colors  = {0: "#4A9EFF", 1: "#FF6B6B"}
lbl_map = {0: "No Default", 1: "Default"}
kw_cols = [f"kw_{k}" for k in MODEL_KEYWORDS]
 
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Credit Decisioning Data Analyis", fontsize=14, fontweight="bold", y=0.98)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
 
# defaulted customer w.r.t net cash flow
ax1 = fig.add_subplot(gs[0, 0])
for label, grp in df.groupby("defaulted_within_90d"):
    ax1.bar([lbl_map[label]], [grp["net_cashflow"].mean()],
            color=colors[label], edgecolor="white", width=0.4)
ax1.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax1.set_title("Avg Net Cashflow by Default Status", fontsize=11)
ax1.set_ylabel("Net Cashflow")
 
# defaulted customer w.r.t debit to credit ratio 
ax2 = fig.add_subplot(gs[0, 1])
for label, grp in df.groupby("defaulted_within_90d"):
    ax2.bar([lbl_map[label]], [grp["debit_to_credit_ratio"].mean()],
            color=colors[label], edgecolor="white", width=0.4)
ax2.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", label="Ratio = 1")
ax2.set_title("Avg Debit/Credit Ratio by Default Status", fontsize=11)
ax2.set_ylabel("Ratio")
ax2.legend(fontsize=8)
 
# For each flag we want to check the distribution of default vs no default based on customers who qualify
ax3 = fig.add_subplot(gs[1, :])
flag_counts = df.groupby("defaulted_within_90d")[kw_cols].sum()
flag_pct    = flag_counts.div(flag_counts.sum(axis=0), axis=1) * 100
 
x     = range(len(kw_cols))
width = 0.35
for i, (label, row) in enumerate(flag_pct.iterrows()):
    offset = -width/2 + i * width
    bars = ax3.bar([xi + offset for xi in x], row.values, width,
                   label=lbl_map[label], color=colors[label], edgecolor="white")
    # Get each bar with its percentage
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1,
                f"{h:.1f}%",
                ha="center", va="bottom", fontsize=7.5
            )
ax3.set_xticks(list(x))
ax3.set_xticklabels([c.replace("kw_", "") for c in kw_cols], rotation=20)
ax3.set_title("Default Share per Keyword Flag\n(% of customers with that flag, by default status)", fontsize=11)
ax3.set_ylabel("% of customers with flag")
ax3.set_ylim(0, 115)
ax3.legend()
 
plt.savefig(ARTIFACTS / "eda.png", dpi=150, bbox_inches="tight")
print("Saved in artifacts/eda.png")
print("All Complete")