import pandas as pd

df = pd.read_csv("data/raw/allnodes_enhancement.csv")

grouped = df.groupby("address_id").agg(
    row_count=("sequence", "size"),
    total_transactions=("total_transactions", "first"),
    initiated=("transactions_initiated", "first"),
    received=("transactions_received", "first"),
    label=("label", "first")
)

grouped["difference"] = (
    grouped["total_transactions"] - grouped["row_count"]
)

mismatch = grouped[grouped["difference"] != 0]

print("=" * 60)
print("TRANSACTION COUNT DISCREPANCY ANALYSIS")
print("=" * 60)

print("Total mismatched accounts:", len(mismatch))
print("\nDifference distribution:")
print(mismatch["difference"].value_counts().sort_index().to_string())

print("\nMismatch examples:")
print(mismatch.head(20).to_string())

print("\nMismatch by label:")
print(mismatch["label"].value_counts().to_string())

print("\nMismatch statistics:")
print(mismatch["difference"].describe().to_string())

