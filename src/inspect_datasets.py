import os
import pandas as pd

print("=" * 60)
print("ENHANCED DATASET")
print("=" * 60)

df = pd.read_csv("data/raw/allnodes_enhancement.csv")

print("Columns:", df.columns.tolist())

for col in ["From", "To", "address_id"]:
    print(f"\n{col} samples:")
    print(df[col].dropna().astype(str).head(5).tolist())

print("\nFrom data type:", df["From"].dtype)
print("To data type:", df["To"].dtype)

print("\n" + "=" * 60)
print("TRANSACTION ARCHIVE")
print("=" * 60)

archive = "data/raw/transactions"

files = []
for root, dirs, filenames in os.walk(archive):
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            files.append(os.path.join(root, filename))

sample = files[0]
print("Sample filename:", os.path.basename(sample))

tx = pd.read_csv(sample)

print("\nTransaction columns:", tx.columns.tolist())
print("\nFirst 5 transactions:")
print(tx.head().to_string(index=False))

print("\nTransaction data types:")
print(tx.dtypes)

