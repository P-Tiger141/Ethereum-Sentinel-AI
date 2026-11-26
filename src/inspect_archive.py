import os
import pandas as pd

df = pd.read_csv("data/raw/allnodes_enhancement.csv")
labels = df.drop_duplicates("address_id")

known = set(labels["From"].dropna().astype(str).str.lower())
known.update(labels["To"].dropna().astype(str).str.lower())

archive = "data/raw/transactions"

files = []

for root, dirs, filenames in os.walk(archive):
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            files.append(os.path.join(root, filename))

addresses = {
    os.path.splitext(os.path.basename(f))[0].lower()
    for f in files
}

matched = addresses & known

print("Archive CSV files:", len(files))
print("Addresses in enhanced From/To:", len(known))
print("Matching archive addresses:", len(matched))
print("Unmatched archive addresses:", len(addresses - known))

print("\nFirst 5 archive paths:")
for path in files[:5]:
    print(path)

print("\nSample matching addresses:")
for address in sorted(matched)[:10]:
    print(address)

print("\nSample unmatched addresses:")
for address in sorted(addresses - known)[:10]:
    print(address)
