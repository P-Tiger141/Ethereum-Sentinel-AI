import pandas as pd

df = pd.read_csv("data/raw/allnodes_enhancement.csv")

print("UNIQUE NODE COUNTS")
print("Unique From:", df["From"].nunique())
print("Unique To:", df["To"].nunique())
print("Unique address_id:", df["address_id"].nunique())

print("\nNODE ID RANGE")
print("From:", df["From"].min(), "to", df["From"].max())
print("To:", df["To"].min(), "to", df["To"].max())

print("\nADDRESS ID TO NODE EXAMPLES")

mapping = df[["address_id", "From", "To", "label"]].drop_duplicates()

for address_id in mapping["address_id"].drop_duplicates().head(10):
    rows = mapping[mapping["address_id"] == address_id]
    print("\nAddress ID:", address_id)
    print(rows.head(5).to_string(index=False))

print("\nNODE ID EXAMPLES")
nodes = pd.concat([df["From"], df["To"]]).drop_duplicates()
print(nodes.head(20).tolist())

print("\nLABEL DISTRIBUTION PER ADDRESS ID")
print(df.drop_duplicates("address_id")["label"].value_counts())
