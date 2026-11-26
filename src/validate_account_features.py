import pandas as pd

df = pd.read_csv("data/raw/allnodes_enhancement.csv")

print("=" * 60)
print("FEATURE CONSISTENCY CHECK")
print("=" * 60)

features = [
    "total_transactions",
    "transactions_initiated",
    "transactions_received",
    "average_transaction_value",
    "total_transaction_value",
    "average_outgoing_value",
    "average_incoming_value",
    "total_outgoing_value",
    "total_incoming_value",
    "last_outgoing_transaction_value",
    "last_incoming_transaction_value"
]

for col in features:
    counts = df.groupby("address_id")[col].nunique()
    print(f"{col}: inconsistent address groups = {(counts > 1).sum()}")

print("\nTRANSACTION COUNT VALIDATION:")
account = df.drop_duplicates("address_id")

print("Total transactions equals row count:",
      (account["total_transactions"] ==
       df.groupby("address_id").size().reindex(account["address_id"]).values).sum(),
      "/", len(account))

print("\nINITIATED + RECEIVED:")
print("Groups where initiated + received != total:")
print(
    ((account["transactions_initiated"] +
      account["transactions_received"]) !=
     account["total_transactions"]).sum()
)

print("\nSEQUENCE CHECK:")
print("Unique sequence:", df["sequence"].nunique())
print("Sequence continuous:",
      sorted(df["sequence"].tolist()) == list(range(len(df))))

