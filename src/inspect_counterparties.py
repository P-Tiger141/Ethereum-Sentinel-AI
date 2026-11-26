import pandas as pd

df = pd.read_csv("data/raw/allnodes_enhancement.csv")

account = df.groupby("address_id").agg(
    row_count=("sequence", "size"),
    unique_destinations=("To", "nunique"),
    total_transactions=("total_transactions", "first"),
    initiated=("transactions_initiated", "first"),
    received=("transactions_received", "first"),
    label=("label", "first")
)

account["repeated_destination_rows"] = (
    account["row_count"] - account["unique_destinations"]
)

print("=" * 65)
print("COUNTERPARTY / ROW STRUCTURE")
print("=" * 65)

print("\nAccounts with repeated destinations:",
      (account["repeated_destination_rows"] > 0).sum())

print("Total repeated destination rows:",
      account["repeated_destination_rows"].sum())

print("\nRow count equals unique destinations:",
      (account["row_count"] == account["unique_destinations"]).sum(),
      "/", len(account))

print("\nTotal transactions equals unique destinations:",
      (account["total_transactions"] == account["unique_destinations"]).sum(),
      "/", len(account))

print("\nCorrelation:")
print(account[
    ["row_count", "unique_destinations", "total_transactions"]
].corr().round(3).to_string())

print("\nSelected accounts:")
print(account.loc[
    [6, 13, 54, 56, 3896]
].to_string())

print("\nRepeated destination examples:")
repeated = df[
    df.duplicated(["address_id", "To"], keep=False)
].sort_values(["address_id", "To"])

print(repeated[
    ["address_id", "To", "total_transactions", "label"]
].head(20).to_string(index=False))

