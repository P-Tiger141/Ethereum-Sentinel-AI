import pandas as pd

df = pd.read_csv("data/raw/allnodes_enhancement.csv")

print("ADDRESS GROUP SIZE")
print(df.groupby("address_id").size().describe())

print("\nROWS PER LABEL")
print(df.groupby("label")["address_id"].nunique())

print("\nDOES FROM EQUAL ADDRESS_ID?")
print("Matches:", (df["From"] == df["address_id"]).sum(), "/", len(df))

print("\nDOES TO EQUAL ADDRESS_ID?")
print("Matches:", (df["To"] == df["address_id"]).sum(), "/", len(df))

print("\nSAMPLE ADDRESS GROUPS")
for aid in [1, 2, 3, 4, 10, 100]:
    group = df[df["address_id"] == aid]
    if not group.empty:
        print(f"\naddress_id={aid}, label={group['label'].iloc[0]}")
        print(group[["From", "To"]].head(10).to_string(index=False))

print("\nFIRST 20 ADDRESS ID MAPPINGS")
print(
    df.groupby("address_id")
      .agg(
          from_min=("From", "min"),
          from_max=("From", "max"),
          rows=("From", "size"),
          unique_destinations=("To", "nunique"),
          label=("label", "first")
      )
      .head(20)
      .to_string()
)
