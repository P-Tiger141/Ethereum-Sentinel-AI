import pandas as pd

df = pd.read_csv("data/raw/allnodes_enhancement.csv")

print("=" * 65)
print("RAW DATASET STRUCTURE")
print("=" * 65)

print("\nCOLUMN TYPES:")
print(df.dtypes.to_string())

print("\nFIRST 15 COMPLETE ROWS:")
print(df.head(15).to_string(index=False))

print("\nLAST 10 COMPLETE ROWS:")
print(df.tail(10).to_string(index=False))

print("\nLABEL TRANSITION BY ADDRESS ID:")
groups = df.groupby("address_id").agg(
    label=("label", "first"),
    rows=("label", "size"),
    first_from=("From", "min"),
    last_from=("From", "max"),
    destination_count=("To", "nunique")
)

print(groups.head(20).to_string())

print("\nVALUE AND TIMESTAMP COLUMN SAMPLE:")
cols = [
    "address_id", "value1", "value2", "value3", "value4",
    "timestamps1", "timestamps2", "timestamps3", "timestamps4"
]
print(df[cols].head(15).to_string(index=False))

print("\nLABEL COUNTS BY ADDRESS:")
print(groups["label"].value_counts().to_string())

