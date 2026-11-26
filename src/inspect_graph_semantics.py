import pandas as pd

df = pd.read_csv("data/raw/allnodes_enhancement.csv")

print("=" * 60)
print("GRAPH RELATIONSHIP ANALYSIS")
print("=" * 60)

# How many different address groups point to each destination?
destination_groups = df.groupby("To")["address_id"].nunique()

print("\nDESTINATION SHARING:")
print("Total destinations:", len(destination_groups))
print("Destinations shared by multiple address groups:",
      (destination_groups > 1).sum())
print("Maximum address groups sharing one destination:",
      destination_groups.max())

print("\nADDRESS GROUP SIZE DISTRIBUTION:")
group_sizes = df.groupby("address_id").size()
print(group_sizes.value_counts().sort_index().head(20).to_string())

print("\nSOURCE ID RANGE CHECK:")
ranges = df.groupby("address_id")["From"].agg(["min", "max", "count"])
ranges["range_size"] = ranges["max"] - ranges["min"] + 1
ranges["continuous"] = ranges["range_size"] == ranges["count"]

print("Groups with continuous source IDs:",
      ranges["continuous"].sum(), "/", len(ranges))

print("\nDESTINATIONS SHARED ACROSS LABELS:")
destination_labels = df.groupby("To")["label"].nunique()
print("Destinations connected to both labels:",
      (destination_labels > 1).sum())

print("\nTOP 10 MOST SHARED DESTINATIONS:")
print(destination_groups.sort_values(ascending=False).head(10).to_string())

print("\nSAMPLE SHARED DESTINATIONS:")
shared = destination_groups[destination_groups > 1].index[:10]

for destination in shared:
    subset = df[df["To"] == destination]
    print(
        f"Destination {destination}: "
        f"address groups={subset['address_id'].nunique()}, "
        f"labels={sorted(subset['label'].unique().tolist())}"
    )

