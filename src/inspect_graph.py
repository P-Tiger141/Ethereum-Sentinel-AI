import pandas as pd
import networkx as nx

df = pd.read_csv("data/raw/allnodes_enhancement.csv")

G = nx.DiGraph()

G.add_edges_from(zip(df["From"], df["To"]))

labels = (
    df[["address_id", "label"]]
    .drop_duplicates()
)

print("=" * 55)
print("GRAPH STRUCTURE")
print("=" * 55)

print("Nodes:", G.number_of_nodes())
print("Directed edges:", G.number_of_edges())
print("Weakly connected components:", nx.number_weakly_connected_components(G))

print("\nDEGREE STATISTICS")
print("Average in-degree:", sum(dict(G.in_degree()).values()) / G.number_of_nodes())
print("Average out-degree:", sum(dict(G.out_degree()).values()) / G.number_of_nodes())

print("\nLABEL MAPPING CHECK")

source_ids = set(df["From"])
destination_ids = set(df["To"])

print("Labeled address IDs:", len(labels))
print("Address IDs appearing as source nodes:", len(set(labels["address_id"]) & source_ids))
print("Address IDs appearing as destination nodes:", len(set(labels["address_id"]) & destination_ids))

print("\nSOURCE NODE SAMPLE")
print(df[["From", "To", "address_id", "label"]].head(10).to_string(index=False))

print("\nGRAPH INSPECTION COMPLETE")
