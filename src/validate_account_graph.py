import pandas as pd
import networkx as nx

df = pd.read_csv("data/raw/allnodes_enhancement.csv")

labels = (
    df[["address_id", "label"]]
    .drop_duplicates()
    .set_index("address_id")["label"]
    .to_dict()
)

edges = set(zip(df["address_id"], df["To"]))

self_loops = [(u, v) for u, v in edges if u == v]
reciprocal = sum(1 for u, v in edges if u != v and (v, u) in edges) // 2

same_label = 0
different_label = 0
missing_label = 0

for u, v in edges:
    if u not in labels or v not in labels:
        missing_label += 1
    elif labels[u] == labels[v]:
        same_label += 1
    else:
        different_label += 1

G = nx.DiGraph()
G.add_edges_from(edges)

print("=" * 60)
print("ACCOUNT GRAPH VALIDATION")
print("=" * 60)

print("Unique edges:", len(edges))
print("Duplicate edge rows:", len(df) - len(edges))
print("Self-loops:", len(self_loops))
print("Reciprocal edge pairs:", reciprocal)

print("\nEDGE LABEL RELATIONSHIPS:")
print("Same-label edges:", same_label)
print("Different-label edges:", different_label)
print("Edges with missing endpoint labels:", missing_label)

print("\nLABEL MIXING:")
print("Phishing-to-legitimate edges:",
      sum(1 for u, v in edges
          if labels.get(u) == 1 and labels.get(v) == 0))
print("Legitimate-to-phishing edges:",
      sum(1 for u, v in edges
          if labels.get(u) == 0 and labels.get(v) == 1))

print("\nSAMPLE EDGES:")
for u, v in list(sorted(edges))[:15]:
    print(f"{u} ({labels.get(u)}) -> {v} ({labels.get(v)})")

print("\nDEGREE CHECK:")
print("Nodes with different in/out degree:",
      sum(1 for n in G.nodes()
          if G.in_degree(n) != G.out_degree(n)))

print("\nCONNECTED COMPONENT SIZES:")
sizes = sorted(
    [len(c) for c in nx.weakly_connected_components(G)],
    reverse=True
)
print(sizes[:15])

