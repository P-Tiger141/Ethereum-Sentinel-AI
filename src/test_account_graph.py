import pandas as pd
import networkx as nx

df = pd.read_csv("data/raw/allnodes_enhancement.csv")

G = nx.DiGraph()
G.add_edges_from(zip(df["address_id"], df["To"]))

print("=" * 60)
print("ALTERNATIVE ACCOUNT-LEVEL GRAPH")
print("=" * 60)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Self-loops:", nx.number_of_selfloops(G))
print("Weakly connected components:",
      nx.number_weakly_connected_components(G))

print("\nIN-DEGREE DISTRIBUTION:")
print(pd.Series(dict(G.in_degree()).values())
      .value_counts().sort_index().to_string())

print("\nOUT-DEGREE DISTRIBUTION:")
print(pd.Series(dict(G.out_degree()).values())
      .value_counts().sort_index().to_string())

print("\nTOP 10 NODES BY IN-DEGREE:")
print(sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:10])

print("\nTOP 10 NODES BY OUT-DEGREE:")
print(sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:10])

print("\nLABELLED ACCOUNT NODES:", df["address_id"].nunique())
print("DESTINATION NODES:", df["To"].nunique())
print("TOTAL UNIQUE ACCOUNT IDS:",
      len(set(df["address_id"]) | set(df["To"])))

