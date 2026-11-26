import pandas as pd
import networkx as nx

df = pd.read_csv("data/raw/allnodes_enhancement.csv")

G = nx.DiGraph()
G.add_edges_from(zip(df["From"], df["To"]))

in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

print("=" * 55)
print("DEGREE DISTRIBUTION")
print("=" * 55)

print("\nIn-degree frequency:")
print(pd.Series(list(in_degree.values())).value_counts().sort_index().to_string())

print("\nOut-degree frequency:")
print(pd.Series(list(out_degree.values())).value_counts().sort_index().to_string())

print("\nSelf-loops:", nx.number_of_selfloops(G))
print("Unique directed edges:", len(set(zip(df["From"], df["To"]))))
print("Original rows:", len(df))

print("\nTOP 10 NODES BY OUT-DEGREE:")
print(sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10])

print("\nTOP 10 NODES BY IN-DEGREE:")
print(sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10])

print("\nADDRESS ID 3 CONNECTIONS:")
print(df[df["address_id"] == 3][["From", "To", "label"]].to_string(index=False))

