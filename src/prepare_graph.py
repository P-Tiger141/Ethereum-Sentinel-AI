import os
import pandas as pd
import numpy as np
import torch

from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

INPUT = "data/raw/allnodes_enhancement.csv"
OUTPUT = "data/processed/ethereum_graph.pt"

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv(INPUT)

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

accounts = df.groupby("address_id").first().sort_index()

scaler = StandardScaler()
x = scaler.fit_transform(accounts[features].fillna(0))
x = torch.tensor(x, dtype=torch.float32)

labels = torch.tensor(
    accounts["label"].values,
    dtype=torch.long
)

edges = df[["address_id", "To"]].drop_duplicates()

node_ids = accounts.index.to_list()
node_map = {node_id: i for i, node_id in enumerate(node_ids)}

edge_index = torch.tensor(
    [
        [node_map[u] for u in edges["address_id"]],
        [node_map[v] for v in edges["To"]]
    ],
    dtype=torch.long
)

data = Data(
    x=x,
    edge_index=edge_index,
    y=labels
)

data.num_nodes = len(accounts)

torch.save(data, OUTPUT)
torch.save(scaler, "models/feature_scaler.pt")

print("=" * 55)
print("GRAPH DATASET CREATED")
print("=" * 55)
print("Nodes:", data.num_nodes)
print("Features:", data.num_node_features)
print("Edges:", data.edge_index.shape[1])
print("Phishing:", int((labels == 1).sum()))
print("Legitimate:", int((labels == 0).sum()))
print("Saved:", OUTPUT)
