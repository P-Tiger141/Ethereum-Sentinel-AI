import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    roc_auc_score
)

torch.manual_seed(42)

class GCN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

data = torch.load(
    "data/processed/ethereum_graph.pt",
    weights_only=False
)

model = GCN(data.num_node_features)
model.load_state_dict(torch.load(
    "models/gcn_model.pt",
    map_location="cpu",
    weights_only=True
))
model.eval()

indices = list(range(data.num_nodes))
labels = data.y.numpy()

from sklearn.model_selection import train_test_split

_, temp = train_test_split(
    indices, test_size=0.4, random_state=42, stratify=labels
)
_, test_idx = train_test_split(
    temp, test_size=0.5, random_state=42,
    stratify=labels[temp]
)

test_idx = torch.tensor(test_idx)
y_true = data.y[test_idx].numpy()

results = {}

for drop_rate in [0.0, 0.1, 0.2, 0.3]:
    edge_count = data.edge_index.shape[1]

    keep = torch.rand(edge_count) >= drop_rate
    if drop_rate == 0:
        keep[:] = True

    modified_edges = data.edge_index[:, keep]

    with torch.no_grad():
        logits = model(data.x, modified_edges)
        probabilities = logits.softmax(dim=1)[:, 1]
        predictions = (probabilities >= 0.5).long()

    y_prob = probabilities[test_idx].numpy()
    y_pred = predictions[test_idx].numpy()

    metrics = {
        "remaining_edges": int(modified_edges.shape[1]),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }

    results[str(int(drop_rate * 100)) + "%"] = metrics

    print(
        f"Edge removal {drop_rate:.0%} | "
        f"Edges {metrics['remaining_edges']} | "
        f"F1 {metrics['f1']:.4f} | "
        f"PR-AUC {metrics['pr_auc']:.4f} | "
        f"ROC-AUC {metrics['roc_auc']:.4f}"
    )

os.makedirs("results", exist_ok=True)

with open("results/robustness_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved: results/robustness_metrics.json")
