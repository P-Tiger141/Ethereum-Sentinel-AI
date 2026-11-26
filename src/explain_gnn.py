import os
import torch
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
import torch.nn.functional as F
import torch.nn as nn

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
    "models/gcn_model.pt", map_location="cpu", weights_only=True
))
model.eval()

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=100),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config={
        "mode": "multiclass_classification",
        "task_level": "node",
        "return_type": "raw"
    }
)

probabilities = model(data.x, data.edge_index).softmax(dim=1)[:, 1]
phishing_nodes = torch.where(data.y == 1)[0]

selected = phishing_nodes[
    torch.argsort(probabilities[phishing_nodes], descending=True)
][:5]

os.makedirs("results/explainability", exist_ok=True)

feature_names = [
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

results = []

for node in selected.tolist():
    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        index=node
    )

    feature_scores = explanation.node_mask.mean(dim=0).detach().cpu().numpy()
    edge_scores = explanation.edge_mask.detach().cpu().numpy()

    for name, score in zip(feature_names, feature_scores):
        results.append({
            "node": node,
            "feature": name,
            "importance": float(score)
        })

    print(
        f"Node {node} | Actual label: {int(data.y[node])} | "
        f"Phishing probability: {probabilities[node].item():.4f} | "
        f"Important edges: {(edge_scores > 0.5).sum()}"
    )

pd.DataFrame(results).to_csv(
    "results/explainability/gnn_feature_importance.csv",
    index=False
)

print("\nGNN EXPLAINER COMPLETE")
print("Saved: results/explainability/gnn_feature_importance.csv")
