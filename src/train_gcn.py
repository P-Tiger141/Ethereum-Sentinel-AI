import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

torch.manual_seed(42)

DEVICE = torch.device("cpu")
DATA_PATH = "data/processed/ethereum_graph.pt"

data = torch.load(DATA_PATH, weights_only=False).to(DEVICE)

indices = list(range(data.num_nodes))
labels = data.y.cpu().numpy()

train_idx, temp_idx = train_test_split(
    indices, test_size=0.4, random_state=42, stratify=labels
)

val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42,
    stratify=labels[temp_idx]
)

train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2)
        self.dropout = 0.5

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

model = GCN(data.num_node_features).to(DEVICE)

counts = torch.bincount(data.y[train_mask], minlength=2).float()
weights = counts.sum() / (2 * counts)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.01, weight_decay=0.005
)

best_score = -1
best_state = None
patience = 30
wait = 0

for epoch in range(1, 301):
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)
    loss = criterion(logits[train_mask], data.y[train_mask])

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = model(data.x, data.edge_index).softmax(dim=1)[:, 1]
        val_y = data.y[val_mask].cpu().numpy()
        val_probs = probs[val_mask].cpu().numpy()

        score = average_precision_score(val_y, val_probs)

    if score > best_score:
        best_score = score
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Val PR-AUC {score:.4f}")

    if wait >= patience:
        print("Early stopping at epoch", epoch)
        break

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    probabilities = model(data.x, data.edge_index).softmax(dim=1)[:, 1]
    predictions = (probabilities >= 0.5).long()

y_true = data.y[test_mask].cpu().numpy()
y_pred = predictions[test_mask].cpu().numpy()
y_prob = probabilities[test_mask].cpu().numpy()

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0
)

metrics = {
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc_score(y_true, y_prob)),
    "pr_auc": float(average_precision_score(y_true, y_prob)),
    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    "best_validation_pr_auc": float(best_score)
}

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

torch.save(model.state_dict(), "models/gcn_model.pt")

with open("results/gcn_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n========== GCN TEST RESULTS ==========")
for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nSaved: models/gcn_model.pt")
print("Saved: results/gcn_metrics.json")
