import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from xgboost import XGBClassifier

DATA_PATH = "data/raw/allnodes_enhancement.csv"

FEATURES = [
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

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates(subset="address_id").copy()

X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df["label"].astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = ["Random Forest", "XGBoost"]
results = {}

for name in models:
    fold_results = []

    print("\n" + "=" * 60)
    print(name.upper(), "- 5 FOLD CROSS VALIDATION")
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if name == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        else:
            model = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=4,
                random_state=42
            )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
            "pr_auc": average_precision_score(y_test, prob)
        }

        fold_results.append(metrics)

        print(
            f"Fold {fold}: "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f}"
        )

    summary = {}

    for metric in fold_results[0]:
        values = [r[metric] for r in fold_results]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

    results[name] = summary

    print("\nMEAN ± STANDARD DEVIATION")
    for metric, values in summary.items():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")

with open("results/cross_validation_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved: results/cross_validation_metrics.json")
