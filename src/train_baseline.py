import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from xgboost import XGBClassifier

DATA_PATH = "data/raw/allnodes_enhancement.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

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

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

df = df.drop_duplicates(subset="address_id").copy()
df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
df[FEATURES] = df[FEATURES].fillna(0)

X = df[FEATURES]
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDATA SPLIT")
print("Training addresses:", len(X_train))
print("Testing addresses:", len(X_test))
print("Training phishing:", int(y_train.sum()))
print("Testing phishing:", int(y_test.sum()))

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
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
}

results = {}

for name, model in models.items():
    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    roc = roc_auc_score(y_test, probabilities)
    pr = average_precision_score(y_test, probabilities)

    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC:  {pr:.4f}")

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join(MODEL_DIR, filename))

    results[name] = {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "phishing_recall": float(
            ((predictions == 1) & (y_test == 1)).sum() / (y_test == 1).sum()
        )
    }

joblib.dump(FEATURES, os.path.join(MODEL_DIR, "feature_columns.joblib"))

with open("results/baseline_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n" + "=" * 60)
print("BASELINE TRAINING COMPLETE")
print("=" * 60)
print(json.dumps(results, indent=4))
print("\nModels saved in:", MODEL_DIR)
