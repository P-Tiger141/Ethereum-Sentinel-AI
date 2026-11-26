import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

DATA = "data/raw/allnodes_enhancement.csv"

features = joblib.load("models/feature_columns.joblib")
rf = joblib.load("models/random_forest.joblib")
xgb = joblib.load("models/xgboost.joblib")

df = pd.read_csv(DATA)
accounts = df.drop_duplicates("address_id")
X = accounts[features].fillna(0)

os.makedirs("results/explainability", exist_ok=True)

sample = X.sample(min(500, len(X)), random_state=42)

for name, model in [("random_forest", rf), ("xgboost", xgb)]:
    print(f"\nGenerating SHAP explanation: {name}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        values = shap_values[1]
    elif len(shap_values.shape) == 3:
        values = shap_values[:, :, 1]
    else:
        values = shap_values

    plt.figure()
    shap.summary_plot(
        values, sample,
        feature_names=features,
        show=False,
        max_display=11
    )
    plt.tight_layout()
    plt.savefig(
        f"results/explainability/{name}_summary.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    importance = pd.DataFrame({
        "feature": features,
        "mean_absolute_shap": abs(values).mean(axis=0)
    }).sort_values("mean_absolute_shap", ascending=False)

    importance.to_csv(
        f"results/explainability/{name}_importance.csv",
        index=False
    )

    print(importance.to_string(index=False))

print("\nSHAP EXPLANATIONS COMPLETE")
print("Saved in results/explainability/")
