
import os
import json
import joblib
import pandas as pd
import torch

from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.live import router as live_router

app = FastAPI(
    title="Ethereum Sentinel AI",
    description="Ethereum phishing detection and threat intelligence API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(live_router)

rf = joblib.load("models/random_forest.joblib")
xgb = joblib.load("models/xgboost.joblib")
features = joblib.load("models/feature_columns.joblib")


class PredictionRequest(BaseModel):
    features: Dict[str, float]


@app.get("/")
def home():
    return {
        "project": "Ethereum Sentinel AI",
        "status": "running"
    }


@app.get("/api/health")
def health():
    return {
        "api": "online",
        "random_forest": "loaded",
        "xgboost": "loaded"
    }


@app.get("/api/metrics")
def metrics():

    files = {
        "baseline": "results/baseline_metrics.json",
        "gcn": "results/gcn_metrics.json",
        "gat": "results/gat_metrics.json",
        "robustness": "results/robustness_metrics.json"
    }

    output = {}

    for name, path in files.items():
        if os.path.exists(path):
            with open(path) as f:
                output[name] = json.load(f)

    return output


@app.get("/api/explainability")
def explainability():

    files = {
        "random_forest": "results/explainability/random_forest_importance.csv",
        "xgboost": "results/explainability/xgboost_importance.csv"
    }

    output = {}

    for name, path in files.items():
        if os.path.exists(path):
            output[name] = pd.read_csv(path).to_dict(orient="records")

    return output


@app.post("/api/predict")
def predict(request: PredictionRequest):

    missing = [
        f for f in features
        if f not in request.features
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={"missing_features": missing}
        )

    row = pd.DataFrame(
        [[request.features[f] for f in features]],
        columns=features
    )

    rf_probability = float(rf.predict_proba(row)[0][1])
    xgb_probability = float(xgb.predict_proba(row)[0][1])

    return {
        "random_forest": {
            "prediction": int(rf_probability >= 0.5),
            "phishing_probability": rf_probability
        },
        "xgboost": {
            "prediction": int(xgb_probability >= 0.5),
            "phishing_probability": xgb_probability
        }
    }

@app.get("/api/graph")
def get_graph():

    graph = torch.load(
        "data/processed/ethereum_graph.pt",
        map_location="cpu",
        weights_only=False
    )

    df = pd.read_csv("data/raw/allnodes_enhancement.csv")
    accounts = df.groupby("address_id").first().sort_index()

    nodes = []

    for i in range(graph.num_nodes):
        nodes.append({
            "id": i,
            "account_id": int(accounts.index[i]),
            "label": int(graph.y[i]),
            "type": "phishing" if int(graph.y[i]) == 1 else "legitimate"
        })

    edges = [
        {
            "source": int(graph.edge_index[0, i]),
            "target": int(graph.edge_index[1, i])
        }
        for i in range(graph.edge_index.shape[1])
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges)
    }