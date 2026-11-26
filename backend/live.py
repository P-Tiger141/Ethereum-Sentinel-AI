
import asyncio
import os
import pandas as pd

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import joblib

router = APIRouter()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET = os.path.join(BASE, "data", "raw", "allnodes_enhancement.csv")

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

rf = joblib.load(os.path.join(BASE, "models", "random_forest.joblib"))
xgb = joblib.load(os.path.join(BASE, "models", "xgboost.joblib"))


@router.websocket("/ws/process")
async def process_dataset(websocket: WebSocket):

    await websocket.accept()

    try:
        await websocket.send_json({
            "type": "status",
            "message": "Loading Ethereum dataset..."
        })

        if not os.path.exists(DATASET):
            await websocket.send_json({
                "type": "error",
                "message": "Dataset not found: " + DATASET
            })
            await websocket.close()
            return

        df = pd.read_csv(DATASET)

        await websocket.send_json({
            "type": "status",
            "message": "Preparing unique Ethereum accounts..."
        })

        accounts = df.drop_duplicates(
            subset="address_id"
        ).copy()

        missing = [f for f in FEATURES if f not in accounts.columns]

        if missing:
            await websocket.send_json({
                "type": "error",
                "message": "Missing features: " + str(missing)
            })
            await websocket.close()
            return

        total = len(accounts)
        processed = 0
        rf_suspicious = 0
        xgb_suspicious = 0
        alerts = []

        batch_size = 100

        await websocket.send_json({
            "type": "start",
            "total": total,
            "dataset_rows": len(df),
            "unique_accounts": total
        })

        for start in range(0, total, batch_size):

            batch = accounts.iloc[start:start + batch_size]
            X = batch[FEATURES]

            rf_probs = rf.predict_proba(X)[:, 1]
            xgb_probs = xgb.predict_proba(X)[:, 1]

            rf_predictions = rf_probs >= 0.5
            xgb_predictions = xgb_probs >= 0.5

            rf_suspicious += int(rf_predictions.sum())
            xgb_suspicious += int(xgb_predictions.sum())

            for i in range(len(batch)):
                if rf_predictions[i] or xgb_predictions[i]:
                    alerts.append({
                        "address_id": int(batch.iloc[i]["address_id"]),
                        "rf_probability": round(float(rf_probs[i]), 4),
                        "xgb_probability": round(float(xgb_probs[i]), 4),
                        "rf_prediction": int(rf_predictions[i]),
                        "xgb_prediction": int(xgb_predictions[i])
                    })

            processed += len(batch)

            await websocket.send_json({
                "type": "progress",
                "processed": processed,
                "total": total,
                "percentage": round(processed / total * 100, 2),
                "rf_suspicious": rf_suspicious,
                "xgb_suspicious": xgb_suspicious,
                "alerts_count": len(alerts),
                "latest_alerts": alerts[-5:]
            })

            await asyncio.sleep(0.05)

        await websocket.send_json({
            "type": "complete",
            "processed": processed,
            "total": total,
            "rf_suspicious": rf_suspicious,
            "xgb_suspicious": xgb_suspicious,
            "alerts_count": len(alerts),
            "alerts": alerts[:100]
        })

    except WebSocketDisconnect:
        print("Dashboard disconnected")

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
