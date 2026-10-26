"""
trainer/api.py

Flask API that wraps model.py so training can be triggered over HTTP.

Endpoints:
    GET  /health    -> service status + whether a trained model exists
    POST /train     -> trains the model, saves it, returns metrics
    GET  /metrics   -> returns metrics of the last training run
    POST /predict   -> classify a message  (JSON: {"text": "..."})

Run locally:
    python api.py
"""

import json
import os
import threading

from flask import Flask, jsonify, request

from model import DATA_PATH, METRICS_PATH, MODEL_PATH, predict, train_model

app = Flask(__name__)

# Prevents two training runs from overlapping.
train_lock = threading.Lock()


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "dataset_found": os.path.exists(DATA_PATH),
            "model_trained": os.path.exists(MODEL_PATH),
        }
    )


@app.post("/train")
def train():
    if not train_lock.acquire(blocking=False):
        return jsonify({"error": "Training is already in progress."}), 409

    try:
        # Optional JSON body: {"test_size": 0.2, "random_state": 42}
        body = request.get_json(silent=True) or {}
        test_size = float(body.get("test_size", 0.2))
        random_state = int(body.get("random_state", 42))

        if not 0.05 <= test_size <= 0.5:
            return jsonify({"error": "test_size must be between 0.05 and 0.5."}), 400

        metrics = train_model(test_size=test_size, random_state=random_state)
        return jsonify({"message": "Training complete.", "metrics": metrics})

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Training failed: {e}"}), 500
    finally:
        train_lock.release()


@app.get("/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "No metrics yet. Call POST /train first."}), 404

    with open(METRICS_PATH) as f:
        return jsonify(json.load(f))


@app.post("/predict")
def predict_route():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "No trained model. Call POST /train first."}), 404

    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Send JSON like {\"text\": \"your message\"}."}), 400

    return jsonify(predict(text))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    # host 0.0.0.0 is required so the container port is reachable from outside.
    app.run(host="0.0.0.0", port=port)