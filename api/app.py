"""
api/app.py — Flask REST API and dashboard server for TCSP-PC.

Endpoints:
  GET  /                        → Dashboard HTML
  GET  /api/status              → System health
  POST /api/predict             → Predict single vehicle emission class
  POST /api/predict_batch       → Predict batch of vehicles
  GET  /api/realtime            → Simulated real-time IoT reading + prediction
  GET  /api/metrics             → Saved test metrics (from evaluate.py)
  GET  /api/recommendations     → Current pollution-control recommendations

Usage:
    python api/app.py
    # Open: http://localhost:5000
"""

import os
import sys
import json
import time
import numpy as np
import joblib

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

from data_generator import generate_realtime_sample, VEHICLE_TYPES, CITY_PROFILES
from preprocessing import load_processed

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "dashboard", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "dashboard", "static"),
)
CORS(app)

SAVE_DIR    = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (lazy, cached)
# ─────────────────────────────────────────────────────────────────────────────
_model_cache = {}

def load_model(name: str):
    """Load a saved model by name (capsnet / veqm / mptcprp / mwis / convbilstm)."""
    if name in _model_cache:
        return _model_cache[name]
    path = os.path.join(SAVE_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    m = joblib.load(path)
    _model_cache[name] = m
    return m


def load_scaler():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "scaler.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector builder (mirrors preprocessing.py)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_ORDER = [
    "hour", "day_of_week", "distance_km",
    "emission_rate", "vehicle_density", "travel_time_min",
    "city_emission_factor", "vehicle_type_enc", "city_enc",
]

VEHICLE_TYPE_MAP = {v: i for i, v in enumerate(sorted(VEHICLE_TYPES.keys()))}
CITY_MAP         = {c: i for i, c in enumerate(sorted(CITY_PROFILES.keys()))}


def record_to_feature_vector(record: dict) -> np.ndarray:
    """Convert a raw IoT record dict to a scaled feature vector."""
    vtype_enc = VEHICLE_TYPE_MAP.get(record.get("vehicle_type", "sedan"), 0)
    city_enc  = CITY_MAP.get(record.get("city", "New York"), 0)

    vec = np.array([[
        record.get("hour", 12),
        record.get("day_of_week", 0),
        record.get("distance_km", 20),
        record.get("emission_rate", 15.0),
        record.get("vehicle_density", 50),
        record.get("travel_time_min", 3.0),
        record.get("city_emission_factor", 1.3),
        vtype_enc,
        city_enc,
    ]], dtype=np.float32)

    scaler = load_scaler()
    if scaler is not None:
        vec = scaler.transform(vec)
    return vec


def generate_recommendations(emission_rate: float, vehicle_density: int,
                              prediction: int) -> list:
    """Rule-based pollution recommendations (mirrors paper's Fig. 6 logic)."""
    recs = []
    if prediction == 1 or emission_rate >= 30:
        recs.append("Restrict high-emission vehicle access to this zone")
        recs.append("Suggest alternate routing via low-pollution corridors")
        recs.append("Issue emission violation alert to registered owner")
        if emission_rate >= 45:
            recs.append("Immediate mandatory emission inspection required")
    if vehicle_density > 60:
        recs.append("Activate dynamic traffic signal adjustment to reduce congestion")
        recs.append("Enable park-and-ride recommendation for nearby lots")
    if prediction == 0:
        recs.append("Vehicle is eco-compliant — no immediate action needed")
        recs.append("Continue monitoring emission trend at 1 km intervals")
    return recs


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve dashboard."""
    tmpl = os.path.join(
        os.path.dirname(__file__), "..", "dashboard", "templates", "index.html"
    )
    if os.path.exists(tmpl):
        return render_template("index.html")
    return "<h2>TCSP-PC API is running. See <a href='/api/status'>/api/status</a></h2>"


@app.route("/api/status")
def status():
    models_found = []
    for name in ["capsnet", "veqm", "mptcprp", "mwis", "convbilstm"]:
        path = os.path.join(SAVE_DIR, f"{name}.pkl")
        if os.path.exists(path):
            models_found.append(name)
    return jsonify({
        "status": "running",
        "models_available": models_found,
        "scaler_loaded": load_scaler() is not None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict emission class for a single vehicle record.
    Body JSON fields (all optional — defaults applied):
      hour, day_of_week, distance_km, emission_rate, vehicle_density,
      travel_time_min, city_emission_factor, vehicle_type, city
    Optional: model_name (default: capsnet)
    """
    data = request.get_json(force=True) or {}
    model_name = data.pop("model_name", "capsnet")

    model = load_model(model_name)
    if model is None:
        return jsonify({"error": f"Model '{model_name}' not found. Run train.py first."}), 404

    X = record_to_feature_vector(data)
    t0 = time.time()
    pred  = int(model.predict(X)[0])
    try:
        prob = float(model.predict_proba(X)[0][1])
    except Exception:
        prob = float(pred)
    elapsed = time.time() - t0

    emission_rate   = float(data.get("emission_rate", 15.0))
    vehicle_density = int(data.get("vehicle_density", 50))
    recs = generate_recommendations(emission_rate, vehicle_density, pred)

    return jsonify({
        "prediction":   pred,
        "label":        "Polluting" if pred == 1 else "Eco-friendly",
        "probability":  round(prob, 4),
        "model_used":   model_name,
        "inference_ms": round(elapsed * 1000, 2),
        "recommendations": recs,
        "input_received": data,
    })


@app.route("/api/predict_batch", methods=["POST"])
def predict_batch():
    """Predict emission class for a list of vehicle records."""
    payload = request.get_json(force=True) or {}
    records = payload.get("records", [])
    model_name = payload.get("model_name", "capsnet")

    if not records:
        return jsonify({"error": "No records provided"}), 400

    model = load_model(model_name)
    if model is None:
        return jsonify({"error": f"Model '{model_name}' not found."}), 404

    X = np.vstack([record_to_feature_vector(r) for r in records])
    t0 = time.time()
    preds = model.predict(X).tolist()
    try:
        probs = model.predict_proba(X)[:, 1].tolist()
    except Exception:
        probs = [float(p) for p in preds]
    elapsed = time.time() - t0

    results = []
    for i, (rec, pred, prob) in enumerate(zip(records, preds, probs)):
        results.append({
            "index":       i,
            "prediction":  pred,
            "label":       "Polluting" if pred == 1 else "Eco-friendly",
            "probability": round(prob, 4),
        })

    return jsonify({
        "total":        len(records),
        "polluting":    sum(preds),
        "eco_friendly": len(preds) - sum(preds),
        "model_used":   model_name,
        "inference_ms": round(elapsed * 1000, 2),
        "results":      results,
    })


@app.route("/api/realtime")
def realtime():
    """Simulate one real-time IoT sensor reading + instant prediction."""
    city = request.args.get("city", "New York")
    model_name = request.args.get("model", "capsnet")

    record = generate_realtime_sample(city)
    model  = load_model(model_name)

    pred, prob = 0, 0.0
    if model is not None:
        X = record_to_feature_vector(record)
        pred = int(model.predict(X)[0])
        try:
            prob = float(model.predict_proba(X)[0][1])
        except Exception:
            prob = float(pred)

    recs = generate_recommendations(
        record["emission_rate"], record["vehicle_density"], pred
    )

    return jsonify({
        "sensor_reading": record,
        "prediction":     pred,
        "label":          "Polluting" if pred == 1 else "Eco-friendly",
        "probability":    round(prob, 4),
        "recommendations": recs,
        "model_used":     model_name,
    })


@app.route("/api/metrics")
def metrics():
    """Return test metrics saved by evaluate.py."""
    path = os.path.join(RESULTS_DIR, "test_metrics.csv")
    if not os.path.exists(path):
        return jsonify({"error": "No metrics found. Run evaluate.py first."}), 404
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k != "name" else v for k, v in row.items()})
    return jsonify({"metrics": rows})


@app.route("/api/recommendations")
def recommendations():
    """Return current recommendations based on a simulated city reading."""
    city = request.args.get("city", "New York")
    record = generate_realtime_sample(city)
    model  = load_model("capsnet")

    pred = 0
    if model is not None:
        X    = record_to_feature_vector(record)
        pred = int(model.predict(X)[0])

    recs = generate_recommendations(
        record["emission_rate"], record["vehicle_density"], pred
    )
    return jsonify({
        "city":            city,
        "emission_rate":   record["emission_rate"],
        "vehicle_density": record["vehicle_density"],
        "classification":  "Polluting" if pred == 1 else "Eco-friendly",
        "recommendations": recs,
    })


@app.route("/api/cities")
def cities():
    return jsonify({"cities": sorted(CITY_PROFILES.keys())})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  TCSP-PC Dashboard & API")
    print("  http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)