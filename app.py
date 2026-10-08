from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
DATA_PATH = BASE_DIR / "UCI_Credit_Card.csv"

# The notebook's models were trained on these 23 columns, excluding ID and target.
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "KNN": "knn.pkl",
    "SVM": "svm.pkl",
    "XGBoost": "xgboost.pkl",
    "CatBoost": "catboost.pkl",
}

app = Flask(__name__)

def load_artifact(path):
    if not path.exists():
        raise FileNotFoundError(f"Required model file was not found: {path}")
    # The notebook saves artifacts with joblib.dump(), so use joblib.load().
    return joblib.load(path)

def normalize_model_name(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    aliases = {
        "logistic regression": "Logistic Regression",
        "logistic_regression": "Logistic Regression",
        "knn": "KNN",
        "svm": "SVM",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
    }
    return aliases.get(text.lower(), text)

def load_best_model():
    best_name_path = MODEL_DIR / "best_model_name.pkl"
    if best_name_path.exists():
        raw_name = load_artifact(best_name_path)
        name = normalize_model_name(raw_name)
    else:
        name = "XGBoost"

    if name not in MODEL_FILES:
        # If metadata contains an unexpected format, try to extract a known name.
        normalized = str(name).lower()
        name = next((label for label in MODEL_FILES if label.lower() in normalized), "XGBoost")

    model = load_artifact(MODEL_DIR / MODEL_FILES[name])
    threshold_path = MODEL_DIR / "best_threshold.pkl"
    threshold = float(load_artifact(threshold_path)) if threshold_path.exists() else 0.5
    threshold = min(max(threshold, 0.0), 1.0)
    return name, model, threshold

MODEL_LOAD_ERROR = None
BEST_MODEL_NAME = None
MODEL = None
THRESHOLD = 0.5
try:
    BEST_MODEL_NAME, MODEL, THRESHOLD = load_best_model()
except Exception as exc:
    MODEL_LOAD_ERROR = str(exc)

def clean_input(form):
    values = {}
    for feature in FEATURES:
        raw = form.get(feature, "").strip()
        if raw == "":
            raise ValueError(f"Please fill in {feature.replace('_', ' ')}.")
        try:
            values[feature] = float(raw)
        except ValueError:
            raise ValueError(f"{feature.replace('_', ' ')} must be a number.")
    frame = pd.DataFrame([values], columns=FEATURES)

    # Match notebook preprocessing: uncommon education/marriage codes are grouped.
    frame["EDUCATION"] = frame["EDUCATION"].replace({0: 5, 6: 5})
    frame["MARRIAGE"] = frame["MARRIAGE"].replace({0: 3})

    # Keep values numeric; trained pipelines and CatBoost both expect this schema.
    for col in FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame.isna().any().any():
        raise ValueError("Please enter valid numeric values for every field.")
    return frame

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    submitted = {}
    if request.method == "POST":
        submitted = request.form.to_dict()
        if MODEL_LOAD_ERROR:
            error = (
                "The model could not be loaded. Check that saved_models contains the "
                "training artifacts and that the required ML libraries are installed. "
                f"Technical detail: {MODEL_LOAD_ERROR}"
            )
        else:
            try:
                features = clean_input(request.form)
                probability = float(MODEL.predict_proba(features)[0][1])
                prediction = int(probability >= THRESHOLD)
                result = {
                    "prediction": prediction,
                    "probability": probability * 100,
                    "threshold": THRESHOLD * 100,
                    "model_name": BEST_MODEL_NAME,
                    "risk_label": "Higher default risk" if prediction else "Lower default risk",
                }
            except Exception as exc:
                error = str(exc)
    return render_template(
        "index.html",
        features=FEATURES,
        result=result,
        error=error,
        submitted=submitted,
        model_name=BEST_MODEL_NAME or "Model unavailable",
        model_ready=MODEL is not None and MODEL_LOAD_ERROR is None,
    )

@app.route("/health")
def health():
    return {
        "status": "ok" if MODEL is not None and MODEL_LOAD_ERROR is None else "model_error",
        "model": BEST_MODEL_NAME,
        "detail": MODEL_LOAD_ERROR,
    }, (200 if MODEL is not None and MODEL_LOAD_ERROR is None else 503)

if __name__ == "__main__":
    if MODEL_LOAD_ERROR:
        print(f"[MODEL WARNING] {MODEL_LOAD_ERROR}")
    else:
        print(f"[READY] Loaded {BEST_MODEL_NAME}; threshold={THRESHOLD:.3f}")
    app.run(debug=True)
