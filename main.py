"""ML Pipeline — Production end-to-end ML pipeline.

Data ingestion, model training with MLflow tracking,
model registry, and REST prediction API.
"""

import os
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import redis
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Pipeline", version="1.0.0")

# --- Redis ---
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True,
)

# --- MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("ml-pipeline")


# --- Models ---
class TrainingConfig(BaseModel):
    model_type: str = "random_forest"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100


class PredictionRequest(BaseModel):
    features: list[float]


class PredictionResponse(BaseModel):
    prediction: int
    probabilities: list[float]
    model_version: str
    inference_time_ms: float


# --- Data ingestion ---
def load_data(source: str = "csv") -> pd.DataFrame:
    """Load data from various sources."""
    if source == "csv":
        path = os.getenv("DATA_PATH", "data/training.csv")
        df = pd.read_csv(path)
    else:
        # Generate synthetic data for demo
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, n),
            "feature_2": np.random.normal(5, 2, n),
            "feature_3": np.random.exponential(1, n),
            "feature_4": np.random.uniform(0, 10, n),
            "target": np.random.randint(0, 2, n),
        })
    logger.info(f"Loaded {len(df)} rows from {source}")
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess dataframe for training."""
    df = df.dropna()
    X = df.drop("target", axis=1, errors="ignore")
    y = df["target"] if "target" in df.columns else pd.Series(np.zeros(len(df)))

    # Encode categorical columns
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return X, y


# --- Training ---
def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
) -> tuple[object, float]:
    """Train a model and return model + accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    if config.model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config.n_estimators, random_state=config.random_state
        )
    elif config.model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=config.n_estimators, random_state=config.random_state
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    with mlflow.start_run():
        mlflow.log_params(config.dict())
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("test_size", config.test_size)
        mlflow.sklearn.log_model(model, "model")
        logger.info(f"Logged model with accuracy: {accuracy:.4f}")

    return model, accuracy


# --- Model registry ---
_model_registry: dict[str, object] = {}
_model_version = "1.0.0"


def register_model(name: str, model: object) -> str:
    """Register a trained model."""
    _model_registry[name] = model
    logger.info(f"Registered model '{name}' version {_model_version}")
    return _model_version


# --- API ---
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/train")
def train(config: Optional[TrainingConfig] = None):
    """Train a model on the dataset."""
    cfg = config or TrainingConfig()
    df = load_data()
    X, y = preprocess(df)
    model, accuracy = train_model(X, y, cfg)
    version = register_model("default", model)
    return {
        "status": "trained",
        "accuracy": round(accuracy, 4),
        "model_version": version,
        "features": list(X.columns),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """Run inference on a single sample."""
    import time

    start = time.time()
    model = _model_registry.get("default")
    if model is None:
        raise HTTPException(status_code=404, detail="No model trained yet. POST /train first.")

    features = np.array(req.features).reshape(1, -1)
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0].tolist()
    elapsed = (time.time() - start) * 1000

    return PredictionResponse(
        prediction=prediction,
        probabilities=probabilities,
        model_version=_model_version,
        inference_time_ms=round(elapsed, 2),
    )


@app.get("/models")
def list_models():
    """List registered models."""
    return {
        "models": list(_model_registry.keys()),
        "version": _model_version,
        "mlflow_uri": MLFLOW_TRACKING_URI,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)