"""Training, evaluation and persistence for the phishing URL classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from .data import featurize_dataframe, generate_synthetic_dataset, load_csv_dataset
from .features import FEATURE_NAMES, extract_features

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "model.joblib"


@dataclass
class EvalMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list  # [[tn, fp], [fn, tp]]
    n_train: int
    n_test: int

    def as_dict(self):
        return asdict(self)

    def pretty(self) -> str:
        tn, fp, fn, tp = (
            self.confusion_matrix[0][0],
            self.confusion_matrix[0][1],
            self.confusion_matrix[1][0],
            self.confusion_matrix[1][1],
        )
        return (
            f"Accuracy:  {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall:    {self.recall:.4f}\n"
            f"F1 score:  {self.f1:.4f}\n"
            f"Train/Test sizes: {self.n_train} / {self.n_test}\n"
            f"Confusion matrix (rows=actual, cols=predicted, order=[legit, phish]):\n"
            f"                 pred_legit  pred_phish\n"
            f"  actual_legit   {tn:>10}  {fp:>10}\n"
            f"  actual_phish   {fn:>10}  {tp:>10}\n"
        )


def build_dataset(csv_path: Optional[str] = None, n_per_class: int = 800, seed: int = 42) -> pd.DataFrame:
    """Return a featurized DataFrame (FEATURE_NAMES + label + url), either
    from a real external CSV (if csv_path given) or the synthetic generator.
    """
    if csv_path:
        return load_csv_dataset(csv_path)
    raw = generate_synthetic_dataset(n_per_class=n_per_class, seed=seed)
    return featurize_dataframe(raw)


def train_model(
    csv_path: Optional[str] = None,
    n_per_class: int = 800,
    seed: int = 42,
    test_size: float = 0.2,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
):
    """Train a RandomForestClassifier and return (pipeline, metrics, dataset_df)."""
    df = build_dataset(csv_path=csv_path, n_per_class=n_per_class, seed=seed)

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    metrics = EvalMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred)),
        recall=float(recall_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred)),
        confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
        n_train=len(y_train),
        n_test=len(y_test),
    )
    return clf, metrics, df


def save_model(clf, path: Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES}, path)
    return path


def load_model(path: Path = DEFAULT_MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `phishing-detector train` first."
        )
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_names"]


def predict_url(url: str, model=None, feature_names=None, model_path: Path = DEFAULT_MODEL_PATH):
    """Predict phishing (1) vs legitimate (0) for a single URL, along with
    the model's predicted probability of phishing and the raw feature
    values (useful for explaining the decision)."""
    if model is None:
        model, feature_names = load_model(model_path)
    feats = extract_features(url)
    x = np.array([[feats[name] for name in feature_names]])
    pred = int(model.predict(x)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(x)[0][1])
    return {
        "url": url,
        "prediction": "phishing" if pred == 1 else "legitimate",
        "phishing_probability": proba,
        "features": dict(feats),
    }
