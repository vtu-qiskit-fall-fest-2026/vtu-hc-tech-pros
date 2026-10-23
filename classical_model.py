"""
Q-Climate AI - Classical Machine Learning Module
------------------------------------------------
Implements a classical Support Vector Classifier (SVC) baseline for climate
risk classification across Low, Medium, and High categories.
"""

import time
import sys
import warnings
from pathlib import Path

# Suppress harmless sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure root directory is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.data_generator import FEATURE_NAMES, RISK_LABELS


class ClassicalClimateClassifier:
    """
    Classical Support Vector Classifier for Climate Risk Categorization.
    Uses an RBF kernel with feature standardization.
    """

    def __init__(self, kernel: str = "rbf", C: float = 1.0, gamma: str = "scale", random_state: int = 42):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,
            random_state=self.random_state,
        )
        self.is_fitted = False
        self.fit_time = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalClimateClassifier":
        """Scales features and fits the SVC model, recording training latency."""
        t0 = time.perf_counter()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fit_time = time.perf_counter() - t0
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Predicts risk classes and measures inference time."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        t0 = time.perf_counter()
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        infer_time = time.perf_counter() - t0
        return preds, infer_time

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict_single(
        self,
        temperature: float,
        humidity: float,
        rainfall: float,
        wind_speed: float,
    ) -> dict:
        """
        Inference on a single weather record with class probabilities and timing.
        """
        x_vec = np.array([[temperature, humidity, rainfall, wind_speed]])
        preds, infer_time = self.predict(x_vec)
        probs = self.predict_proba(x_vec)[0]

        pred_class = int(preds[0])
        # Model may have classes_ attribute reflecting observed classes
        classes = list(self.model.classes_)
        prob_dict = {
            RISK_LABELS[c]: float(probs[classes.index(c)]) if c in classes else 0.0
            for c in [0, 1, 2]
        }

        return {
            "predicted_label": pred_class,
            "predicted_category": RISK_LABELS[pred_class],
            "confidence": float(probs[classes.index(pred_class)]),
            "probabilities": prob_dict,
            "inference_time_ms": round(infer_time * 1000, 3),
        }


def train_classical_model(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    kernel: str = "rbf",
    C: float = 1.0,
) -> dict:
    """
    Trains and evaluates the classical model on a train-test split.
    """
    X = df[FEATURE_NAMES].values
    y = df["risk_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = ClassicalClimateClassifier(kernel=kernel, C=C, random_state=random_state)
    clf.fit(X_train, y_train)

    y_train_pred, _ = clf.predict(X_train)
    y_test_pred, infer_time = clf.predict(X_test)
    y_test_prob = clf.predict_proba(X_test)

    return {
        "classifier": clf,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "y_test_prob": y_test_prob,
        "fit_time_sec": clf.fit_time,
        "infer_time_sec": infer_time,
    }


if __name__ == "__main__":
    from src.data_generator import ensure_dataset_exists
    df = ensure_dataset_exists("data/climate_risk_dataset.csv", n_samples=80)
    res = train_classical_model(df)
    print(f"Classical SVC trained in {res['fit_time_sec']*1000:.2f} ms")
    single_res = res["classifier"].predict_single(30.0, 85.0, 90.0, 45.0)
    print("Single prediction test:", single_res)
