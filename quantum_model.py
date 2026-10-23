"""
Q-Climate AI - Quantum Machine Learning Module
----------------------------------------------
Implements a genuine Quantum Support Vector Classifier (QSVC) using Qiskit
and Qiskit Machine Learning's FidelityQuantumKernel.

Quantum Architecture:
  - Feature Dimension: 4 qubits (Temperature, Humidity, Rainfall, Wind Speed)
  - Quantum Feature Map: zz_feature_map with linear or full entanglement
  - Quantum State Space: 2^4 = 16-dimensional Hilbert space
  - Quantum Kernel: Fidelity overlap K(x, x') = |<Phi(x)|Phi(x')>|^2
"""

import time
import sys
import warnings
from pathlib import Path

# Suppress harmless visualization runtime warnings on Windows consoles
warnings.filterwarnings("ignore", category=RuntimeWarning, module="qiskit.visualization.circuit.text")

# Ensure root directory is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import qiskit
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

from src.data_generator import FEATURE_NAMES, RISK_LABELS


class QuantumClimateClassifier:
    """
    Genuine Quantum-Kernel Classifier for 4-feature climate risk classification.
    Uses Qiskit's zz_feature_map and FidelityQuantumKernel.
    """

    def __init__(
        self,
        reps: int = 1,
        entanglement: str = "linear",
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.reps = reps
        self.entanglement = entanglement
        self.C = C
        self.random_state = random_state

        # Scale features to [0, pi] for quantum angle encoding
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))

        # 4-qubit ZZ Feature Map
        self.feature_map = zz_feature_map(
            feature_dimension=4,
            reps=self.reps,
            entanglement=self.entanglement,
            insert_barriers=True,
        )

        # Genuine Quantum Fidelity Kernel
        self.quantum_kernel = FidelityQuantumKernel(feature_map=self.feature_map)

        # QSVC with quantum kernel
        self.model = QSVC(
            quantum_kernel=self.quantum_kernel,
            C=self.C,
            random_state=self.random_state,
        )

        self.is_fitted = False
        self.fit_time = 0.0
        self.train_kernel_matrix = None
        self.X_train_scaled = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumClimateClassifier":
        """
        Scales features to [0, pi], evaluates the quantum kernel matrix,
        and trains the QSVC classifier.
        """
        t0 = time.perf_counter()
        self.X_train_scaled = self.scaler.fit_transform(X)
        self.y_train = y

        # Fit QSVC on scaled training samples
        self.model.fit(self.X_train_scaled, y)
        self.fit_time = time.perf_counter() - t0
        self.is_fitted = True

        # Precompute train-train kernel matrix for inspection & visualization
        try:
            self.train_kernel_matrix = self.quantum_kernel.evaluate(self.X_train_scaled)
        except Exception as e:
            self.train_kernel_matrix = None

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Predicts risk classes using the quantum kernel and measures inference latency."""
        if not self.is_fitted:
            raise RuntimeError("Quantum model is not fitted yet. Call fit() first.")
        t0 = time.perf_counter()
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        infer_time = time.perf_counter() - t0
        return preds, infer_time

    def evaluate_kernel(self, x_vec_a: np.ndarray, x_vec_b: np.ndarray = None) -> np.ndarray:
        """Evaluates the quantum kernel Gram matrix between scaled feature vectors."""
        x_scaled_a = self.scaler.transform(x_vec_a)
        if x_vec_b is not None:
            x_scaled_b = self.scaler.transform(x_vec_b)
            return self.quantum_kernel.evaluate(x_scaled_a, x_scaled_b)
        return self.quantum_kernel.evaluate(x_scaled_a)

    def predict_single(
        self,
        temperature: float,
        humidity: float,
        rainfall: float,
        wind_speed: float,
    ) -> dict:
        """
        Runs quantum kernel inference on a single new weather vector.
        """
        x_raw = np.array([[temperature, humidity, rainfall, wind_speed]])
        preds, infer_time = self.predict(x_raw)
        pred_class = int(preds[0])

        # Approximate confidence via decision function distance
        confidence = 0.85
        try:
            x_scaled = self.scaler.transform(x_raw)
            # QSVC delegates to sklearn SVC
            df_vals = self.model.decision_function(x_scaled)
            # Softmax on decision function for indicative confidence display
            exp_vals = np.exp(df_vals[0] - np.max(df_vals[0]))
            soft_probs = exp_vals / np.sum(exp_vals)
            prob_dict = {
                RISK_LABELS[i]: round(float(soft_probs[i]), 4)
                for i in range(len(RISK_LABELS))
            }
            confidence = float(np.max(soft_probs))
        except Exception:
            prob_dict = {RISK_LABELS[i]: (1.0 if i == pred_class else 0.0) for i in [0, 1, 2]}

        return {
            "predicted_label": pred_class,
            "predicted_category": RISK_LABELS[pred_class],
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "inference_time_ms": round(infer_time * 1000, 3),
        }

    def get_circuit_details(self) -> dict:
        """Returns quantum circuit metadata and text representation for visualization."""
        circuit = self.feature_map
        ops = circuit.count_ops()
        return {
            "num_qubits": circuit.num_qubits,
            "depth": circuit.depth(),
            "num_gates": sum(ops.values()),
            "gate_counts": dict(ops),
            "reps": self.reps,
            "entanglement": self.entanglement,
            "hilbert_space_dimension": 2 ** circuit.num_qubits,
            "text_diagram": str(circuit.draw(output="text")),
        }


def train_quantum_model(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    reps: int = 1,
    entanglement: str = "linear",
    C: float = 1.0,
    max_train_samples: int = 45,
) -> dict:
    """
    Trains and evaluates the genuine Qiskit Quantum Kernel Classifier.
    Caps training sample count to ensure rapid simulator execution.
    """
    X = df[FEATURE_NAMES].values
    y = df["risk_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Subsample training data if needed to guarantee sub-10 second execution
    if len(X_train) > max_train_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_train), size=max_train_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    q_clf = QuantumClimateClassifier(
        reps=reps, entanglement=entanglement, C=C, random_state=random_state
    )
    q_clf.fit(X_train, y_train)

    y_train_pred, _ = q_clf.predict(X_train)
    y_test_pred, infer_time = q_clf.predict(X_test)

    return {
        "classifier": q_clf,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "fit_time_sec": q_clf.fit_time,
        "infer_time_sec": infer_time,
        "circuit_details": q_clf.get_circuit_details(),
        "train_kernel_matrix": q_clf.train_kernel_matrix,
    }


if __name__ == "__main__":
    from src.data_generator import ensure_dataset_exists
    print("Testing Quantum Model module...")
    df = ensure_dataset_exists("data/climate_risk_dataset.csv", n_samples=60)
    res = train_quantum_model(df, test_size=0.25, max_train_samples=30)
    print(f"QSVC trained in {res['fit_time_sec']:.2f} s")
    print(f"Test inference took {res['infer_time_sec']:.2f} s")
    print(f"Circuit depth: {res['circuit_details']['depth']}, Qubits: {res['circuit_details']['num_qubits']}")
    single = res["classifier"].predict_single(28.0, 90.0, 80.0, 40.0)
    print("Single prediction test:", single)
