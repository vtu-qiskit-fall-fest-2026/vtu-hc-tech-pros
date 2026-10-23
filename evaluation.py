"""
Q-Climate AI - Evaluation & Benchmarking Module
-----------------------------------------------
Computes rigorous, honest performance metrics for Classical SVM and QSVC:
  - Accuracy, Precision (weighted & macro), Recall (weighted & macro), F1-Score
  - Multi-class Confusion Matrices
  - Training and Inference latencies
  - Honest scientific comparison commentary (no fabricated claims)
"""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from src.data_generator import RISK_LABELS


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fit_time_sec: float,
    infer_time_sec: float,
    n_test_samples: int,
) -> dict:
    """
    Computes standard evaluation metrics and latency statistics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=[RISK_LABELS[i] for i in [0, 1, 2]], output_dict=True, zero_division=0
    )

    avg_infer_latency_ms = (infer_time_sec / max(1, n_test_samples)) * 1000.0

    return {
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "fit_time_sec": round(fit_time_sec, 4),
        "infer_time_sec": round(infer_time_sec, 4),
        "infer_latency_per_sample_ms": round(avg_infer_latency_ms, 2),
    }


def compare_models(classical_eval: dict, quantum_eval: dict) -> dict:
    """
    Builds a structured comparison summary between Classical SVM and Quantum QSVC.
    Provides an evidence-grounded scientific assessment without unjustified hype.
    """
    c_acc = classical_eval["accuracy"]
    q_acc = quantum_eval["accuracy"]
    c_f1 = classical_eval["f1_weighted"]
    q_f1 = quantum_eval["f1_weighted"]
    c_fit = classical_eval["fit_time_sec"]
    q_fit = quantum_eval["fit_time_sec"]

    acc_delta = q_acc - c_acc
    f1_delta = q_f1 - c_f1
    speedup_ratio = q_fit / max(1e-6, c_fit)

    # Honest assessment generation
    if acc_delta > 0.05:
        performance_verdict = (
            f"The Quantum QSVC achieved a higher test accuracy ({q_acc*100:.1f}% vs {c_acc*100:.1f}%), "
            f"demonstrating that the 4-qubit ZZ feature map effectively captured nonlinear interactions "
            f"in this small sample. However, this does NOT constitute practical quantum advantage."
        )
    elif abs(acc_delta) <= 0.05:
        performance_verdict = (
            f"Both models achieved competitive performance (Quantum: {q_acc*100:.1f}%, Classical: {c_acc*100:.1f}%). "
            f"The classical RBF SVM and the 4-qubit ZZ quantum kernel construct comparable decision boundaries "
            f"on this meteorological dataset."
        )
    else:
        performance_verdict = (
            f"The Classical SVM achieved higher test accuracy ({c_acc*100:.1f}% vs {q_acc*100:.1f}%). "
            f"Classical RBF kernels benefit from smooth Gaussian metric distances and instant optimization on "
            f"classical architectures."
        )

    latency_verdict = (
        f"Classical SVM training finished in {c_fit*1000:.1f} ms, whereas Quantum simulation required "
        f"{q_fit:.2f} s (~{speedup_ratio:.0f}x longer). This overhead is expected because simulating 4-qubit "
        f"statevector fidelity overlaps on classical CPUs involves evaluating exponential vector tensor products "
        f"for every pair of samples."
    )

    quantum_advantage_analysis = (
        "Scientific Integrity Note (Qiskit Fall Fest):\n"
        "1. No Practical Quantum Advantage: On small tabular climate datasets, classical machine learning "
        "(SVM, XGBoost, Random Forests) remains faster and highly accurate.\n"
        "2. Value of Quantum Kernels: The ZZFeatureMap maps 4 climate variables into a 16-dimensional "
        "reproducing kernel Hilbert space (RKHS). In larger feature spaces with complex entangled correlations "
        "that are classically hard to compute (e.g., Liu et al., Nature Physics 2021), quantum kernels offer "
        "rigorous theoretical promise.\n"
        "3. Hardware Perspective: On physical NISQ quantum hardware (e.g. IBM Quantum Heron / Eagle QPUs), "
        "quantum kernels execute natively via pulse sequences rather than statevector simulation."
    )

    metrics_table = [
        {"Metric": "Accuracy", "Classical SVM": f"{c_acc * 100:.1f}%", "Quantum QSVC": f"{q_acc * 100:.1f}%", "Difference": f"{acc_delta * 100:+.1f}%"},
        {"Metric": "F1-Score (Weighted)", "Classical SVM": f"{c_f1:.4f}", "Quantum QSVC": f"{q_f1:.4f}", "Difference": f"{f1_delta:+.4f}"},
        {"Metric": "Precision (Weighted)", "Classical SVM": f"{classical_eval['precision_weighted']:.4f}", "Quantum QSVC": f"{quantum_eval['precision_weighted']:.4f}", "Difference": f"{quantum_eval['precision_weighted'] - classical_eval['precision_weighted']:+.4f}"},
        {"Metric": "Recall (Weighted)", "Classical SVM": f"{classical_eval['recall_weighted']:.4f}", "Quantum QSVC": f"{quantum_eval['recall_weighted']:.4f}", "Difference": f"{quantum_eval['recall_weighted'] - classical_eval['recall_weighted']:+.4f}"},
        {"Metric": "Training Time", "Classical SVM": f"{c_fit * 1000:.2f} ms", "Quantum QSVC": f"{q_fit:.2f} s", "Difference": f"{q_fit / max(1e-6, c_fit):.1f}x slower (sim)"},
        {"Metric": "Inference Latency (per sample)", "Classical SVM": f"{classical_eval['infer_latency_per_sample_ms']:.2f} ms", "Quantum QSVC": f"{quantum_eval['infer_latency_per_sample_ms']:.2f} ms", "Difference": f"{quantum_eval['infer_latency_per_sample_ms'] / max(1e-6, classical_eval['infer_latency_per_sample_ms']):.1f}x"},
    ]

    return {
        "metrics_table": metrics_table,
        "performance_verdict": performance_verdict,
        "latency_verdict": latency_verdict,
        "quantum_advantage_analysis": quantum_advantage_analysis,
        "acc_delta": acc_delta,
        "f1_delta": f1_delta,
        "speedup_ratio": speedup_ratio,
    }


if __name__ == "__main__":
    from src.data_generator import ensure_dataset_exists
    from src.classical_model import train_classical_model
    from src.quantum_model import train_quantum_model

    df = ensure_dataset_exists("data/climate_risk_dataset.csv", n_samples=60)
    c_res = train_classical_model(df, test_size=0.25)
    q_res = train_quantum_model(df, test_size=0.25, max_train_samples=30)

    c_eval = compute_metrics(c_res["y_test"], c_res["y_test_pred"], c_res["fit_time_sec"], c_res["infer_time_sec"], len(c_res["y_test"]))
    q_eval = compute_metrics(q_res["y_test"], q_res["y_test_pred"], q_res["fit_time_sec"], q_res["infer_time_sec"], len(q_res["y_test"]))

    comparison = compare_models(c_eval, q_eval)
    print("Metrics Table:")
    for row in comparison["metrics_table"]:
        print(row)
    print("\nVerdict:\n", comparison["performance_verdict"])
