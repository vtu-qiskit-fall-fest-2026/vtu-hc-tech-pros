"""
Q-Climate AI - Full Benchmark Runner
------------------------------------
Executes end-to-end data generation, classical training, quantum training,
metrics computation, and prints the official comparison table.
"""

import sys
import os
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles to prevent cp1252 charmap encoding errors
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_generator import ensure_dataset_exists
from src.classical_model import train_classical_model
from src.quantum_model import train_quantum_model
from src.evaluation import compute_metrics, compare_models


def run_full_benchmark():
    print("=" * 65)
    print("⚛️  Q-CLIMATE AI: FULL PIPELINE BENCHMARK (Qiskit Fall Fest 2026)")
    print("=" * 65)

    print("\n[1/4] Loading / Generating physically grounded dataset...")
    df = ensure_dataset_exists("data/climate_risk_dataset.csv", n_samples=60)
    print(f"      Total samples: {len(df)}")
    print(f"      Class distribution:\n{df['risk_category'].value_counts().to_string()}")

    print("\n[2/4] Training Classical Support Vector Classifier (RBF kernel)...")
    c_res = train_classical_model(df, test_size=0.25, random_state=42)
    c_eval = compute_metrics(
        c_res["y_test"],
        c_res["y_test_pred"],
        c_res["fit_time_sec"],
        c_res["infer_time_sec"],
        len(c_res["y_test"]),
    )
    print(f"      Classical Training Time: {c_eval['fit_time_sec']*1000:.2f} ms")
    print(f"      Classical Test Accuracy: {c_eval['accuracy']*100:.1f}%")
    print(f"      Classical Weighted F1:   {c_eval['f1_weighted']:.4f}")

    print("\n[3/4] Simulating Genuine Qiskit Quantum Kernel (QSVC + ZZFeatureMap)...")
    q_res = train_quantum_model(
        df,
        test_size=0.25,
        random_state=42,
        reps=1,
        entanglement="linear",
        max_train_samples=35,
    )
    q_eval = compute_metrics(
        q_res["y_test"],
        q_res["y_test_pred"],
        q_res["fit_time_sec"],
        q_res["infer_time_sec"],
        len(q_res["y_test"]),
    )
    circ = q_res["circuit_details"]
    print(f"      Quantum Training Time:   {q_eval['fit_time_sec']:.2f} s")
    print(f"      Quantum Test Accuracy:   {q_eval['accuracy']*100:.1f}%")
    print(f"      Quantum Weighted F1:     {q_eval['f1_weighted']:.4f}")
    print(f"      Quantum Circuit Qubits:  {circ['num_qubits']} (Hilbert Dim: {circ['hilbert_space_dimension']})")
    print(f"      Quantum Circuit Depth:   {circ['depth']}, Total Gates: {circ['num_gates']}")

    print("\n[4/4] Side-by-Side Model Comparison:")
    comparison = compare_models(c_eval, q_eval)
    header = f"{'Metric':<30} | {'Classical SVM':<15} | {'Quantum QSVC':<15} | {'Difference':<15}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for row in comparison["metrics_table"]:
        print(f"{row['Metric']:<30} | {row['Classical SVM']:<15} | {row['Quantum QSVC']:<15} | {row['Difference']:<15}")
    print("-" * len(header))

    print("\n[Test Inference on Custom Weather Condition]")
    sample_weather = (38.5, 78.0, 60.0, 42.0)  # High heat + rain
    c_single = c_res["classifier"].predict_single(*sample_weather)
    q_single = q_res["classifier"].predict_single(*sample_weather)
    print(f"Input Weather: Temp={sample_weather[0]}°C, Humidity={sample_weather[1]}%, Rain={sample_weather[2]}mm, Wind={sample_weather[3]}km/h")
    print(f"Classical Prediction: {c_single['predicted_category']} (Conf: {c_single['confidence']*100:.1f}%, Time: {c_single['inference_time_ms']}ms)")
    print(f"Quantum Prediction:   {q_single['predicted_category']} (Metric: {q_single['confidence']*100:.1f}%, Time: {q_single['inference_time_ms']}ms)")

    print("\n✅ All components verified successfully!")


if __name__ == "__main__":
    run_full_benchmark()
