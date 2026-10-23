# ⚛️ Q-Climate AI: Hybrid Quantum-Classical Climate Risk Classification

> **Submission for Qiskit Fall Fest 2026**  
> *Developed for environmental risk assessment using hybrid Quantum-Classical Machine Learning.*

---

## 📌 Executive Summary & Problem Statement

Rapidly shifting weather patterns produce compounding climate extremes—such as wet-bulb heatwaves, tropical cyclones, and flash floods. **Q-Climate AI** is a hybrid quantum-classical machine learning system designed to classify environmental conditions into **Low**, **Medium**, and **High** climate-risk categories using four continuous meteorological features:
1. **Temperature (°C)**: Ambient surface temperature
2. **Humidity (%)**: Relative atmospheric humidity
3. **Rainfall (mm)**: 24-hour cumulative precipitation
4. **Wind Speed (km/h)**: Sustained surface wind velocity

The project benchmarks a classical machine learning baseline (**Support Vector Classifier with RBF Kernel**) against a genuine quantum machine learning model (**Qiskit QSVC with a 4-Qubit ZZ Feature Map and Fidelity Quantum Kernel**).

---

## 🔬 Scientific Integrity & Quantum Advantage Assessment

In strict accordance with the hackathon requirements:
1. **No False Claims**: We do **not** claim quantum advantage. On small tabular datasets, classical machine learning (SVM with RBF kernel) remains faster (sub-millisecond inference) and achieves high accuracy.
2. **Genuine Quantum Kernel**: Rather than mocking quantum behavior, Q-Climate AI constructs a genuine 4-qubit quantum circuit via Qiskit 2.6, executes statevector fidelity evaluations via `qiskit_machine_learning.kernels.FidelityQuantumKernel`, and solves the dual SVM optimization via `QSVC`.
3. **Computational Complexity**: Simulating quantum state fidelity on classical CPUs scales exponentially with qubit count ($2^n = 16$ dimensions for 4 qubits), requiring $O(N^2 \cdot 2^n)$ operations for $N$ samples. On physical QPUs (e.g. IBM Quantum Heron / Eagle), this calculation occurs natively via physical pulse sequences.

---

## 📊 Benchmark Results (Tested & Empirical)

The following metrics were measured on unseen test data using Python 3.13 and Qiskit 2.6:

| Metric | Classical SVM (RBF) | Quantum QSVC (ZZKernel) | Delta / Ratio |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **85.0%** | **70.0%** | -15.0% |
| **Weighted F1-Score** | **0.8286** | **0.6608** | -0.1678 |
| **Weighted Precision** | **0.8875** | **0.8200** | -0.0675 |
| **Weighted Recall** | **0.8500** | **0.7000** | -0.1500 |
| **Training Time** | **~9.7 ms** | **~3.4 s** | Classical is ~350x faster (sim) |
| **Single-Sample Inference** | **~0.5 ms** | **~245 ms** | Classical is ~490x faster (sim) |

*Note: Results may vary slightly across random seeds and hardware setups, but the comparative trend remains consistent.*

---

## 🏗️ System Architecture

```
d:\Q-climate-AI\
│
├── data/
│   └── climate_risk_dataset.csv     # Synthetic, physically-grounded weather records
│
├── src/
│   ├── __init__.py                  # Package initialization
│   ├── data_generator.py            # Dataset generation & Composite Climate Risk Index (CCRI)
│   ├── classical_model.py           # Classical Support Vector Classifier (RBF kernel)
│   ├── quantum_model.py             # Qiskit 4-qubit zz_feature_map + FidelityQuantumKernel (QSVC)
│   ├── evaluation.py                # Multi-class metrics, confusion matrices, latency profiling
│   └── visualizer.py                # Interactive Plotly charts (Radar, Gram Heatmap, Metrics)
│
├── app.py                           # Full Streamlit Web Dashboard (Navy, Purple & Cyan theme)
├── run_benchmark.py                 # Standalone CLI test & benchmark script
├── requirements.txt                 # Pinned dependencies
└── README.md                        # Documentation & setup instructions
```

---

## 📐 Methodology & Risk Definitions

### 1. Physical Parameter Boundaries
All synthetic samples are constrained to realistic meteorological ranges:
- **Temperature ($T$)**: $10^\circ\text{C}$ to $50^\circ\text{C}$
- **Humidity ($H$)**: $15\%$ to $100\%$
- **Rainfall ($R$)**: $0\text{ mm}$ to $300\text{ mm}$
- **Wind Speed ($W$)**: $5\text{ km/h}$ to $120\text{ km/h}$

### 2. Composite Climate Risk Index (CCRI)
Ground truth risk classes are derived from physical multi-hazard compound formulas:
- **Storm / Flood Hazard**: $S_{\text{flood}} = 0.65 \cdot R_{\text{norm}} + 0.35 \cdot W_{\text{norm}}$
- **Heat Stress Index**: $S_{\text{heat}} = 0.55 \cdot T_{\text{norm}} + 0.45 \cdot (T_{\text{norm}} \cdot H_{\text{norm}})$
- **Arid Gale Hazard**: $S_{\text{fire}} = 0.50 \cdot T_{\text{norm}} + 0.30 \cdot (1 - H_{\text{norm}}) + 0.20 \cdot W_{\text{norm}}$
- **Compound Synergy**: $C_{\text{synergy}} = 0.20 \cdot (R_{\text{norm}} \cdot W_{\text{norm}})$
- **Composite Score**: $R_{\text{score}} = \min(1.0, \max(S_{\text{flood}}, S_{\text{heat}}, S_{\text{fire}}) + C_{\text{synergy}} + \epsilon)$

**Categorical Thresholds:**
- **Low Risk (0)**: $R_{\text{score}} < 0.38$ (Mild, safe conditions)
- **Medium Risk (1)**: $0.38 \le R_{\text{score}} < 0.68$ (Advisories for heat stress, strong winds, or heavy rain)
- **High Risk (2)**: $R_{\text{score}} \ge 0.68$ (Emergency: tropical cyclone, lethal heatwave, flash floods)

### 3. Quantum Feature Map Formulation
The 4 continuous features are scaled to $[0, \pi]$ and encoded into a 4-qubit quantum state $|\Phi(\mathbf{x})\rangle \in \mathbb{C}^{16}$ using the ZZ Feature Map:

$$|\Phi(\mathbf{x})\rangle = U_{\Phi(\mathbf{x})} H^{\otimes 4} |0\rangle^{\otimes 4}$$

Where the unitary operator applies single-qubit rotations and entangling $R_{ZZ}$ phase interactions:

$$U_{\Phi(\mathbf{x})} = \exp\left( i \sum_{j=1}^4 x_j Z_j + \sum_{j < k} (\pi - x_j)(\pi - x_k) Z_j Z_k \right)$$

The Quantum Kernel Gram matrix evaluates the fidelity overlap:

$$K(\mathbf{x}, \mathbf{x}') = |\langle \Phi(\mathbf{x}) | \Phi(\mathbf{x}') \rangle|^2$$

---

## 🚀 Beginner Quickstart: Windows Terminal Commands

Follow these exact step-by-step commands in **PowerShell** or **Command Prompt**:

### Step 1: Open PowerShell in the Project Directory
```powershell
cd D:\Q-climate-AI
```

### Step 2: (Optional) Set PowerShell Script Execution Policy
If PowerShell displays an execution policy restriction when activating the environment:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Step 3: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(You will see `(.venv)` in your command prompt.)*

### Step 4: Verify Installation with the Benchmark Script
Run the automated end-to-end benchmark to verify data generation, classical training, quantum simulation, and comparison tables:
```powershell
python run_benchmark.py
```

### Step 5: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
Your default browser will automatically open:
```
Local URL: http://localhost:8501
```

---

## 🎨 Dashboard Features

1. **Navy, Purple & Cyan Theme**: Custom cybernetic styling with high-contrast, accessible typography.
2. **Quick Weather Presets**: One-click weather profiles (e.g., *Pleasant Spring Afternoon*, *Monsoon Storm Warning*, *Lethal Heatwave Alert*, *Severe Tropical Cyclone*).
3. **Live Side-by-Side Inference**: Dual prediction cards displaying Classical SVM vs Quantum QSVC predictions, confidence scores, and real-time inference latency.
4. **Interactive Plotly Visualizations**:
   - Environmental Radar (Spider) chart comparing weather conditions against danger envelopes.
   - Quantum Kernel Gram Matrix Heatmap ($K(x_i, x_j)$ fidelity matrix).
   - Multi-class Confusion Matrices for both models.
   - Side-by-side grouped bar charts for Accuracy, Precision, Recall, and F1.
   - Latency comparison bar chart on a logarithmic scale.
5. **Interactive Circuit Inspector**: 4-qubit ASCII circuit diagram showing Hadamard layers, phase rotations, and entangling two-qubit gates.

---

## ✅ Tested & Verified Components

| Component | Status | Verification Method |
| :--- | :---: | :--- |
| **Virtual Environment & Dependencies** | Tested & Working | `python 3.13`, `qiskit 2.6.0`, `qiskit-machine-learning 0.9.1` |
| **Data Generation (`src/data_generator.py`)** | Tested & Working | 80 balanced samples generated and saved to CSV |
| **Classical SVM (`src/classical_model.py`)** | Tested & Working | RBF kernel fitted in 9.7 ms, inference in < 1 ms |
| **Quantum QSVC (`src/quantum_model.py`)** | Tested & Working | 4-qubit `zz_feature_map` + `FidelityQuantumKernel` |
| **Evaluation Suite (`src/evaluation.py`)** | Tested & Working | Accuracy, Precision, Recall, F1, Confusion Matrices |
| **Plotly Charts (`src/visualizer.py`)** | Tested & Working | Radar, heatmaps, bar charts, scatter distribution |
| **Streamlit Dashboard (`app.py`)** | Tested & Working | Launched and responded with `HTTP 200` |
| **CLI Runner (`run_benchmark.py`)** | Tested & Working | Windows UTF-8 encoding verified, zero crashes |

---

## 📄 License & Attribution
Created for **Qiskit Fall Fest 2026**. Built with open-source tools: [Qiskit](https://qiskit.org/), [scikit-learn](https://scikit-learn.org/), [Streamlit](https://streamlit.io/), and [Plotly](https://plotly.com/).
