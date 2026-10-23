"""
Q-Climate AI - Streamlit Dashboard
Hybrid Quantum-Classical Machine Learning for Climate Risk Categorization
Qiskit Fall Fest 2026
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from src.data_generator import (
    FEATURE_NAMES,
    FEATURE_RANGES,
    FEATURE_DISPLAY_NAMES,
    RISK_LABELS,
    RISK_COLORS,
    PRESET_SCENARIOS,
    compute_climate_risk_score,
    ensure_dataset_exists,
    generate_climate_dataset,
)
from src.classical_model import ClassicalClimateClassifier, train_classical_model
from src.quantum_model import QuantumClimateClassifier, train_quantum_model
from src.evaluation import compute_metrics, compare_models
from src.visualizer import (
    THEME_COLORS,
    plot_confusion_matrix,
    plot_metrics_comparison_bar,
    plot_latency_comparison,
    plot_weather_radar,
    plot_probability_breakdown,
    plot_quantum_kernel_heatmap,
    plot_dataset_scatter,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Q-Climate AI | Qiskit Fall Fest",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Modern Navy, Purple & Cyan Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-main: #070B1E;
        --bg-card: #0F1738;
        --bg-card-alt: #16204D;
        --border-card: #222F68;
        --accent-cyan: #00F2FE;
        --accent-purple: #8B5CF6;
        --accent-glow: rgba(0, 242, 254, 0.15);
        --text-main: #F8FAFC;
        --text-sub: #94A3B8;
        --badge-low: #00F2FE;
        --badge-med: #F59E0B;
        --badge-high: #EF4444;
    }

    /* Main background */
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #0A0F29 !important;
        border-right: 1px solid var(--border-card);
    }

    /* Hero header */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 56, 0.9) 0%, rgba(30, 20, 60, 0.9) 100%);
        border: 1px solid var(--border-card);
        border-left: 5px solid var(--accent-cyan);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 242, 254, 0.15);
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00F2FE 0%, #8B5CF6 60%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding-bottom: 6px;
    }

    .hero-subtitle {
        color: var(--text-sub);
        font-size: 1.05rem;
        margin-top: 4px;
    }

    .hero-tags {
        display: flex;
        gap: 10px;
        margin-top: 14px;
        flex-wrap: wrap;
    }

    .tag-badge {
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.35);
        color: #C4B5FD;
        font-family: 'JetBrains Mono', monospace;
    }

    .tag-badge.cyan {
        background: rgba(0, 242, 254, 0.12);
        border-color: rgba(0, 242, 254, 0.35);
        color: var(--accent-cyan);
    }

    /* Metric & prediction cards */
    .custom-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-card);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .custom-card:hover {
        border-color: rgba(0, 242, 254, 0.4);
        box-shadow: 0 8px 24px -6px rgba(0, 242, 254, 0.12);
    }

    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-main);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Risk Badges */
    .risk-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        letter-spacing: 0.5px;
    }

    .risk-low {
        background: rgba(0, 242, 254, 0.18);
        border: 1.5px solid #00F2FE;
        color: #00F2FE;
        box-shadow: 0 0 16px rgba(0, 242, 254, 0.25);
    }

    .risk-med {
        background: rgba(245, 158, 11, 0.18);
        border: 1.5px solid #F59E0B;
        color: #F59E0B;
        box-shadow: 0 0 16px rgba(245, 158, 11, 0.25);
    }

    .risk-high {
        background: rgba(239, 68, 68, 0.18);
        border: 1.5px solid #EF4444;
        color: #EF4444;
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.25);
    }

    /* Stat Box */
    .stat-box {
        background: rgba(22, 32, 77, 0.6);
        border: 1px solid var(--border-card);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }

    .stat-label {
        font-size: 0.78rem;
        color: var(--text-sub);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #FFFFFF;
        font-family: 'Space Grotesk', sans-serif;
    }

    .stat-delta {
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Code & circuit display */
    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #7928CA 0%, #00F2FE 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 15px rgba(121, 40, 202, 0.35) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.45) !important;
    }

    /* Streamlit tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-card);
        border: 1px solid var(--border-card);
        border-radius: 8px 8px 0 0;
        color: var(--text-sub);
        padding: 10px 18px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--bg-card-alt) !important;
        border-color: var(--accent-cyan) !important;
        color: var(--accent-cyan) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Cached Model Pipeline & Training
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Simulating Quantum Kernel & Training Models...")
def load_and_train_system(
    n_samples: int = 60,
    reps: int = 1,
    entanglement: str = "linear",
    max_train_samples: int = 35,
    random_state: int = 42,
):
    """
    Initializes dataset and trains both Classical SVC and genuine Quantum QSVC.
    Cached in memory to enable instantaneous dashboard responsiveness.
    """
    # 1. Dataset
    df = ensure_dataset_exists("data/climate_risk_dataset.csv", n_samples=n_samples)

    # 2. Classical Model
    c_res = train_classical_model(df, test_size=0.25, random_state=random_state)
    c_eval = compute_metrics(
        c_res["y_test"],
        c_res["y_test_pred"],
        c_res["fit_time_sec"],
        c_res["infer_time_sec"],
        len(c_res["y_test"]),
    )

    # 3. Quantum Model (QSVC with zz_feature_map and FidelityQuantumKernel)
    q_res = train_quantum_model(
        df,
        test_size=0.25,
        random_state=random_state,
        reps=reps,
        entanglement=entanglement,
        max_train_samples=max_train_samples,
    )
    q_eval = compute_metrics(
        q_res["y_test"],
        q_res["y_test_pred"],
        q_res["fit_time_sec"],
        q_res["infer_time_sec"],
        len(q_res["y_test"]),
    )

    # 4. Comparison
    comparison = compare_models(c_eval, q_eval)

    return {
        "dataset": df,
        "classical_model": c_res["classifier"],
        "classical_eval": c_eval,
        "quantum_model": q_res["classifier"],
        "quantum_eval": q_eval,
        "comparison": comparison,
        "circuit_details": q_res["circuit_details"],
        "train_kernel_matrix": q_res["train_kernel_matrix"],
    }


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚛️ Q-Climate AI")
    st.markdown(
        "<div style='font-size:0.85rem; color:#94A3B8; margin-top:-8px; margin-bottom:14px;'>"
        "Qiskit Fall Fest 2026 Submission</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Quantum Simulator Settings")

    sim_reps = st.selectbox(
        "ZZ Feature Map Reps",
        options=[1, 2],
        index=0,
        help="Number of circuit layers repeating single-qubit rotations and ZZ entangling gates.",
    )

    sim_entanglement = st.selectbox(
        "Entanglement Topology",
        options=["linear", "full"],
        index=0,
        help="Qubit coupling: 'linear' connects adjacent qubits (0-1, 1-2, 2-3); 'full' connects all pairs.",
    )

    sim_samples = st.slider(
        "Dataset Size",
        min_value=40,
        max_value=100,
        value=60,
        step=10,
        help="Kept compact to ensure simulator statevector evaluations complete in seconds.",
    )

    if st.button("🔄 Retrain Quantum Simulator"):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.75rem; color:#64748B; line-height: 1.5;">
        <b>Frameworks:</b><br/>
        • Qiskit 2.6.0 & Qiskit ML<br/>
        • scikit-learn Support Vector Machines<br/>
        • Streamlit UI & Plotly Charts<br/>
        • Python 3.13 Virtual Environment
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Load models & evaluation
# ---------------------------------------------------------
system_data = load_and_train_system(
    n_samples=sim_samples,
    reps=sim_reps,
    entanglement=sim_entanglement,
)

c_model = system_data["classical_model"]
q_model = system_data["quantum_model"]
c_eval = system_data["classical_eval"]
q_eval = system_data["quantum_eval"]
comparison = system_data["comparison"]
dataset_df = system_data["dataset"]
circuit_details = system_data["circuit_details"]
kernel_mat = system_data["train_kernel_matrix"]


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">Q-Climate AI</h1>
        <div class="hero-subtitle">
            Hybrid Quantum-Classical Machine Learning for Real-Time Multi-Hazard Climate Risk Categorization
        </div>
        <div class="hero-tags">
            <span class="tag-badge cyan">Qiskit Fall Fest 2026</span>
            <span class="tag-badge">Genuine 4-Qubit ZZ Feature Map</span>
            <span class="tag-badge">Quantum Fidelity Kernel</span>
            <span class="tag-badge cyan">Classical SVM Baseline</span>
            <span class="tag-badge">Zero Fabricated Results</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Main Navigation Tabs
# ---------------------------------------------------------
tab_predict, tab_benchmark, tab_circuit, tab_methodology = st.tabs(
    [
        "⚡ Live Climate Risk Predictor",
        "📊 Model Benchmark & Comparison",
        "⚛️ Quantum Circuit & Kernel Matrix",
        "📖 Methodology & Scientific Honesty",
    ]
)


# =========================================================
# TAB 1: LIVE CLIMATE RISK PREDICTOR
# =========================================================
with tab_predict:
    st.markdown("### 🌤️ Real-Time Environmental Risk Assessment")
    st.markdown(
        "Test current meteorological conditions with both the **Classical Support Vector Machine** "
        "and the **Qiskit Quantum Kernel Classifier (QSVC)** running in parallel."
    )

    # Preset scenarios selector
    st.markdown("#### Quick Weather Scenarios")
    scenario_cols = st.columns(len(PRESET_SCENARIOS))
    selected_preset = None

    col_idx = 0
    for preset_name, preset_vals in PRESET_SCENARIOS.items():
        with scenario_cols[col_idx % len(scenario_cols)]:
            if st.button(f"{preset_name}", key=f"btn_{preset_name}", use_container_width=True):
                st.session_state["weather_temp"] = preset_vals["temperature"]
                st.session_state["weather_hum"] = preset_vals["humidity"]
                st.session_state["weather_rain"] = preset_vals["rainfall"]
                st.session_state["weather_wind"] = preset_vals["wind_speed"]
                st.session_state["selected_preset_desc"] = preset_vals["description"]
        col_idx += 1

    if "weather_temp" not in st.session_state:
        st.session_state["weather_temp"] = 32.0
        st.session_state["weather_hum"] = 72.0
        st.session_state["weather_rain"] = 45.0
        st.session_state["weather_wind"] = 30.0
        st.session_state["selected_preset_desc"] = "Custom weather input."

    if "selected_preset_desc" in st.session_state:
        st.info(f"ℹ️ **Current Scenario Context:** {st.session_state['selected_preset_desc']}")

    st.markdown("---")

    # Environmental Parameter Sliders
    st.markdown("#### Environmental Parameters")
    input_c1, input_c2, input_c3, input_c4 = st.columns(4)

    with input_c1:
        in_temp = st.slider(
            "🌡️ Temperature (°C)",
            min_value=10.0,
            max_value=50.0,
            value=float(st.session_state["weather_temp"]),
            step=0.5,
            help="Surface ambient temperature. Critical factor for heat stress and aridity.",
        )
    with input_c2:
        in_hum = st.slider(
            "💧 Humidity (%)",
            min_value=15.0,
            max_value=100.0,
            value=float(st.session_state["weather_hum"]),
            step=1.0,
            help="Relative atmospheric humidity. Affects heat index and storm potential.",
        )
    with input_c3:
        in_rain = st.slider(
            "🌧️ 24h Rainfall (mm)",
            min_value=0.0,
            max_value=300.0,
            value=float(st.session_state["weather_rain"]),
            step=5.0,
            help="Accumulated rainfall. Key predictor for flash floods and cyclones.",
        )
    with input_c4:
        in_wind = st.slider(
            "💨 Wind Speed (km/h)",
            min_value=5.0,
            max_value=120.0,
            value=float(st.session_state["weather_wind"]),
            step=1.0,
            help="Sustained surface wind velocity. Key driver for gales, storms, and wildfires.",
        )

    # Compute Ground Truth Meteorological Score
    gt_score, gt_label, gt_cat = compute_climate_risk_score(in_temp, in_hum, in_rain, in_wind)

    # Perform Single Inferences
    c_pred = c_model.predict_single(in_temp, in_hum, in_rain, in_wind)
    q_pred = q_model.predict_single(in_temp, in_hum, in_rain, in_wind)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Side-by-Side Model Prediction Cards
    res_c1, res_c2, res_c3 = st.columns(3)

    badge_classes = {0: "risk-low", 1: "risk-med", 2: "risk-high"}

    with res_c1:
        st.markdown(
            f"""
            <div class="custom-card">
                <div class="card-title">🖥️ Classical SVM (RBF)</div>
                <div style="margin: 14px 0;">
                    <span class="risk-badge {badge_classes[c_pred['predicted_label']]}">
                        {c_pred['predicted_category']}
                    </span>
                </div>
                <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 8px;">
                    <b>Model Confidence:</b> {c_pred['confidence'] * 100:.1f}%<br/>
                    <b>Inference Latency:</b> <code style="color:#00F2FE;">{c_pred['inference_time_ms']} ms</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_c2:
        st.markdown(
            f"""
            <div class="custom-card">
                <div class="card-title">⚛️ Quantum QSVC (ZZKernel)</div>
                <div style="margin: 14px 0;">
                    <span class="risk-badge {badge_classes[q_pred['predicted_label']]}">
                        {q_pred['predicted_category']}
                    </span>
                </div>
                <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 8px;">
                    <b>Decision Metric:</b> {q_pred['confidence'] * 100:.1f}%<br/>
                    <b>Simulator Latency:</b> <code style="color:#C4B5FD;">{q_pred['inference_time_ms']} ms</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_c3:
        st.markdown(
            f"""
            <div class="custom-card">
                <div class="card-title">📐 Meteorological Composite Index</div>
                <div style="margin: 14px 0;">
                    <span class="risk-badge {badge_classes[gt_label]}">
                        {gt_cat}
                    </span>
                </div>
                <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 8px;">
                    <b>Multi-Hazard Score:</b> {gt_score:.3f} / 1.000<br/>
                    <b>Status:</b> {"🚨 Extreme Warning" if gt_label==2 else "⚠️ Weather Advisory" if gt_label==1 else "✅ Benign Conditions"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Interactive Plots: Radar Chart & Probability Breakdown
    st.markdown("<br/>", unsafe_allow_html=True)
    chart_c1, chart_c2 = st.columns([1, 1])

    with chart_c1:
        radar_fig = plot_weather_radar(in_temp, in_hum, in_rain, in_wind)
        st.plotly_chart(radar_fig, use_container_width=True)

    with chart_c2:
        prob_fig = plot_probability_breakdown(
            c_pred["probabilities"], title="Classical SVM Probability Distribution"
        )
        st.plotly_chart(prob_fig, use_container_width=True)


# =========================================================
# TAB 2: MODEL BENCHMARK & COMPARISON
# =========================================================
with tab_benchmark:
    st.markdown("### 📊 Comprehensive Performance Benchmark")
    st.markdown(
        "Side-by-side empirical comparison between the **Classical RBF SVM** and the **Genuine Qiskit QSVC**. "
        "All metrics are calculated on the unseen test set."
    )

    # Top KPI Metrics Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-label">Classical SVM Accuracy</div>
                <div class="stat-value" style="color: #8B5CF6;">{c_eval['accuracy'] * 100:.1f}%</div>
                <div class="stat-delta" style="color: #94A3B8;">F1: {c_eval['f1_weighted']:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-label">Quantum QSVC Accuracy</div>
                <div class="stat-value" style="color: #00F2FE;">{q_eval['accuracy'] * 100:.1f}%</div>
                <div class="stat-delta" style="color: #94A3B8;">F1: {q_eval['f1_weighted']:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        acc_diff = (q_eval['accuracy'] - c_eval['accuracy']) * 100
        delta_color = "#10B981" if acc_diff >= 0 else "#EF4444"
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-label">Accuracy Difference</div>
                <div class="stat-value" style="color: {delta_color};">{acc_diff:+.1f}%</div>
                <div class="stat-delta" style="color: #94A3B8;">Quantum vs Classical</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-label">Quantum Training Time</div>
                <div class="stat-value" style="color: #00F2FE;">{q_eval['fit_time_sec']:.2f} s</div>
                <div class="stat-delta" style="color: #94A3B8;">Classical: {c_eval['fit_time_sec'] * 1000:.1f} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Comparison Charts Row
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        bar_fig = plot_metrics_comparison_bar(c_eval, q_eval)
        st.plotly_chart(bar_fig, use_container_width=True)

    with b_col2:
        latency_fig = plot_latency_comparison(c_eval, q_eval)
        st.plotly_chart(latency_fig, use_container_width=True)

    # Confusion Matrices Row
    st.markdown("#### Multi-Class Confusion Matrices")
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        c_cm_fig = plot_confusion_matrix(c_eval["confusion_matrix"], model_name="Classical SVM")
        st.plotly_chart(c_cm_fig, use_container_width=True)

    with cm_col2:
        q_cm_fig = plot_confusion_matrix(q_eval["confusion_matrix"], model_name="Quantum QSVC")
        st.plotly_chart(q_cm_fig, use_container_width=True)

    # Detailed Comparison Table
    st.markdown("#### Quantitative Metrics Breakdown")
    metrics_df = pd.DataFrame(comparison["metrics_table"])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Empirical Analysis Box
    st.markdown("#### Scientific Analysis & Findings")
    st.info(f"📌 **Performance Summary:** {comparison['performance_verdict']}")
    st.warning(f"⏱️ **Computational Complexity:** {comparison['latency_verdict']}")


# =========================================================
# TAB 3: QUANTUM CIRCUIT & KERNEL MATRIX
# =========================================================
with tab_circuit:
    st.markdown("### ⚛️ Quantum Circuit Architecture & Hilbert Space")
    st.markdown(
        "Inspect the genuine 4-qubit quantum circuit used by Qiskit to map the 4-dimensional environmental "
        "feature vectors into quantum state space."
    )

    # Quantum Specs Row
    q_stat1, q_stat2, q_stat3, q_stat4 = st.columns(4)
    with q_stat1:
        st.metric("Number of Qubits", f"{circuit_details['num_qubits']} Qubits", "1 Qubit per feature")
    with q_stat2:
        st.metric("Circuit Depth", f"{circuit_details['depth']} Layers", "ZZ Entanglement")
    with q_stat3:
        st.metric("Hilbert Space Dimension", f"{circuit_details['hilbert_space_dimension']} Dimensions", "2^4 State Vector")
    with q_stat4:
        st.metric("Quantum Gates", f"{circuit_details['num_gates']} Total Gates", "H, RZ, CX/RZZ")

    st.markdown("<br/>", unsafe_allow_html=True)

    c_row1, c_row2 = st.columns([1.1, 0.9])

    with c_row1:
        st.markdown("#### Quantum Circuit Diagram (Qiskit 2.6 ZZFeatureMap)")
        st.caption("Angle encoding maps normalized Temperature, Humidity, Rainfall, and Wind Speed to qubit rotations:")
        st.code(circuit_details["text_diagram"], language="text")

    with c_row2:
        st.markdown("#### Genuine Quantum Kernel Gram Matrix")
        st.caption("Statevector fidelity overlap K(x_i, x_j) = |⟨Φ(x_i) | Φ(x_j)⟩|² computed by Qiskit:")
        if kernel_mat is not None:
            kernel_fig = plot_quantum_kernel_heatmap(kernel_mat, max_samples=20)
            st.plotly_chart(kernel_fig, use_container_width=True)
        else:
            st.info("Kernel matrix display available after simulator evaluation.")

    # Quantum Math Breakdown
    st.markdown("---")
    st.markdown("#### Mathematical Formalism of the Quantum Feature Map")
    st.markdown(
        r"""
        The **ZZ Feature Map** maps classical environmental vector $\mathbf{x} = [T, H, R, W]^\top \in [0, \pi]^4$ 
        into the quantum state $|\Phi(\mathbf{x})\rangle$:
        
        $$|\Phi(\mathbf{x})\rangle = U_{\Phi(\mathbf{x})} H^{\otimes 4} |0\rangle^{\otimes 4}$$
        
        Where the unitary evolution operator $U_{\Phi(\mathbf{x})}$ consists of single-qubit phase rotations and two-qubit entangling gates:
        
        $$U_{\Phi(\mathbf{x})} = \exp\left( i \sum_{j=1}^4 x_j Z_j + \sum_{j < k} (\pi - x_j)(\pi - x_k) Z_j Z_k \right)$$
        
        The **Quantum Kernel** measures the state overlap (quantum fidelity) between two environmental conditions $\mathbf{x}$ and $\mathbf{x}'$:
        
        $$K(\mathbf{x}, \mathbf{x}') = \left| \langle \Phi(\mathbf{x}) | \Phi(\mathbf{x}') \rangle \right|^2$$
        
        This matrix is then fed into a dual Support Vector Machine optimizer to discover maximum-margin separating hyperplanes inside the 16-dimensional quantum reproducing kernel Hilbert space (RKHS).
        """
    )


# =========================================================
# TAB 4: METHODOLOGY & SCIENTIFIC HONESTY
# =========================================================
with tab_methodology:
    st.markdown("### 📖 Methodology, Data Grounding & Scientific Integrity")

    st.markdown("#### 1. Environmental Feature Definitions & Operational Bounds")
    st.markdown(
        """
        To guarantee physical consistency, all continuous variables fall within operational meteorological extremes:
        - **Temperature (°C)**: Range [10.0, 50.0]. Extreme high temperatures induce dangerous heat stress and increase wildfire risks.
        - **Humidity (%)**: Range [15.0, 100.0]. Combined with high heat, elevated humidity prevents evaporative cooling (wet-bulb effect).
        - **Rainfall (mm)**: Range [0.0, 300.0]. Cumulative 24-hour precipitation indicates flash flooding, urban inundation, and storm surges.
        - **Wind Speed (km/h)**: Range [5.0, 120.0]. Sustained wind velocity drives cyclonic damage, tree uprooting, and electrical grid hazards.
        """
    )

    st.markdown("#### 2. Composite Climate Risk Index (CCRI) Definition")
    st.markdown(
        """
        The ground truth climate risk is defined via a multi-hazard composite formula:
        - **Storm / Flood Hazard:** $S_{\\text{flood}} = 0.65 \\cdot R_{\\text{norm}} + 0.35 \\cdot W_{\\text{norm}}$
        - **Heat Stress Index:** $S_{\\text{heat}} = 0.55 \\cdot T_{\\text{norm}} + 0.45 \\cdot (T_{\\text{norm}} \\cdot H_{\\text{norm}})$
        - **Arid Gale Hazard:** $S_{\\text{fire}} = 0.50 \\cdot T_{\\text{norm}} + 0.30 \\cdot (1.0 - H_{\\text{norm}}) + 0.20 \\cdot W_{\\text{norm}}$
        - **Compound Synergy:** $C_{\\text{compound}} = 0.20 \\cdot (R_{\\text{norm}} \\cdot W_{\\text{norm}})$
        - **Composite Score:** $R_{\\text{score}} = \\min(1.0, \\max(S_{\\text{flood}}, S_{\\text{heat}}, S_{\\text{fire}}) + C_{\\text{compound}} + \\epsilon)$
        
        **Risk Thresholds:**
        - **Low Risk (0):** $R_{\\text{score}} < 0.38$ (Benign weather, safe conditions)
        - **Medium Risk (1):** $0.38 \\le R_{\\text{score}} < 0.68$ (Moderate alerts, advisories, waterlogging or heat cautions)
        - **High Risk (2):** $R_{\\text{score}} \\ge 0.68$ (Severe emergencies: tropical storms, torrential floods, extreme heatwaves)
        """
    )

    st.markdown("---")
    st.markdown("#### 3. Climate Risk Dataset Inspection")
    scatter_fig = plot_dataset_scatter(dataset_df)
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.download_button(
        label="📥 Download Climate Dataset (CSV)",
        data=dataset_df.to_csv(index=False),
        file_name="climate_risk_dataset.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("#### 4. Scientific Honesty & Quantum Advantage Assessment")
    st.warning(
        """
        **Scientific Integrity Statement for Qiskit Fall Fest 2026:**
        
        1. **No False Quantum Advantage Claims:** We do NOT claim that quantum machine learning outperforms classical machine learning on small tabular datasets. Classical SVMs with RBF kernels compute predictions in sub-millisecond latencies and achieve strong accuracy on structured tabular features.
        2. **Why Quantum Simulation is Slower:** Simulating an $n$-qubit quantum system on classical CPUs requires calculating statevector overlaps of dimension $2^n$. Evaluating a full $N \\times N$ Gram matrix takes $O(N^2 \\cdot 2^n)$ operations on a classical computer.
        3. **Genuine Quantum Architecture:** Unlike many prototypes that mock quantum behavior, Q-Climate AI runs genuine Qiskit circuits with `zz_feature_map` and evaluates real quantum fidelity states via Qiskit Machine Learning.
        4. **Path to Quantum Advantage:** Theoretical quantum advantage in kernel methods emerges when data requires quantum feature maps that are provably hard to estimate classically (e.g., discrete logarithm problems or quantum sensor data directly gathered in quantum states).
        """
    )

st.markdown("<br/><hr/>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 10px 0;">
    <b>Q-Climate AI</b> — Qiskit Fall Fest 2026 Hackathon Prototype<br/>
    Built with Qiskit 2.6.0, Qiskit Machine Learning, scikit-learn, Streamlit, and Plotly.
    </div>
    """,
    unsafe_allow_html=True,
)
