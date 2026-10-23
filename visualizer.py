"""
Q-Climate AI - Visualization Module
-----------------------------------
Plotly visualization charts with Navy (#0B0F29), Purple (#8B5CF6),
and Cyan (#00F2FE) styling for the Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.data_generator import RISK_LABELS, RISK_COLORS, FEATURE_RANGES, FEATURE_DISPLAY_NAMES


THEME_COLORS = {
    "bg": "#0B0F29",
    "card_bg": "#121A42",
    "card_border": "#212D6E",
    "text_bright": "#F8FAFC",
    "text_dim": "#94A3B8",
    "grid": "#1E2958",
    "cyan": "#00F2FE",
    "purple": "#8B5CF6",
    "magenta": "#D946EF",
    "green": "#10B981",
    "amber": "#F59E0B",
    "red": "#EF4444",
}


def base_layout(title: str = "") -> dict:
    """Returns a unified dark Navy-Purple-Cyan Plotly layout template."""
    return dict(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family="Inter, sans-serif", size=15, color=THEME_COLORS["text_bright"]),
            x=0.02,
        ),
        paper_bgcolor=THEME_COLORS["card_bg"],
        plot_bgcolor=THEME_COLORS["card_bg"],
        font=dict(family="Inter, sans-serif", color=THEME_COLORS["text_bright"], size=12),
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis=dict(
            gridcolor=THEME_COLORS["grid"],
            zerolinecolor=THEME_COLORS["grid"],
            color=THEME_COLORS["text_dim"],
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor=THEME_COLORS["grid"],
            zerolinecolor=THEME_COLORS["grid"],
            color=THEME_COLORS["text_dim"],
            tickfont=dict(size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color=THEME_COLORS["text_bright"]),
        ),
    )


def plot_confusion_matrix(cm: list, model_name: str = "Classical SVM") -> go.Figure:
    """Generates an annotated confusion matrix heatmap."""
    categories = [RISK_LABELS[i] for i in [0, 1, 2]]
    cm_arr = np.array(cm)

    # Custom colorscale: Deep Navy to Purple to Cyan
    colorscale = [
        [0.0, "#0B0F29"],
        [0.5, "#4C1D95"],
        [1.0, "#00F2FE"],
    ]

    # Text annotations
    annotations = []
    for i in range(3):
        for j in range(3):
            val = cm_arr[i, j]
            annotations.append(
                dict(
                    x=categories[j],
                    y=categories[i],
                    text=f"<b>{val}</b>",
                    font=dict(color="#FFFFFF" if val > 0 else THEME_COLORS["text_dim"], size=14),
                    showarrow=False,
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr,
            x=categories,
            y=categories,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(text="Count", font=dict(color=THEME_COLORS["text_bright"])),
                tickfont=dict(color=THEME_COLORS["text_bright"]),
            ),
        )
    )

    layout = base_layout(f"{model_name} Confusion Matrix")
    layout.update(
        annotations=annotations,
        xaxis_title="Predicted Category",
        yaxis_title="True Category",
        yaxis=dict(autorange="reversed", color=THEME_COLORS["text_dim"]),
    )
    fig.update_layout(layout)
    return fig


def plot_metrics_comparison_bar(classical_eval: dict, quantum_eval: dict) -> go.Figure:
    """Grouped bar chart comparing Accuracy, Precision, Recall, and F1."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    classical_vals = [
        classical_eval["accuracy"] * 100,
        classical_eval["precision_weighted"] * 100,
        classical_eval["recall_weighted"] * 100,
        classical_eval["f1_weighted"] * 100,
    ]
    quantum_vals = [
        quantum_eval["accuracy"] * 100,
        quantum_eval["precision_weighted"] * 100,
        quantum_eval["recall_weighted"] * 100,
        quantum_eval["f1_weighted"] * 100,
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Classical SVM (RBF)",
            x=metrics,
            y=classical_vals,
            marker_color=THEME_COLORS["purple"],
            text=[f"{v:.1f}%" for v in classical_vals],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Quantum QSVC (ZZKernel)",
            x=metrics,
            y=quantum_vals,
            marker_color=THEME_COLORS["cyan"],
            text=[f"{v:.1f}%" for v in quantum_vals],
            textposition="auto",
        )
    )

    layout = base_layout("Model Classification Benchmark (%)")
    layout.update(
        barmode="group",
        yaxis=dict(range=[0, 105], title="Score (%)", color=THEME_COLORS["text_dim"], gridcolor=THEME_COLORS["grid"]),
    )
    fig.update_layout(layout)
    return fig


def plot_latency_comparison(classical_eval: dict, quantum_eval: dict) -> go.Figure:
    """Latency comparison chart (training and inference)."""
    categories = ["Training Time (seconds)", "Inference Latency per sample (ms)"]
    c_times = [classical_eval["fit_time_sec"], classical_eval["infer_latency_per_sample_ms"]]
    q_times = [quantum_eval["fit_time_sec"], quantum_eval["infer_latency_per_sample_ms"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Classical SVM",
            x=categories,
            y=c_times,
            marker_color=THEME_COLORS["purple"],
            text=[f"{c_times[0]:.4f}s", f"{c_times[1]:.2f}ms"],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Quantum QSVC",
            x=categories,
            y=q_times,
            marker_color=THEME_COLORS["cyan"],
            text=[f"{q_times[0]:.2f}s", f"{q_times[1]:.2f}ms"],
            textposition="auto",
        )
    )

    layout = base_layout("Latency & Execution Overhead (Log Scale)")
    layout.update(
        barmode="group",
        yaxis=dict(type="log", title="Execution Latency (log scale)", gridcolor=THEME_COLORS["grid"]),
    )
    fig.update_layout(layout)
    return fig


def plot_weather_radar(
    temperature: float,
    humidity: float,
    rainfall: float,
    wind_speed: float,
    title: str = "Live Environmental Radar",
) -> go.Figure:
    """
    Radar (spider) chart showing normalized climate parameters relative to danger thresholds.
    """
    categories = ["Temperature", "Humidity", "Rainfall", "Wind Speed"]

    # Normalize values to 0-100% of their operational extremes
    t_pct = ((temperature - FEATURE_RANGES["temperature"][0]) / (FEATURE_RANGES["temperature"][1] - FEATURE_RANGES["temperature"][0])) * 100
    h_pct = ((humidity - FEATURE_RANGES["humidity"][0]) / (FEATURE_RANGES["humidity"][1] - FEATURE_RANGES["humidity"][0])) * 100
    r_pct = ((rainfall - FEATURE_RANGES["rainfall"][0]) / (FEATURE_RANGES["rainfall"][1] - FEATURE_RANGES["rainfall"][0])) * 100
    w_pct = ((wind_speed - FEATURE_RANGES["wind_speed"][0]) / (FEATURE_RANGES["wind_speed"][1] - FEATURE_RANGES["wind_speed"][0])) * 100

    vals = [t_pct, h_pct, r_pct, w_pct]
    vals.append(vals[0])  # close radar polygon
    cat_closed = categories + [categories[0]]

    # Reference safe and high risk threshold boundaries
    safe_ref = [35, 45, 15, 25, 35]
    danger_ref = [75, 80, 65, 70, 75]

    fig = go.Figure()
    # High Risk Danger Zone boundary
    fig.add_trace(
        go.Scatterpolar(
            r=danger_ref,
            theta=cat_closed,
            fill="toself",
            name="High Risk Zone",
            fillcolor="rgba(239, 68, 68, 0.15)",
            line=dict(color=THEME_COLORS["red"], dash="dash", width=1),
        )
    )
    # Safe Reference Zone boundary
    fig.add_trace(
        go.Scatterpolar(
            r=safe_ref,
            theta=cat_closed,
            fill="toself",
            name="Low Risk Baseline",
            fillcolor="rgba(0, 242, 254, 0.10)",
            line=dict(color=THEME_COLORS["cyan"], dash="dot", width=1),
        )
    )
    # Current Weather
    fig.add_trace(
        go.Scatterpolar(
            r=vals,
            theta=cat_closed,
            fill="toself",
            name="Current Weather Input",
            fillcolor="rgba(139, 92, 246, 0.35)",
            line=dict(color=THEME_COLORS["cyan"], width=2.5),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color=THEME_COLORS["text_dim"], gridcolor=THEME_COLORS["grid"]),
            angularaxis=dict(color=THEME_COLORS["text_bright"], gridcolor=THEME_COLORS["grid"]),
            bgcolor=THEME_COLORS["card_bg"],
        ),
        paper_bgcolor=THEME_COLORS["card_bg"],
        font=dict(color=THEME_COLORS["text_bright"], size=11),
        margin=dict(l=40, r=40, t=50, b=30),
        title=dict(text=f"<b>{title}</b>", font=dict(color=THEME_COLORS["text_bright"], size=14), x=0.02),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    return fig


def plot_probability_breakdown(prob_dict: dict, title: str = "Class Prediction Probabilities") -> go.Figure:
    """Horizontal stacked / grouped bar chart of risk probabilities."""
    labels = list(prob_dict.keys())
    values = [prob_dict[k] * 100 for k in labels]
    colors = [THEME_COLORS["cyan"], THEME_COLORS["amber"], THEME_COLORS["red"]]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
        )
    )

    layout = base_layout(title)
    layout.update(
        xaxis=dict(range=[0, 100], title="Probability (%)", gridcolor=THEME_COLORS["grid"]),
        yaxis=dict(color=THEME_COLORS["text_bright"]),
        margin=dict(l=100, r=30, t=50, b=40),
    )
    fig.update_layout(layout)
    return fig


def plot_quantum_kernel_heatmap(kernel_matrix: np.ndarray, max_samples: int = 25) -> go.Figure:
    """Visualizes the genuine Quantum Kernel Gram matrix K(x_i, x_j)."""
    sub_k = kernel_matrix[:max_samples, :max_samples]
    n = sub_k.shape[0]

    colorscale = [
        [0.0, "#0B0F29"],
        [0.35, "#3B0764"],
        [0.7, "#7C3AED"],
        [1.0, "#00F2FE"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=sub_k,
            x=[f"x_{i}" for i in range(n)],
            y=[f"x_{i}" for i in range(n)],
            colorscale=colorscale,
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(
                title=dict(text="Fidelity |⟨Φ|Φ'⟩|²", font=dict(color=THEME_COLORS["text_bright"])),
                tickfont=dict(color=THEME_COLORS["text_bright"]),
            ),
        )
    )

    layout = base_layout(f"Quantum Kernel Gram Matrix (First {n}×{n} Sample Pairs)")
    layout.update(
        xaxis=dict(title="Sample Index x_j", color=THEME_COLORS["text_dim"]),
        yaxis=dict(title="Sample Index x_i", autorange="reversed", color=THEME_COLORS["text_dim"]),
    )
    fig.update_layout(layout)
    return fig


def plot_dataset_scatter(df: pd.DataFrame) -> go.Figure:
    """Interactive 2D / 3D scatter plot of climate data colored by risk category."""
    fig = px.scatter(
        df,
        x="temperature",
        y="rainfall",
        size="wind_speed",
        color="risk_category",
        color_discrete_map={
            "Low Risk": THEME_COLORS["cyan"],
            "Medium Risk": THEME_COLORS["amber"],
            "High Risk": THEME_COLORS["red"],
        },
        hover_data=["humidity", "wind_speed", "risk_score"],
        labels={
            "temperature": "Temperature (°C)",
            "rainfall": "Rainfall (mm)",
            "wind_speed": "Wind Speed (km/h)",
            "risk_category": "Risk Category",
        },
    )

    layout = base_layout("Climate Dataset Distribution (Temp vs Rain, Size: Wind)")
    fig.update_layout(layout)
    return fig
