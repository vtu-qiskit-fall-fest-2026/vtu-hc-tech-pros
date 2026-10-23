"""
Q-Climate AI - Data Generator Module
------------------------------------
Generates physically grounded synthetic climate datasets for classifying
environmental conditions into Low, Medium, and High climate-risk categories.

Features:
  1. Temperature (°C): Ambient surface temperature [10 - 50 °C]
  2. Humidity (%): Relative atmospheric humidity [15 - 100 %]
  3. Rainfall (mm): 24-hour cumulative precipitation [0 - 300 mm]
  4. Wind Speed (km/h): Sustained surface wind speed [5 - 120 km/h]

Risk Categories:
  - 0: Low Risk   (Mild, safe weather conditions)
  - 1: Medium Risk (Advisory level: moderate storms, high heat index, or gusty winds)
  - 2: High Risk  (Severe weather warning: cyclones, flash floods, or lethal heatwaves)
"""

import os
import numpy as np
import pandas as pd


FEATURE_NAMES = ["temperature", "humidity", "rainfall", "wind_speed"]
FEATURE_DISPLAY_NAMES = {
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "rainfall": "Rainfall (mm)",
    "wind_speed": "Wind Speed (km/h)",
}

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
RISK_COLORS = {0: "#00F2FE", 1: "#F59E0B", 2: "#EF4444"}

FEATURE_RANGES = {
    "temperature": (10.0, 50.0),
    "humidity": (15.0, 100.0),
    "rainfall": (0.0, 300.0),
    "wind_speed": (5.0, 120.0),
}

PRESET_SCENARIOS = {
    "Pleasant Spring Afternoon": {
        "temperature": 23.5,
        "humidity": 45.0,
        "rainfall": 0.0,
        "wind_speed": 12.0,
        "description": "Temperate conditions, mild humidity, zero precipitation.",
    },
    "Humid Summer Day": {
        "temperature": 34.0,
        "humidity": 70.0,
        "rainfall": 10.0,
        "wind_speed": 18.0,
        "description": "Elevated heat index with moderate discomfort.",
    },
    "Monsoon Storm Warning": {
        "temperature": 27.5,
        "humidity": 90.0,
        "rainfall": 95.0,
        "wind_speed": 48.0,
        "description": "Heavy rainfall and gusty winds causing localized waterlogging.",
    },
    "Lethal Heatwave Alert": {
        "temperature": 45.5,
        "humidity": 68.0,
        "rainfall": 0.0,
        "wind_speed": 20.0,
        "description": "Dangerous wet-bulb temperature exceeding safe physiological thresholds.",
    },
    "Severe Tropical Cyclone": {
        "temperature": 26.0,
        "humidity": 95.0,
        "rainfall": 210.0,
        "wind_speed": 98.0,
        "description": "Destructive gale-force winds accompanied by torrential rainfall.",
    },
    "Wildfire / Arid Gale": {
        "temperature": 41.0,
        "humidity": 18.0,
        "rainfall": 0.0,
        "wind_speed": 72.0,
        "description": "Scorching temperatures, dry air, and high winds promoting wildfire spread.",
    },
}


def compute_climate_risk_score(
    temperature: float,
    humidity: float,
    rainfall: float,
    wind_speed: float,
    noise_std: float = 0.0,
    seed: int = None,
) -> tuple[float, int, str]:
    """
    Computes a composite climate risk score in [0.0, 1.0] and categorizes it
    into Low (0), Medium (1), or High (2) risk.

    Meteorological Logic:
      - Storm / Flood Hazard: Weighted combination of rainfall and wind.
      - Heat Stress Index: Joint interaction of ambient heat and humidity.
      - Arid / Wildfire Hazard: High temperature combined with dry air and wind.
      - Severe Compound Threat: High rainfall coupled with cyclonic winds.
    """
    # Min-max normalization against operational physical bounds
    t_min, t_max = FEATURE_RANGES["temperature"]
    h_min, h_max = FEATURE_RANGES["humidity"]
    r_min, r_max = FEATURE_RANGES["rainfall"]
    w_min, w_max = FEATURE_RANGES["wind_speed"]

    t_norm = np.clip((temperature - t_min) / (t_max - t_min), 0.0, 1.0)
    h_norm = np.clip((humidity - h_min) / (h_max - h_min), 0.0, 1.0)
    r_norm = np.clip((rainfall - r_min) / (r_max - r_min), 0.0, 1.0)
    w_norm = np.clip((wind_speed - w_min) / (w_max - w_min), 0.0, 1.0)

    # Sub-indices
    # 1. Storm / Flood: Strong precipitation + wind
    storm_hazard = 0.65 * r_norm + 0.35 * w_norm

    # 2. Heat Stress: Ambient temp + humidity (wet-bulb effect)
    heat_hazard = 0.55 * t_norm + 0.45 * (t_norm * h_norm)

    # 3. Arid Wind / Fire Hazard: High temp + low humidity + high wind
    fire_hazard = 0.50 * t_norm + 0.30 * (1.0 - h_norm) + 0.20 * w_norm

    # 4. Multi-hazard composite peak
    peak_hazard = max(storm_hazard, heat_hazard, fire_hazard)

    # Compound synergy bonus (e.g. monsoon cyclone: both rain and wind high)
    compound_synergy = 0.20 * (r_norm * w_norm) + 0.15 * (t_norm * h_norm * (1.0 if t_norm > 0.6 else 0.0))

    composite_score = peak_hazard + compound_synergy

    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        composite_score += rng.normal(0.0, noise_std)

    composite_score = float(np.clip(composite_score, 0.0, 1.0))

    # Thresholds:
    # < 0.38 -> Low Risk (0)
    # 0.38 <= score < 0.68 -> Medium Risk (1)
    # >= 0.68 -> High Risk (2)
    if composite_score < 0.38:
        risk_class = 0
    elif composite_score < 0.68:
        risk_class = 1
    else:
        risk_class = 2

    return composite_score, risk_class, RISK_LABELS[risk_class]


def generate_climate_dataset(
    n_samples: int = 80,
    random_state: int = 42,
    noise_std: float = 0.03,
) -> pd.DataFrame:
    """
    Generates a synthetic yet physically correlated dataset of meteorological samples.
    Ensures a balanced representation of Low, Medium, and High risk profiles.

    Dataset size is intentionally compact (e.g., 60-100 samples) so that genuine
    quantum-kernel simulations evaluate quickly during interactive hackathon demos.
    """
    rng = np.random.default_rng(random_state)
    records = []

    # Partition samples into weather archetypes to ensure realistic distributions
    archetypes = [
        # (name, count, t_range, h_range, r_range, w_range)
        ("mild_temperate", int(n_samples * 0.30), (16.0, 26.0), (30.0, 60.0), (0.0, 15.0), (8.0, 22.0)),
        ("summer_heat", int(n_samples * 0.22), (32.0, 42.0), (45.0, 75.0), (0.0, 35.0), (12.0, 28.0)),
        ("monsoon_rain", int(n_samples * 0.22), (24.0, 31.0), (75.0, 98.0), (45.0, 160.0), (25.0, 55.0)),
        ("extreme_cyclone_heatwave", n_samples - int(n_samples * 0.30) - 2 * int(n_samples * 0.22), (28.0, 48.0), (20.0, 98.0), (80.0, 280.0), (55.0, 115.0)),
    ]

    for _, count, (t_min, t_max), (h_min, h_max), (r_min, r_max), (w_min, w_max) in archetypes:
        for _ in range(count):
            temp = float(rng.uniform(t_min, t_max))
            hum = float(rng.uniform(h_min, h_max))
            rain = float(rng.uniform(r_min, r_max))
            wind = float(rng.uniform(w_min, w_max))

            score, label, label_name = compute_climate_risk_score(
                temp, hum, rain, wind, noise_std=noise_std, seed=int(rng.integers(0, 100000))
            )

            records.append({
                "temperature": round(temp, 2),
                "humidity": round(hum, 1),
                "rainfall": round(rain, 1),
                "wind_speed": round(wind, 1),
                "risk_score": round(score, 4),
                "risk_label": label,
                "risk_category": label_name,
            })

    df = pd.DataFrame(records)
    # Shuffle records
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def ensure_dataset_exists(filepath: str = "data/climate_risk_dataset.csv", n_samples: int = 80) -> pd.DataFrame:
    """Loads existing dataset or creates and saves one."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        df = generate_climate_dataset(n_samples=n_samples, random_state=42)
        df.to_csv(filepath, index=False)
    return df


if __name__ == "__main__":
    df = ensure_dataset_exists("data/climate_risk_dataset.csv", n_samples=80)
    print("Dataset generated successfully:")
    print(df["risk_category"].value_counts())
    print(df.head())
