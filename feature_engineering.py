"""
Module 4: Feature Engineering
================================
Derives indicators from raw sensor values that make developing faults
easier for the models to recognize, e.g. rolling averages, rate-of-change,
and cumulative counters.
"""

import numpy as np
import pandas as pd

ROLLING_WINDOW = 6  # hours


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["run_id", "operating_hour"]).copy()

    def _per_run(g):
        g = g.copy()
        g["vibration_avg_1h"] = g[[
            "vibration_motor_mms", "vibration_gearbox_mms", "vibration_bearing_mms"
        ]].mean(axis=1).rolling(ROLLING_WINDOW, min_periods=1).mean()

        g["motor_temp_rise_rate"] = g["motor_temp_C"].diff().rolling(
            ROLLING_WINDOW, min_periods=1).mean().fillna(0)
        g["gearbox_temp_rise_rate"] = g["gearbox_temp_C"].diff().rolling(
            ROLLING_WINDOW, min_periods=1).mean().fillna(0)
        g["brake_temp_rise_rate"] = g["brake_temp_C"].diff().rolling(
            ROLLING_WINDOW, min_periods=1).mean().fillna(0)

        g["current_increase_rate"] = g["motor_current_A"].diff().rolling(
            ROLLING_WINDOW, min_periods=1).mean().fillna(0)

        g["oil_pressure_variation"] = g["gearbox_oil_pressure_bar"].rolling(
            ROLLING_WINDOW, min_periods=1).std().fillna(0)

        g["rope_tension_avg"] = g["rope_tension_kN"].rolling(
            ROLLING_WINDOW, min_periods=1).mean()

        g["load_fluctuation"] = g["load_weight_kg"].rolling(
            ROLLING_WINDOW, min_periods=1).std().fillna(0)

        g["max_vibration_today"] = g[[
            "vibration_motor_mms", "vibration_gearbox_mms", "vibration_bearing_mms"
        ]].max(axis=1).rolling(24, min_periods=1).max()

        g["overload_event"] = (g["load_weight_kg"] > 5500).astype(int)
        g["overload_events_cum"] = g["overload_event"].cumsum()

        return g

    processed_groups = []
    for _, g in df.groupby("run_id"):
        processed_groups.append(_per_run(g))
    df = pd.concat(processed_groups, ignore_index=True)
    return df


FEATURE_COLUMNS = [
    "motor_temp_C", "bearing_temp_C", "gearbox_temp_C", "motor_current_A",
    "motor_voltage_V", "power_kW", "motor_speed_rpm", "hoisting_speed_mps",
    "vibration_motor_mms", "vibration_gearbox_mms", "vibration_bearing_mms",
    "brake_temp_C", "brake_wear_pct", "rope_tension_kN", "load_weight_kg",
    "gearbox_oil_pressure_bar", "gearbox_oil_level_pct",
    "vibration_avg_1h", "motor_temp_rise_rate", "gearbox_temp_rise_rate",
    "brake_temp_rise_rate", "current_increase_rate", "oil_pressure_variation",
    "rope_tension_avg", "load_fluctuation", "max_vibration_today",
    "overload_events_cum", "days_since_maintenance",
]
