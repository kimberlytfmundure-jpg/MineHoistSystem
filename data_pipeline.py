"""
Module 2: Data Validation
Module 3: Data Preprocessing
==============================
Cleans the raw sensor feed before it reaches feature engineering / models.
"""

import numpy as np
import pandas as pd

# Physically impossible or implausible bounds for each sensor.
# Anything outside these is rejected and treated as missing, then
# repaired using interpolation (as specified in Module 2).
VALID_RANGES = {
    "motor_temp_C": (-10, 200),
    "bearing_temp_C": (-10, 200),
    "gearbox_temp_C": (-10, 200),
    "motor_current_A": (0, 500),
    "motor_voltage_V": (0, 600),
    "power_kW": (0, 300),
    "motor_speed_rpm": (0, 2000),
    "hoisting_speed_mps": (0, 20),
    "vibration_motor_mms": (0, 50),
    "vibration_gearbox_mms": (0, 50),
    "vibration_bearing_mms": (0, 50),
    "brake_temp_C": (-10, 250),
    "brake_wear_pct": (0, 100),
    "rope_tension_kN": (0, 400),
    "load_weight_kg": (0, 10000),
    "gearbox_oil_pressure_bar": (0, 10),
    "gearbox_oil_level_pct": (0, 100),
    "ambient_temp_C": (-20, 60),
    "humidity_pct": (0, 100),
}


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Module 2: Data Validation.
    Detects impossible values, negative values where not physical, and
    marks them as missing (NaN) so they can be repaired downstream.
    Also flags duplicate records.
    """
    df = df.copy()
    report = {"impossible_values": 0, "duplicates_removed": 0}

    before = len(df)
    df = df.drop_duplicates(subset=["machine_id", "run_id", "operating_hour"])
    report["duplicates_removed"] = before - len(df)

    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        report["impossible_values"] += int(mask.sum())
        df.loc[mask, col] = np.nan

    df.attrs["validation_report"] = report
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Module 3: Data Preprocessing.
    Repairs missing values via per-run linear interpolation (falling back
    to forward/backward fill at run edges), and ensures data is sorted by
    machine/run/time as required for time-series feature engineering.
    """
    df = df.sort_values(["run_id", "operating_hour"]).copy()

    sensor_cols = [c for c in VALID_RANGES if c in df.columns]

    fixed_groups = []
    for _, g in df.groupby("run_id"):
        g = g.copy()
        g[sensor_cols] = g[sensor_cols].interpolate(limit_direction="both")
        fixed_groups.append(g)
    df = pd.concat(fixed_groups, ignore_index=True)

    # Any still-missing values (e.g. whole-column gaps) get the global median
    for col in sensor_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df
