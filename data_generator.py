"""
Module 1: Data Acquisition (Synthetic)
========================================
Simulates sensor data packets that would normally arrive from an ESP32 /
industrial PLC monitoring the Old Nic Mine hoisting system.

Since no live sensor feed exists yet, this generator creates realistic
run-to-failure sequences for four critical components:
    - Motor
    - Gearbox
    - Brake System
    - Hoist Rope / Bearings

Each "run" starts at 100% health and degrades to failure (0% health) over a
random operating-hour lifespan. Sensor readings are generated so that they
correlate with the underlying health of the degrading component, with
realistic noise added. This mirrors the structure of the sensor list given
in the project's AI Purpose & Programming Specification.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

COMPONENTS = ["Motor", "Gearbox", "Brake System", "Bearings", "Hoist Rope"]

# Typical healthy operating baselines (mean, std) per sensor
BASELINE = {
    "motor_temp_C": (55, 3),
    "bearing_temp_C": (48, 3),
    "gearbox_temp_C": (50, 3),
    "motor_current_A": (85, 5),
    "motor_voltage_V": (400, 5),
    "power_kW": (60, 4),
    "motor_speed_rpm": (980, 15),
    "hoisting_speed_mps": (8.0, 0.3),
    "vibration_motor_mms": (1.8, 0.2),
    "vibration_gearbox_mms": (2.0, 0.2),
    "vibration_bearing_mms": (1.6, 0.2),
    "brake_temp_C": (60, 4),
    "brake_wear_pct": (5, 2),
    "rope_tension_kN": (110, 6),
    "load_weight_kg": (4500, 400),
    "gearbox_oil_pressure_bar": (4.2, 0.15),
    "gearbox_oil_level_pct": (95, 2),
    "ambient_temp_C": (22, 3),
    "humidity_pct": (55, 8),
}

# Which sensors are most affected by degradation of each component,
# and the direction/magnitude of drift from healthy -> failed state.
DEGRADATION_EFFECTS = {
    "Motor": {
        "motor_temp_C": 35, "motor_current_A": 40, "vibration_motor_mms": 6.5,
        "power_kW": 18,
    },
    "Gearbox": {
        "gearbox_temp_C": 32, "vibration_gearbox_mms": 7.0,
        "gearbox_oil_pressure_bar": -2.0, "gearbox_oil_level_pct": -20,
    },
    "Brake System": {
        "brake_temp_C": 55, "brake_wear_pct": 85,
    },
    "Bearings": {
        "bearing_temp_C": 30, "vibration_bearing_mms": 7.5,
    },
    "Hoist Rope": {
        "rope_tension_kN": 40,  # tension imbalance grows
    },
}

LIFE_RANGES_HOURS = {
    "Motor": (300, 500),
    "Gearbox": (260, 450),
    "Brake System": (90, 180),
    "Bearings": (220, 380),
    "Hoist Rope": (350, 650),
}


def _sigmoid_progress(t, total, steepness=8):
    """Non-linear degradation curve: slow at first, accelerates near failure."""
    x = (t / total - 0.6) * steepness
    return 1 / (1 + np.exp(-x))


def _generate_run(machine_id, run_id, failing_component, start_time):
    total_life = int(RNG.integers(*LIFE_RANGES_HOURS[failing_component]))
    hours = np.arange(0, total_life)
    n = len(hours)

    df = pd.DataFrame({"operating_hour": hours})
    df["machine_id"] = machine_id
    df["run_id"] = run_id
    df["failing_component"] = failing_component
    df["timestamp"] = pd.to_datetime(start_time) + pd.to_timedelta(hours, unit="h")

    # health goes from 100 -> 0 following an accelerating decay curve
    progress = _sigmoid_progress(hours, total_life)
    df["true_health_pct"] = np.clip(100 * (1 - progress) + RNG.normal(0, 1.0, n), 0, 100)
    df["true_rul_hours"] = total_life - hours

    effects = DEGRADATION_EFFECTS[failing_component]

    for sensor, (mean, std) in BASELINE.items():
        drift = effects.get(sensor, 0) * progress
        noise = RNG.normal(0, std, n)
        df[sensor] = mean + drift + noise

    # Load / ambient conditions vary independently of degradation
    df["load_weight_kg"] = np.clip(
        RNG.normal(BASELINE["load_weight_kg"][0], BASELINE["load_weight_kg"][1], n), 500, 6000
    )
    df["ambient_temp_C"] = RNG.normal(*BASELINE["ambient_temp_C"], n)
    df["humidity_pct"] = np.clip(RNG.normal(*BASELINE["humidity_pct"], n), 10, 100)

    # Maintenance history reference (simple counter of days since last service)
    df["days_since_maintenance"] = (hours / 24).astype(int)

    return df


def generate_dataset(n_runs_per_component=15, machines=("HOIST-01", "HOIST-02", "HOIST-03")):
    """
    Builds a synthetic fleet dataset: multiple run-to-failure sequences per
    component, spread across several hoist machines.
    """
    all_runs = []
    run_id = 0
    start = pd.Timestamp("2024-01-01")

    for component in COMPONENTS:
        for _ in range(n_runs_per_component):
            machine_id = RNG.choice(machines)
            run_df = _generate_run(machine_id, run_id, component, start)
            all_runs.append(run_df)
            run_id += 1
            start = start + pd.Timedelta(days=int(RNG.integers(5, 20)))

    full = pd.concat(all_runs, ignore_index=True)

    # Introduce a small amount of missingness / bad readings for the
    # Data Validation module to catch (Module 2 in the spec).
    corrupt_idx = RNG.choice(full.index, size=int(0.01 * len(full)), replace=False)
    sensor_cols = list(BASELINE.keys())
    for idx in corrupt_idx:
        col = RNG.choice(sensor_cols)
        choice = RNG.random()
        if choice < 0.4:
            full.loc[idx, col] = np.nan
        elif choice < 0.7:
            full.loc[idx, col] = full[col].mean() * 20  # impossible spike
        else:
            full.loc[idx, col] = -abs(full.loc[idx, col])  # impossible negative

    return full


if __name__ == "__main__":
    data = generate_dataset()
    print(f"Generated {len(data):,} sensor records across {data['run_id'].nunique()} runs "
          f"and {data['machine_id'].nunique()} machines.")
    data.to_csv("synthetic_hoist_data.csv", index=False)
    print("Saved to synthetic_hoist_data.csv")
