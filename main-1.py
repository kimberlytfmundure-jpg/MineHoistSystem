"""
Programming Workflow (per the AI Purpose & Programming Specification):
1. Receive sensor data (simulated here)      -> data_generator.py
2. Validate incoming data                    -> data_pipeline.validate_data
3. Clean and preprocess                      -> data_pipeline.preprocess_data
4. Calculate additional features             -> feature_engineering.engineer_features
5. Store processed data                      -> CSV (stand-in for a database)
6. Run anomaly detection                     -> models.AnomalyDetector
7. Classify detected faults                  -> models.FaultClassifier
8. Predict remaining useful life             -> models.RULPredictor
9. Calculate health scores                   -> health_engine.calculate_health_score
10. Generate maintenance recommendations     -> health_engine.assess_component
11. Display results on the dashboard         -> print_dashboard() below
"""

import numpy as np
import pandas as pd

from data_generator import generate_dataset, COMPONENTS, LIFE_RANGES_HOURS
from data_pipeline import validate_data, preprocess_data
from feature_engineering import engineer_features, FEATURE_COLUMNS
from models import AnomalyDetector, FaultClassifier, RULPredictor, train_test_split_by_run
from health_engine import assess_component, format_dashboard_card

pd.set_option("display.width", 120)


def build_pipeline_dataset():
    print("Step 1: Generating synthetic sensor data (simulating ESP32/PLC feed)...")
    raw = generate_dataset(n_runs_per_component=25)
    print(f"  -> {len(raw):,} records, {raw['run_id'].nunique()} runs, "
          f"{raw['machine_id'].nunique()} machines")

    print("Step 2: Validating data (rejecting impossible values, duplicates)...")
    validated = validate_data(raw)
    report = validated.attrs.get("validation_report", {})
    print(f"  -> Flagged {report.get('impossible_values', 0)} impossible values, "
          f"removed {report.get('duplicates_removed', 0)} duplicates")

    print("Step 3: Preprocessing (interpolating missing values)...")
    clean = preprocess_data(validated)
    sensor_cols = [c for c in FEATURE_COLUMNS if c in clean.columns]
    print(f"  -> Remaining NaNs: {int(clean[sensor_cols].isna().sum().sum())}")

    print("Step 4: Engineering features (rolling averages, rates of change)...")
    featured = engineer_features(clean)

    print("Step 5: Storing processed dataset (processed_hoist_data.csv)...")
    featured.to_csv("processed_hoist_data.csv", index=False)

    return featured


def train_models(df: pd.DataFrame):
    train_df, test_df = train_test_split_by_run(df, test_size=0.25)

    print("\nStep 6: Training anomaly detector on healthy (early-life) data...")
    healthy_mask = train_df["true_health_pct"] > 90
    detector = AnomalyDetector(contamination=0.05).fit(train_df.loc[healthy_mask, FEATURE_COLUMNS])
    is_anom, anom_score = detector.predict(test_df[FEATURE_COLUMNS])
    test_df = test_df.copy()
    test_df["anomaly_score"] = anom_score
    flagged_rate = is_anom.mean() * 100
    print(f"  -> Flagged {flagged_rate:.1f}% of test readings as anomalous")

    print("\nStep 7: Training fault classifier (identifying affected component)...")
    # Label: which component is degrading, or 'Healthy' if health is still high
    def _label(row):
        return row["failing_component"] if row["true_health_pct"] < 90 else "Healthy"

    train_df = train_df.copy()
    train_df["fault_label"] = train_df.apply(_label, axis=1)
    test_df["fault_label"] = test_df.apply(_label, axis=1)

    clf = FaultClassifier().fit(train_df[FEATURE_COLUMNS], train_df["fault_label"])
    acc, report = clf.evaluate(test_df[FEATURE_COLUMNS], test_df["fault_label"])
    print(f"  -> Test accuracy: {acc * 100:.1f}%")
    print(report)

    print("Step 8: Training Remaining Useful Life (RUL) regressors per component...")
    rul_model = RULPredictor().fit(train_df, FEATURE_COLUMNS)
    mae_by_component = rul_model.evaluate(test_df, FEATURE_COLUMNS)
    for comp, mae in mae_by_component.items():
        print(f"  -> {comp}: MAE = {mae:.1f} hours")

    return detector, clf, rul_model, test_df


def print_dashboard(detector, clf, rul_model, test_df):
    print("\n" + "=" * 60)
    print("Step 9-11: HOIST HEALTH DASHBOARD (sample current readings)")
    print("=" * 60)

    # Show one healthy snapshot first (early-life reading), to demonstrate
    # the full range of dashboard output -- mirrors the spec's own example
    # of a healthy Main Hoist Gearbox card.
    healthy_rows = test_df[(test_df["failing_component"] == "Gearbox") & (test_df["true_health_pct"] > 90)]
    snapshots = []
    if not healthy_rows.empty:
        snapshots.append(("Gearbox", healthy_rows.sample(1, random_state=2)))

    # Then one "current" snapshot per component: a row somewhere in the
    # degrading region, to demonstrate realistic mid/late-degradation readings.
    for component in COMPONENTS:
        comp_rows = test_df[test_df["failing_component"] == component]
        if comp_rows.empty:
            continue
        candidates = comp_rows[(comp_rows["true_health_pct"] > 25) & (comp_rows["true_health_pct"] < 70)]
        sample = candidates.sample(1, random_state=1) if not candidates.empty else comp_rows.sample(1, random_state=1)
        snapshots.append((component, sample))

    for component, sample in snapshots:
        row = sample.iloc[0]

        X_row = sample[FEATURE_COLUMNS]
        fault_probs = clf.predict_proba(X_row)[0]
        class_labels = clf.model.classes_
        prob_map = dict(zip(class_labels, fault_probs))
        fault_probability = 1 - prob_map.get("Healthy", 0)

        rul_pred = rul_model.predict(component, X_row)[0]
        rul_pred = max(0, rul_pred)

        max_life = np.mean(LIFE_RANGES_HOURS[component])
        assessment = assess_component(
            component=component,
            anomaly_score=row["anomaly_score"],
            fault_probability=fault_probability,
            rul_hours=rul_pred,
            max_expected_life_hours=max_life,
        )
        print()
        print(format_dashboard_card(assessment))
        print(f"  (reference: true health at this point was {row['true_health_pct']:.0f}%, "
              f"true RUL {row['true_rul_hours']:.0f}h)")


if __name__ == "__main__":
    dataset = build_pipeline_dataset()
    detector, clf, rul_model, test_df = train_models(dataset)
    print_dashboard(detector, clf, rul_model, test_df)
    print("\nPipeline complete. Processed data saved to processed_hoist_data.csv")
