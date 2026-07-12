"""
Web Dashboard for the Hoist Predictive Maintenance System
=============================================================
Run with:
    streamlit run dashboard.py

This wraps the existing pipeline (data_generator, data_pipeline,
feature_engineering, models, health_engine) in a browser-based interface,
so instead of reading printed text in a terminal, you get:
  - A colour-coded card per component (green/amber/red)
  - Health score and fault probability gauges
  - A degradation chart showing health trending toward failure
  - A machine selector so you can inspect different hoists
"""

import streamlit as st
import pandas as pd
import numpy as np

from data_generator import generate_dataset, COMPONENTS, LIFE_RANGES_HOURS
from data_pipeline import validate_data, preprocess_data
from feature_engineering import engineer_features, FEATURE_COLUMNS
from models import AnomalyDetector, FaultClassifier, RULPredictor, train_test_split_by_run
from health_engine import assess_component

st.set_page_config(page_title="Hoist Predictive Maintenance", layout="wide")

STATUS_COLORS = {
    "Normal operation": "#2ecc71",
    "Monitor closely": "#f1c40f",
    "Schedule maintenance": "#e67e22",
    "Immediate inspection": "#e74c3c",
}


@st.cache_data(show_spinner="Generating synthetic sensor data and processing it...")
def load_data():
    raw = generate_dataset(n_runs_per_component=25)
    validated = validate_data(raw)
    clean = preprocess_data(validated)
    featured = engineer_features(clean)
    return featured


@st.cache_resource(show_spinner="Training models (anomaly detector, fault classifier, RUL regressors)...")
def train(df):
    train_df, test_df = train_test_split_by_run(df, test_size=0.25)

    healthy_mask = train_df["true_health_pct"] > 90
    detector = AnomalyDetector(contamination=0.05).fit(train_df.loc[healthy_mask, FEATURE_COLUMNS])
    _, anom_score = detector.predict(test_df[FEATURE_COLUMNS])
    test_df = test_df.copy()
    test_df["anomaly_score"] = anom_score

    def _label(row):
        return row["failing_component"] if row["true_health_pct"] < 90 else "Healthy"

    train_df = train_df.copy()
    train_df["fault_label"] = train_df.apply(_label, axis=1)
    test_df["fault_label"] = test_df.apply(_label, axis=1)

    clf = FaultClassifier().fit(train_df[FEATURE_COLUMNS], train_df["fault_label"])
    rul_model = RULPredictor().fit(train_df, FEATURE_COLUMNS)

    return detector, clf, rul_model, test_df


def get_assessment(component, row, sample, clf, rul_model):
    X_row = sample[FEATURE_COLUMNS]
    fault_probs = clf.predict_proba(X_row)[0]
    prob_map = dict(zip(clf.model.classes_, fault_probs))
    fault_probability = 1 - prob_map.get("Healthy", 0)

    rul_pred = max(0, rul_model.predict(component, X_row)[0])
    max_life = np.mean(LIFE_RANGES_HOURS[component])

    return assess_component(
        component=component,
        anomaly_score=row["anomaly_score"],
        fault_probability=fault_probability,
        rul_hours=rul_pred,
        max_expected_life_hours=max_life,
    )


# ---------------------------------------------------------------- UI ----

st.title("⛏️ Old Nic Mine — Hoist Predictive Maintenance Dashboard")
st.caption(
    "AI-based predictive maintenance system for mine hoisting machines. "
    "Currently running on synthetic sensor data (see README) until live "
    "sensor feeds are connected."
)

df = load_data()
detector, clf, rul_model, test_df = train(df)

machines = sorted(test_df["machine_id"].unique())
selected_machine = st.sidebar.selectbox("Select hoist machine", machines)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Status legend**\n\n"
    "🟢 Normal operation\n\n"
    "🟡 Monitor closely\n\n"
    "🟠 Schedule maintenance\n\n"
    "🔴 Immediate inspection"
)

machine_runs = test_df[test_df["machine_id"] == selected_machine]

st.subheader(f"Component Health — {selected_machine}")
cols = st.columns(len(COMPONENTS))

assessments = {}
for i, component in enumerate(COMPONENTS):
    comp_rows = machine_runs[machine_runs["failing_component"] == component]
    if comp_rows.empty:
        comp_rows = test_df[test_df["failing_component"] == component]

    # use the most recent (highest operating_hour) reading available as "now"
    sample = comp_rows.sort_values("operating_hour").tail(1)
    row = sample.iloc[0]
    assessment = get_assessment(component, row, sample, clf, rul_model)
    assessments[component] = (assessment, sample)

    color = STATUS_COLORS.get(assessment.status, "#95a5a6")
    with cols[i]:
        st.markdown(
            f"""
            <div style="border:2px solid {color}; border-radius:10px; padding:14px; text-align:center;">
                <h4 style="margin:0;">{component}</h4>
                <h1 style="color:{color}; margin:6px 0;">{assessment.health_score}%</h1>
                <p style="margin:0;"><b>{assessment.status}</b></p>
                <p style="margin:4px 0; font-size:0.85em;">Priority: {assessment.priority}</p>
                <p style="margin:4px 0; font-size:0.85em;">Fault probability: {assessment.fault_probability}%</p>
                <p style="margin:4px 0; font-size:0.85em;">RUL: {assessment.estimated_rul_hours:.0f} hrs</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.subheader("Maintenance Recommendations")
for component, (assessment, _) in assessments.items():
    color = STATUS_COLORS.get(assessment.status, "#95a5a6")
    st.markdown(
        f"**{component}** — <span style='color:{color}'>{assessment.priority}</span>: "
        f"{assessment.recommendation}",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.subheader("Component Degradation Trend")
trend_component = st.selectbox("Select component to inspect", COMPONENTS)
trend_rows = machine_runs[machine_runs["failing_component"] == trend_component]
if trend_rows.empty:
    trend_rows = test_df[test_df["failing_component"] == trend_component]

if not trend_rows.empty:
    run_id = trend_rows["run_id"].iloc[0]
    run_data = df[df["run_id"] == run_id].sort_values("operating_hour")
    chart_df = run_data.set_index("operating_hour")[["true_health_pct"]]
    chart_df.columns = ["Health Score (%)"]
    st.line_chart(chart_df)
    st.caption(
        f"Simulated health trajectory for a {trend_component} run-to-failure sequence "
        f"({run_data['operating_hour'].max():.0f} operating hours to failure)."
    )
