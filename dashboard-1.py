"""
Web Dashboard for the Hoist Predictive Maintenance System
=============================================================
Run with:
    streamlit run dashboard.py

Tabs:
  1. Live Dashboard   - component health cards + recommendations + degradation
                        trend, with a time-scrubber so you can see the fleet
                        at any point in its life (not just near failure).
  2. Model Performance - fault classification accuracy + RUL prediction error,
                        addressing the proposal's requirement to "evaluate
                        the effectiveness of the proposed system."
  3. Downtime Impact  - estimates how much advance warning the AI gives before
                        failure, and an illustrative downtime-reduction figure,
                        addressing the proposal's research question on
                        reducing unplanned downtime.
  4. Log Maintenance Outcome - Module 12 (Continuous Learning) from the spec:
                        lets a technician record what was actually found, so
                        those records can extend the training set over time.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

from data_generator import generate_dataset, COMPONENTS, LIFE_RANGES_HOURS
from data_pipeline import validate_data, preprocess_data
from feature_engineering import engineer_features, FEATURE_COLUMNS
from models import AnomalyDetector, FaultClassifier, RULPredictor, train_test_split_by_run
from health_engine import assess_component

st.set_page_config(page_title="Old Nic Mine — Hoist Predictive Maintenance", layout="wide")

STATUS_COLORS = {
    "Normal operation": "#2ecc71",
    "Monitor closely": "#f1c40f",
    "Schedule maintenance": "#e67e22",
    "Immediate inspection": "#e74c3c",
}

# Illustrative assumptions used only for the Downtime Impact estimate below.
# These are NOT measured facts about Old Nic Mine -- they are placeholders
# so the calculation can be shown; replace with real figures once available.
ASSUMED_UNPLANNED_DOWNTIME_HOURS = 48
ASSUMED_PLANNED_DOWNTIME_HOURS = 6


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

    # Whole-test-set predictions, computed once and reused by every tab.
    fault_probs = clf.predict_proba(test_df[FEATURE_COLUMNS])
    healthy_idx = list(clf.model.classes_).index("Healthy") if "Healthy" in clf.model.classes_ else None
    test_df["fault_probability"] = (
        1 - fault_probs[:, healthy_idx] if healthy_idx is not None else fault_probs.max(axis=1)
    )

    return detector, clf, rul_model, train_df, test_df


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


def pick_row_at_pct(rows: pd.DataFrame, pct: float):
    """Given all rows for one component (possibly several runs), pick a single
    representative run (deterministically, so it doesn't jump around every
    rerun) and return the row whose operating_hour is closest to `pct`% of
    that run's total life."""
    run_id = sorted(rows["run_id"].unique())[0]
    run_rows = rows[rows["run_id"] == run_id].sort_values("operating_hour")
    target_hour = (pct / 100.0) * run_rows["operating_hour"].max()
    idx = (run_rows["operating_hour"] - target_hour).abs().idxmin()
    return run_rows.loc[[idx]], run_id


# ---------------------------------------------------------------- Data ----

st.title("⛏️ Old Nic Mine — Hoist Predictive Maintenance Dashboard")
st.caption(
    "AI-based predictive maintenance system for mine hoisting machines. "
    "Currently running on synthetic sensor data (see README) until live "
    "sensor feeds are connected."
)

df = load_data()
detector, clf, rul_model, train_df, test_df = train(df)

machines = sorted(test_df["machine_id"].unique())
selected_machine = st.sidebar.selectbox("Select hoist machine", machines)

st.sidebar.markdown("---")
sim_pct = st.sidebar.slider(
    "Simulated operating time (% of component life)",
    min_value=0, max_value=100, value=40, step=1,
    help="Scrub through a component's life to see how its health card and "
         "recommendation change over time, from brand new to near failure.",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Status legend**\n\n"
    "🟢 Normal operation\n\n"
    "🟡 Monitor closely\n\n"
    "🟠 Schedule maintenance\n\n"
    "🔴 Immediate inspection"
)

machine_runs = test_df[test_df["machine_id"] == selected_machine]

tab_live, tab_perf, tab_impact, tab_log = st.tabs(
    ["📊 Live Dashboard", "🧪 Model Performance", "⏱️ Downtime Impact", "📝 Log Maintenance Outcome"]
)

# ------------------------------------------------------------ Tab 1 ----

with tab_live:
    st.subheader(f"Component Health — {selected_machine} (at {sim_pct}% of component life)")
    cols = st.columns(len(COMPONENTS))

    assessments = {}
    for i, component in enumerate(COMPONENTS):
        comp_rows = machine_runs[machine_runs["failing_component"] == component]
        if comp_rows.empty:
            comp_rows = test_df[test_df["failing_component"] == component]

        sample, run_id = pick_row_at_pct(comp_rows, sim_pct)
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
    trend_component = st.selectbox("Select component to inspect", COMPONENTS, key="trend_component")
    trend_rows = machine_runs[machine_runs["failing_component"] == trend_component]
    if trend_rows.empty:
        trend_rows = test_df[test_df["failing_component"] == trend_component]

    if not trend_rows.empty:
        run_id = sorted(trend_rows["run_id"].unique())[0]
        run_data = df[df["run_id"] == run_id].sort_values("operating_hour")
        chart_df = run_data.set_index("operating_hour")[["true_health_pct"]]
        chart_df.columns = ["Health Score (%)"]
        st.line_chart(chart_df)
        st.caption(
            f"Simulated health trajectory for a {trend_component} run-to-failure sequence "
            f"({run_data['operating_hour'].max():.0f} operating hours to failure). "
            f"The sidebar slider's {sim_pct}% marker sits at "
            f"~{(sim_pct/100)*run_data['operating_hour'].max():.0f} operating hours on this chart."
        )

# ------------------------------------------------------------ Tab 2 ----

with tab_perf:
    st.subheader("Fault Classification Performance")
    st.caption(
        "Accuracy of identifying *which* component is developing a fault, "
        "measured on hoist runs the models never saw during training."
    )
    acc, report = clf.evaluate(test_df[FEATURE_COLUMNS], test_df["fault_label"])
    st.metric("Overall classification accuracy", f"{acc * 100:.1f}%")
    st.text(report)

    st.markdown("---")
    st.subheader("Remaining Useful Life (RUL) Prediction Error")
    st.caption(
        "Mean Absolute Error (MAE) in operating hours between the AI's predicted "
        "RUL and the true simulated RUL, per component."
    )
    mae_by_component = rul_model.evaluate(test_df, FEATURE_COLUMNS)
    mae_df = pd.DataFrame(
        {"Component": list(mae_by_component.keys()), "MAE (hours)": list(mae_by_component.values())}
    ).set_index("Component")
    st.bar_chart(mae_df)
    st.dataframe(mae_df.style.format({"MAE (hours)": "{:.1f}"}), use_container_width=True)

    st.info(
        "These numbers come from synthetic run-to-failure data, so treat them as "
        "a proof of concept for the pipeline architecture rather than a claim "
        "about real-world accuracy. Retrain on real Old Nic Mine sensor and "
        "maintenance records once available -- the code does not need to change."
    )

# ------------------------------------------------------------ Tab 3 ----

with tab_impact:
    st.subheader("Advance Warning Lead Time")
    st.caption(
        "For each simulated run, this finds the earliest point where the AI's "
        "fault probability for the affected component crosses 60% (the "
        "'Schedule maintenance' / 'High priority' threshold), and measures how "
        "many operating hours remained before actual failure at that point. "
        "A larger number means more advance warning."
    )

    lead_times = {}
    for component in COMPONENTS:
        comp_df = test_df[test_df["failing_component"] == component].sort_values(
            ["run_id", "operating_hour"]
        )
        run_lead_times = []
        for run_id, run_rows in comp_df.groupby("run_id"):
            triggered = run_rows[run_rows["fault_probability"] > 0.6]
            if not triggered.empty:
                first_trigger = triggered.iloc[0]
                run_lead_times.append(first_trigger["true_rul_hours"])
        if run_lead_times:
            lead_times[component] = float(np.mean(run_lead_times))

    if lead_times:
        lead_df = pd.DataFrame(
            {"Component": list(lead_times.keys()), "Avg. lead time (hours)": list(lead_times.values())}
        ).set_index("Component")
        st.bar_chart(lead_df)
        st.dataframe(lead_df.style.format({"Avg. lead time (hours)": "{:.1f}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("Illustrative Downtime Reduction Estimate")
    st.warning(
        "The figures below use **assumed** downtime durations, not measured data "
        "from Old Nic Mine. They exist to demonstrate how the calculation would "
        "work once real maintenance-cost figures are available -- update the "
        "two constants at the top of dashboard.py with real numbers before "
        "citing this anywhere official."
    )
    col1, col2 = st.columns(2)
    col1.metric("Assumed unplanned-failure downtime", f"{ASSUMED_UNPLANNED_DOWNTIME_HOURS} hrs")
    col2.metric("Assumed planned-maintenance downtime", f"{ASSUMED_PLANNED_DOWNTIME_HOURS} hrs")
    saved = ASSUMED_UNPLANNED_DOWNTIME_HOURS - ASSUMED_PLANNED_DOWNTIME_HOURS
    st.metric(
        "Estimated downtime avoided per correctly predicted failure",
        f"{saved} hrs",
        help="Unplanned downtime minus planned downtime, under the assumptions above.",
    )

# ------------------------------------------------------------ Tab 4 ----

with tab_log:
    st.subheader("Log a Maintenance Outcome")
    st.caption(
        "Spec Module 12 (Continuous Learning): after maintenance, technicians "
        "record what was actually found and whether the AI's prediction was "
        "correct. Over time, these records extend the training data so future "
        "models improve. Note: on this free hosting tier, logged entries persist "
        "only for your current browser session -- export them with the button "
        "below before closing the tab. In production this would write to a "
        "real database."
    )

    if "maintenance_log" not in st.session_state:
        st.session_state.maintenance_log = []

    with st.form("log_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            log_machine = st.selectbox("Machine", machines)
            log_component = st.selectbox("Component", COMPONENTS)
            log_date = st.date_input("Date maintenance performed")
        with c2:
            log_prediction_correct = st.selectbox(
                "Was the AI's prediction correct?", ["Yes", "No", "Partially"]
            )
            log_operating_hours = st.number_input("Operating hours at time of maintenance", min_value=0)
        log_fault_found = st.text_area("Actual fault found / action taken")
        submitted = st.form_submit_button("Save record")

        if submitted:
            st.session_state.maintenance_log.append({
                "machine": log_machine,
                "component": log_component,
                "date": str(log_date),
                "operating_hours": log_operating_hours,
                "ai_prediction_correct": log_prediction_correct,
                "fault_found": log_fault_found,
            })
            st.success("Record saved for this session.")

    if st.session_state.maintenance_log:
        st.markdown("---")
        st.subheader("Logged Records (this session)")
        log_df = pd.DataFrame(st.session_state.maintenance_log)
        st.dataframe(log_df, use_container_width=True)

        csv_buffer = io.StringIO()
        log_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download log as CSV",
            data=csv_buffer.getvalue(),
            file_name="maintenance_log.csv",
            mime="text/csv",
        )
    else:
        st.caption("No records logged yet this session.")
