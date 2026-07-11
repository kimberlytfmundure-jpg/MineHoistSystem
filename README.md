# AI-Based Predictive Maintenance System — Old Nic Mine Hoisting System

Working implementation of the pipeline described in your two project
documents (Project Proposal + AI Purpose & Programming Specification).
Since no live sensor feed exists yet, this uses realistic **synthetic
run-to-failure data** so the full pipeline — data validation through to
health scores and maintenance recommendations — can be built, trained, and
demonstrated end to end. When real sensor data becomes available (from the
ESP32/PLC), it can be swapped in for `data_generator.py` with no changes
needed to the rest of the pipeline.

## How to run

### Option A: Terminal output
```bash
pip install -r requirements.txt
python3 main.py
```

### Option B: Interactive web dashboard (recommended for demos)
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```
This opens a browser tab with colour-coded component cards, a machine
selector, maintenance recommendations, and a live degradation chart. See
"Running the web dashboard" below for full details.

This will:
1. Generate synthetic sensor data for 5 components (Motor, Gearbox, Brake
   System, Bearings, Hoist Rope) across 3 hoist machines.
2. Validate and clean the data (rejecting impossible sensor values).
3. Engineer features (rolling vibration averages, temperature rise rates, etc).
4. Train an Isolation Forest anomaly detector on healthy operating data.
5. Train a Random Forest fault classifier (which component is failing).
6. Train per-component Gradient Boosting regressors to predict Remaining
   Useful Life (RUL) in operating hours.
7. Print a dashboard-style report per component, e.g.:

```
Brake System
  Health Score: 30.9%
  Fault Probability: 97.9%
  Estimated Failure / RUL: Within 55 operating hours
  Status: Immediate inspection  (Priority: Critical)
  Recommendation: Replace brake pads and inspect brake actuators immediately.
```

## File structure (maps directly onto the spec's 12 modules)

| File                     | Spec module(s) covered                                   |
|--------------------------|------------------------------------------------------------|
| `data_generator.py`      | Module 1 — Data Acquisition (synthetic stand-in for ESP32/PLC feed) |
| `data_pipeline.py`       | Modules 2–3 — Data Validation & Preprocessing              |
| `feature_engineering.py` | Module 4 — Feature Engineering                              |
| `models.py`              | Modules 5–8 — Normal behaviour learning, Fault Detection, Fault Classification, RUL Prediction |
| `health_engine.py`       | Modules 9–11 — Health Score, Decision Engine, Recommendation Engine |
| `main.py`                | Ties it all together (Module 12/workflow steps 1–11)       |

`processed_hoist_data.csv` is written after each run as the "database"
stand-in (Module 1's storage requirement).

## Model performance (on held-out test runs)

- Fault classification accuracy: ~97%
- RUL prediction MAE: roughly 10–50 hours depending on component (brake
  system, with its shorter lifespan, predicts tightest; hoist rope, with
  the longest and noisiest degradation curve, is loosest)

These numbers come from synthetic data, so treat them as a proof of concept
for the pipeline architecture rather than real-world accuracy. Once actual
sensor readings and maintenance records from Old Nic Mine are available,
retrain each model on that real data — the code doesn't need to change,
just the data source.

## Running the web dashboard

`dashboard.py` wraps the same pipeline in a browser-based UI using
[Streamlit](https://streamlit.io), a Python library that turns a script
into a local web app -- no HTML/JavaScript needed.

1. Install everything (only needs doing once):
   ```bash
   pip install -r requirements.txt
   ```
2. From inside the project folder, run:
   ```bash
   streamlit run dashboard.py
   ```
3. It will automatically open `http://localhost:8501` in your default
   browser (if it doesn't, copy that URL into your browser manually).
4. First load takes ~20-30 seconds while it generates data and trains the
   models; after that it's cached and instant.
5. Use the sidebar to switch between hoist machines, and the dropdown at
   the bottom to inspect a specific component's degradation trend.
6. To stop it, go back to the terminal and press `Ctrl+C`.

## What "Module 12: Continuous Learning" would need next

Not yet implemented (needs a live system to log against), but the pipeline
is structured to support it: once maintenance technicians start recording
actual faults found vs. predicted, append those records with true labels
to the training set and re-run `train_models()` periodically.

## Notes on swapping in real sensor data

Replace `data_generator.generate_dataset()` with a function that reads from
your actual database/PLC feed, keeping the same column names used
throughout (`motor_temp_C`, `vibration_gearbox_mms`, `rope_tension_kN`,
etc. — see `data_pipeline.VALID_RANGES` for the full sensor list from your
spec). Everything downstream (validation, features, models, dashboard)
will work unchanged.
