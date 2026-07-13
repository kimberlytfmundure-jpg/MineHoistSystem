"""
Module 5: Learning Normal Behaviour
Module 6: Fault Detection (Anomaly Detection)
Module 7: Fault Classification
Module 8: Remaining Useful Life (RUL) Prediction
====================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, mean_absolute_error, accuracy_score,
)


class AnomalyDetector:
    """Module 5 & 6: learns 'normal' behaviour from early-life (healthy)
    data and flags abnormal operating conditions."""

    def __init__(self, contamination=0.05, random_state=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=random_state
        )

    def fit(self, X_healthy: pd.DataFrame):
        Xs = self.scaler.fit_transform(X_healthy)
        self.model.fit(Xs)
        return self

    def predict(self, X: pd.DataFrame):
        """Returns (is_anomaly: bool array, anomaly_score: float array in [0,1] where higher = more abnormal)"""
        Xs = self.scaler.transform(X)
        raw_scores = self.model.score_samples(Xs)  # higher = more normal
        # Normalize to a 0-1 "fault probability"-like scale
        normalized = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-9)
        is_anomaly = self.model.predict(Xs) == -1
        return is_anomaly, normalized


class FaultClassifier:
    """Module 7: given an abnormal reading, determines which component is
    most likely developing a fault."""

    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=random_state, class_weight="balanced"
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, zero_division=0)
        return acc, report


class RULPredictor:
    """Module 8: estimates remaining useful life (in operating hours) for
    the affected component. One model per component tends to perform
    better since degradation dynamics differ; here we train one model
    per failing_component label."""

    def __init__(self, random_state=42):
        self.models = {}
        self.random_state = random_state

    def fit(self, df: pd.DataFrame, feature_cols, component_col="failing_component",
            target_col="true_rul_hours"):
        for component, group in df.groupby(component_col):
            model = GradientBoostingRegressor(random_state=self.random_state, max_depth=3,
                                               n_estimators=250, learning_rate=0.05)
            model.fit(group[feature_cols], group[target_col])
            self.models[component] = model
        return self

    def predict(self, component: str, X: pd.DataFrame):
        if component not in self.models:
            raise ValueError(f"No RUL model trained for component '{component}'")
        return self.models[component].predict(X)

    def evaluate(self, df: pd.DataFrame, feature_cols, component_col="failing_component",
                 target_col="true_rul_hours"):
        results = {}
        for component, group in df.groupby(component_col):
            preds = self.predict(component, group[feature_cols])
            mae = mean_absolute_error(group[target_col], preds)
            results[component] = mae
        return results


def train_test_split_by_run(df: pd.DataFrame, test_size=0.25, random_state=42):
    """Splits by run_id (not by row) so entire degradation sequences stay
    together in either train or test -- avoids leakage between the two."""
    run_ids = df["run_id"].unique()
    train_ids, test_ids = train_test_split(run_ids, test_size=test_size, random_state=random_state)
    return df[df["run_id"].isin(train_ids)].copy(), df[df["run_id"].isin(test_ids)].copy()


def compare_classifiers(X_train, y_train, X_test, y_test, random_state=42):
    """
    Proposal Stage 4: 'Machine learning algorithms such as Random Forest,
    Decision Tree, and Gradient Boosting will be evaluated to determine
    which model provides the highest prediction accuracy.'

    Trains all three on the fault-classification task and returns a
    dict of {algorithm_name: {"accuracy": float, "report": str}}, plus the
    name of whichever performed best.
    """
    candidates = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, random_state=random_state, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=random_state, class_weight="balanced"
        ),
        # HistGradientBoostingClassifier: scikit-learn's fast, histogram-based
        # Gradient Boosting implementation -- same algorithm family named in
        # the spec, chosen over the classic GradientBoostingClassifier here
        # purely for speed on multiclass data of this size.
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.1, max_iter=150, random_state=random_state
        ),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, zero_division=0)
        results[name] = {"accuracy": acc, "report": report, "model": model}

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    return results, best_name
