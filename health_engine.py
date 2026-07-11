"""
Module 9: Health Score Calculation
Module 10: Decision Engine
Module 11: Recommendation Engine
====================================
Turns raw model outputs (anomaly score, fault probability, RUL) into the
operator-facing interpretation described in the spec, e.g.:

    Brake System
    Health Score: 42%
    Fault Probability: 94%
    Estimated Failure: Within 48 operating hours
    Recommendation: Replace brake pads and inspect brake actuators immediately.
"""

from dataclasses import dataclass

# Configurable thresholds (Module 10 -- "thresholds should be configurable
# rather than hard-coded")
HEALTH_THRESHOLDS = {
    "normal": 80,
    "monitor": 60,
    "schedule_maintenance": 40,
    # below 40 -> immediate inspection
}

RECOMMENDATIONS = {
    "Motor": {
        "Critical": "Inspect motor windings and cooling system immediately; prepare for motor replacement.",
        "High": "Schedule motor winding insulation test and bearing inspection.",
        "Medium": "Monitor motor temperature and current trends closely.",
        "Low": "Continue routine motor monitoring.",
    },
    "Gearbox": {
        "Critical": "Stop hoist and inspect gearbox bearings/gears immediately; check for metal debris in oil.",
        "High": "Schedule gearbox inspection and oil analysis within the week.",
        "Medium": "Lubricate gearbox and monitor oil pressure trend.",
        "Low": "Continue routine gearbox monitoring.",
    },
    "Brake System": {
        "Critical": "Replace brake pads and inspect brake actuators immediately.",
        "High": "Schedule brake pad replacement and actuator inspection soon.",
        "Medium": "Monitor brake temperature; inspect wear indicators at next stop.",
        "Low": "Continue routine brake monitoring.",
    },
    "Bearings": {
        "Critical": "Replace affected bearing immediately; halt operation if vibration exceeds safety limits.",
        "High": "Schedule bearing replacement and vibration analysis.",
        "Medium": "Monitor bearing vibration and temperature trend.",
        "Low": "Continue routine bearing monitoring.",
    },
    "Hoist Rope": {
        "Critical": "Stop hoist and inspect rope for broken wires / tension imbalance immediately.",
        "High": "Schedule detailed rope inspection (broken wires, corrosion, diameter loss).",
        "Medium": "Monitor rope tension balance across the drum.",
        "Low": "Continue routine rope monitoring.",
    },
}


@dataclass
class ComponentAssessment:
    component: str
    health_score: float
    fault_probability: float
    estimated_rul_hours: float
    status: str
    priority: str
    recommendation: str


def calculate_health_score(anomaly_score: float, fault_probability: float, rul_hours: float,
                            max_expected_life_hours: float) -> float:
    """
    Module 9: Combines the anomaly score, the classifier's fault
    probability, and the RUL estimate (relative to the component's typical
    lifespan) into a single 0-100% health score.
    """
    rul_fraction = max(0.0, min(1.0, rul_hours / max_expected_life_hours))
    # Weighted blend: RUL fraction carries the most weight, fault probability
    # and anomaly score pull the score down further if elevated.
    raw = (
        0.55 * rul_fraction * 100
        + 0.30 * (1 - fault_probability) * 100
        + 0.15 * (1 - anomaly_score) * 100
    )
    return round(max(0.0, min(100.0, raw)), 1)


def decision_status(health_score: float) -> str:
    """Module 10: Decision Engine."""
    if health_score > HEALTH_THRESHOLDS["normal"]:
        return "Normal operation"
    elif health_score > HEALTH_THRESHOLDS["monitor"]:
        return "Monitor closely"
    elif health_score > HEALTH_THRESHOLDS["schedule_maintenance"]:
        return "Schedule maintenance"
    else:
        return "Immediate inspection"


def priority_level(health_score: float, fault_probability: float) -> str:
    if health_score < 40 or fault_probability > 0.85:
        return "Critical"
    elif health_score < 60 or fault_probability > 0.6:
        return "High"
    elif health_score < 80 or fault_probability > 0.35:
        return "Medium"
    return "Low"


def assess_component(component: str, anomaly_score: float, fault_probability: float,
                      rul_hours: float, max_expected_life_hours: float) -> ComponentAssessment:
    """Module 11: Recommendation Engine, tied together with 9 & 10."""
    health = calculate_health_score(anomaly_score, fault_probability, rul_hours, max_expected_life_hours)
    status = decision_status(health)
    priority = priority_level(health, fault_probability)
    recommendation = RECOMMENDATIONS.get(component, {}).get(priority, "Continue routine monitoring.")

    return ComponentAssessment(
        component=component,
        health_score=health,
        fault_probability=round(fault_probability * 100, 1),
        estimated_rul_hours=round(rul_hours, 1),
        status=status,
        priority=priority,
        recommendation=recommendation,
    )


def format_dashboard_card(assessment: ComponentAssessment) -> str:
    hours = assessment.estimated_rul_hours
    if hours < 72:
        life_str = f"Within {int(hours)} operating hours"
    elif hours < 24 * 30:
        life_str = f"~{hours / 24:.0f} days remaining"
    else:
        life_str = f"~{hours / (24 * 30):.1f} months remaining"

    return (
        f"{assessment.component}\n"
        f"  Health Score: {assessment.health_score}%\n"
        f"  Fault Probability: {assessment.fault_probability}%\n"
        f"  Estimated Failure / RUL: {life_str}\n"
        f"  Status: {assessment.status}  (Priority: {assessment.priority})\n"
        f"  Recommendation: {assessment.recommendation}"
    )
