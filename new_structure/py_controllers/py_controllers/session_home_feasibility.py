"""Pure read-only session-home joint feasibility metrics and decisions."""

import numpy as np


DEFAULT_THRESHOLDS = {
    "max_abs_warn_rad": 0.10,
    "max_abs_refuse_rad": 0.30,
    "l2_warn_rad": 0.20,
    "l2_refuse_rad": 0.50,
    "dq_warn_rad_s": 0.02,
    "dq_refuse_rad_s": 0.05,
}


def finite_vec(values, length):
    """Return a finite vector of the requested length, otherwise None."""
    if values is None:
        return None
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        return None
    return vector.copy()


def compute_joint_home_metrics(q_current, dq_current, q_home):
    """Compute read-only q/dq metrics relative to q_at_capture."""
    q_now = finite_vec(q_current, 7)
    dq_now = finite_vec(dq_current, 7)
    q_capture = finite_vec(q_home, 7)
    metrics = {
        "has_q_home": q_capture is not None,
        "has_q_current": q_now is not None,
        "has_dq_current": dq_now is not None,
        "max_abs_joint_delta_rad": None,
        "joint_l2_delta_rad": None,
        "max_abs_current_dq_rad_s": None,
        "per_joint_delta_rad": None,
        "dq_abs_rad_s": None,
    }
    if q_now is not None and q_capture is not None:
        delta = q_now - q_capture
        metrics.update({
            "max_abs_joint_delta_rad": float(np.max(np.abs(delta))),
            "joint_l2_delta_rad": float(np.linalg.norm(delta)),
            "per_joint_delta_rad": delta,
        })
    if dq_now is not None:
        dq_abs = np.abs(dq_now)
        metrics.update({
            "max_abs_current_dq_rad_s": float(np.max(dq_abs)),
            "dq_abs_rad_s": dq_abs,
        })
    return metrics


def validate_joint_home_thresholds(thresholds=None):
    """Return validated thresholds, raising ValueError for unsafe ordering."""
    values = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        values.update(thresholds)
    for name, value in values.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a finite nonnegative number")
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite nonnegative number")
        values[name] = value
    pairs = (
        ("max_abs_warn_rad", "max_abs_refuse_rad"),
        ("l2_warn_rad", "l2_refuse_rad"),
        ("dq_warn_rad_s", "dq_refuse_rad_s"),
    )
    for warn_name, refuse_name in pairs:
        if values[refuse_name] < values[warn_name]:
            raise ValueError(f"{refuse_name} must be >= {warn_name}")
    return values


def classify_joint_home(
    metrics, thresholds=None, enabled=True, require_q_home=False,
):
    """Classify metrics as ALLOW, WARN_ONLY, REFUSE, or missing-data label."""
    thresholds = validate_joint_home_thresholds(thresholds)
    if not enabled:
        return {"decision": "NOT_ENABLED", "allowed": True, "reasons": []}
    if not metrics.get("has_q_home"):
        return {
            "decision": "NO_Q_AT_CAPTURE",
            "allowed": not require_q_home,
            "reasons": ["session home has no finite q_at_capture"],
        }
    if not metrics.get("has_q_current"):
        return {
            "decision": "NO_CURRENT_Q",
            "allowed": False,
            "reasons": ["current q is missing or invalid"],
        }
    if not metrics.get("has_dq_current"):
        return {
            "decision": "NO_CURRENT_DQ",
            "allowed": False,
            "reasons": ["current dq is missing or invalid"],
        }
    checks = (
        ("max_abs_joint_delta_rad", "max_abs_warn_rad", "max_abs_refuse_rad"),
        ("joint_l2_delta_rad", "l2_warn_rad", "l2_refuse_rad"),
        ("max_abs_current_dq_rad_s", "dq_warn_rad_s", "dq_refuse_rad_s"),
    )
    refuse = []
    warn = []
    for metric_name, warn_name, refuse_name in checks:
        value = float(metrics[metric_name])
        if value > thresholds[refuse_name]:
            refuse.append(
                f"{metric_name}={value:.6f} > {refuse_name}="
                f"{thresholds[refuse_name]:.6f}"
            )
        elif value > thresholds[warn_name]:
            warn.append(
                f"{metric_name}={value:.6f} > {warn_name}="
                f"{thresholds[warn_name]:.6f}"
            )
    if refuse:
        return {"decision": "REFUSE", "allowed": False, "reasons": refuse}
    if warn:
        return {"decision": "WARN_ONLY", "allowed": True, "reasons": warn}
    return {"decision": "ALLOW", "allowed": True, "reasons": []}


def format_joint_home_report(metrics, classification):
    """Format one stable diagnostic line for CLI/controller logs."""
    def display(value):
        return "NA" if value is None else f"{float(value):.6f}"

    reasons = classification.get("reasons") or []
    return (
        f"decision={classification['decision']} "
        f"allowed={int(bool(classification['allowed']))} "
        "max_abs_joint_delta_rad="
        f"{display(metrics.get('max_abs_joint_delta_rad'))} "
        f"joint_l2_delta_rad={display(metrics.get('joint_l2_delta_rad'))} "
        "max_abs_current_dq_rad_s="
        f"{display(metrics.get('max_abs_current_dq_rad_s'))} "
        f"reasons={' | '.join(reasons) if reasons else 'none'}"
    )


def to_jsonable_joint_home_result(metrics, classification):
    """Convert numpy values to a JSON-safe report dictionary."""
    result = dict(metrics)
    for key in ("per_joint_delta_rad", "dq_abs_rad_s"):
        value = result.get(key)
        result[key] = None if value is None else np.asarray(value).tolist()
    result.update(classification)
    return result
