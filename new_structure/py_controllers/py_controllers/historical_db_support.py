"""Pure historical residual DB feature and nearest-support calculations."""

import numpy as np


FEATURE_DIM = 14
TARGET_DIM = 7
DEFAULT_FEATURE_NAMES = tuple(
    [f"joint_pos_{index}" for index in range(1, 8)]
    + [f"joint_vel_{index}" for index in range(1, 8)]
)


def build_joint_feature(q, dq):
    """Build finite [q1..q7,dq1..dq7], or return None for invalid input."""
    try:
        q_arr = np.asarray(q, dtype=float)
        dq_arr = np.asarray(dq, dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        q_arr.shape != (7,)
        or dq_arr.shape != (7,)
        or not np.all(np.isfinite(q_arr))
        or not np.all(np.isfinite(dq_arr))
    ):
        return None
    return np.concatenate([q_arr, dq_arr])


def scale_feature(feature, feature_scale):
    """Scale one finite 14D feature, or return None for invalid input."""
    try:
        feature_arr = np.asarray(feature, dtype=float)
        scale_arr = np.asarray(feature_scale, dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        feature_arr.shape != (FEATURE_DIM,)
        or scale_arr.shape != (FEATURE_DIM,)
        or not np.all(np.isfinite(feature_arr))
        or not np.all(np.isfinite(scale_arr))
        or np.any(scale_arr <= 0.0)
    ):
        return None
    scaled = np.ascontiguousarray(feature_arr / scale_arr, dtype=float)
    return scaled if np.all(np.isfinite(scaled)) else None


def scale_feature_matrix(features, feature_scale):
    """Scale an (N,14) feature matrix, raising ValueError on invalid data."""
    features_arr = np.asarray(features, dtype=float)
    scale_arr = np.asarray(feature_scale, dtype=float)
    if features_arr.ndim != 2 or features_arr.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"features must have shape (N, 14), got {features_arr.shape}"
        )
    if (
        scale_arr.shape != (FEATURE_DIM,)
        or not np.all(np.isfinite(scale_arr))
        or np.any(scale_arr <= 0.0)
    ):
        raise ValueError(
            "feature_scale must contain 14 finite positive values"
        )
    scaled = np.ascontiguousarray(
        features_arr / scale_arr.reshape(1, -1), dtype=float
    )
    if not np.all(np.isfinite(scaled)):
        raise ValueError("scaled X contains non-finite values")
    return scaled


def query_scaled_nearest_support(
    x_db_scaled, y_db, x_query_scaled, k, max_distance,
):
    """Run the legacy scaled KNN query and return support diagnostics."""
    result = {
        "valid": 0,
        "k_used": 0,
        "nearest_index": -1,
        "nearest_distance": 0.0,
        "mean_topk_distance": 0.0,
        "distance_pass": 0,
        "prediction": np.zeros(TARGET_DIM, dtype=float),
    }
    try:
        x_arr = np.asarray(x_db_scaled, dtype=float)
        y_arr = np.asarray(y_db, dtype=float)
        query_arr = np.asarray(x_query_scaled, dtype=float)
        row_count = int(x_arr.shape[0])
        if (
            x_arr.shape != (row_count, FEATURE_DIM)
            or y_arr.shape != (row_count, TARGET_DIM)
            or query_arr.shape != (FEATURE_DIM,)
            or row_count <= 0
        ):
            return result
        with np.errstate(over="ignore", invalid="ignore"):
            delta = x_arr - query_arr.reshape(1, -1)
            distance_sq = np.einsum("ij,ij->i", delta, delta)
        if (
            distance_sq.shape != (row_count,)
            or not np.all(np.isfinite(distance_sq))
            or np.any(distance_sq < 0.0)
        ):
            return result
        k_used = min(int(k), row_count)
        nearest_indices = np.argpartition(
            distance_sq, kth=k_used - 1
        )[:k_used]
        nearest_indices = nearest_indices[
            np.argsort(distance_sq[nearest_indices])
        ]
        nearest_distances = np.sqrt(distance_sq[nearest_indices])
        prediction = np.mean(y_arr[nearest_indices], axis=0)
    except (TypeError, ValueError, IndexError, FloatingPointError):
        return result
    if (
        prediction.shape != (TARGET_DIM,)
        or not np.all(np.isfinite(prediction))
    ):
        return result
    nearest_distance = float(nearest_distances[0])
    result.update({
        "valid": 1,
        "k_used": int(k_used),
        "nearest_index": int(nearest_indices[0]),
        "nearest_distance": nearest_distance,
        "mean_topk_distance": float(np.mean(nearest_distances)),
        "distance_pass": int(nearest_distance <= max_distance),
        "prediction": prediction.copy(),
    })
    return result


def legacy_active_support_available(
    loaded, query_valid, prediction_valid, online_disabled,
):
    """Return the pre-M-HomeSupportGate historical availability decision."""
    return int(bool(
        loaded and query_valid and prediction_valid and not online_disabled
    ))


def select_legacy_gated_prediction(
    prediction, fallback_prediction, fallback_source_code,
    loaded, query_valid, prediction_valid, online_disabled,
):
    """Apply legacy availability/fallback selection without a distance gate."""
    available = legacy_active_support_available(
        loaded, query_valid, prediction_valid, online_disabled
    )
    if available:
        return available, np.asarray(prediction, dtype=float).copy(), 4
    return (
        available,
        np.asarray(fallback_prediction, dtype=float).copy(),
        int(fallback_source_code),
    )


def compute_scaled_delta_contributions(
    x_nearest, x_query, feature_scale, feature_names=None,
):
    """Return each dimension's squared contribution to scaled distance."""
    nearest = np.asarray(x_nearest, dtype=float)
    query = np.asarray(x_query, dtype=float)
    scale = np.asarray(feature_scale, dtype=float)
    if (
        nearest.shape != (FEATURE_DIM,)
        or query.shape != (FEATURE_DIM,)
        or scale.shape != (FEATURE_DIM,)
        or not np.all(np.isfinite(nearest))
        or not np.all(np.isfinite(query))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise ValueError(
            "nearest, query, and scale must be finite 14D vectors with "
            "positive scale"
        )
    names = tuple(feature_names or DEFAULT_FEATURE_NAMES)
    if len(names) != FEATURE_DIM:
        raise ValueError("feature_names must contain 14 names")
    scaled_delta = (query - nearest) / scale
    contribution = np.square(scaled_delta)
    return {
        "feature_names": names,
        "scaled_delta": scaled_delta,
        "contribution": contribution,
        "total_distance": float(np.sqrt(np.sum(contribution))),
    }
