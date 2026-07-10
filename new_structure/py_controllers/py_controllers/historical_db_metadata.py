"""Historical residual DB sidecar creation and runtime identity validation."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


METADATA_SUFFIX = "_metadata.json"


def sha256_file(path):
    """Return the SHA-256 hex digest of a file without modifying it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_metadata_path(db_path):
    """Return <db stem>_metadata.json beside the DB."""
    db_path = Path(db_path)
    return db_path.with_name(f"{db_path.stem}{METADATA_SUFFIX}")


def load_metadata_sidecar(db_path, metadata_path=""):
    """Load optional metadata and return (metadata, resolved path)."""
    resolved = (
        Path(metadata_path).expanduser()
        if metadata_path
        else default_metadata_path(Path(db_path).expanduser())
    )
    if not resolved.is_file():
        return {}, resolved
    try:
        metadata = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"failed to parse metadata sidecar '{resolved}': {exc}"
        )
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata sidecar '{resolved}' is not a JSON object")
    return metadata, resolved


def _read_session_identity(session_home_path):
    if not session_home_path:
        return {}
    path = Path(session_home_path).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"session home '{path}' is not a JSON object")
    identity = {
        "session_home_path": str(path),
        "session_home_sha256": sha256_file(path),
    }
    for key in ("q_at_capture", "ee_pose_xyz"):
        if payload.get(key) is not None:
            identity[key] = payload[key]
    return identity


def create_historical_db_metadata(
    db_path, source_csvs, feature_schema, target_schema,
    session_home_path="", trajectory_id="", frequency_hz=None,
    q_scale=None, dq_scale=None, notes=None,
):
    """Create a complete metadata dictionary for an already-written DB."""
    db_path = Path(db_path).expanduser()
    db_sha256 = sha256_file(db_path)
    metadata = {
        "db_id": f"sha256:{db_sha256}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "database_type": "goal1_historical_residual_db",
        "db_sha256": db_sha256,
        "feature_schema": list(feature_schema),
        "target_schema": list(target_schema),
        "feature_dim": len(feature_schema),
        "target_dim": len(target_schema),
        "source_csvs": [str(Path(path).expanduser()) for path in source_csvs],
        "source_csv_count": len(source_csvs),
        "q_scale": None if q_scale is None else float(q_scale),
        "dq_scale": None if dq_scale is None else float(dq_scale),
        "trajectory_id": str(trajectory_id),
        "frequency_hz": (
            None if frequency_hz is None else float(frequency_hz)
        ),
        "notes": list(notes or []),
    }
    metadata.update(_read_session_identity(session_home_path))
    return metadata


def _finite_optional_vector(metadata, key, length, errors):
    value = metadata.get(key)
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        errors.append(f"{key} is not numeric")
        return None
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        errors.append(f"{key} must contain {length} finite values")
        return None
    return vector


def _list_field(metadata, key, errors):
    value = metadata.get(key)
    if not isinstance(value, (list, tuple)):
        errors.append(f"{key} must be a list")
        return []
    return list(value)


def _validate_db_identity_and_schema(
    metadata, db_path, expected_feature_schema, q_scale, dq_scale,
    require_metadata, errors,
):
    required = (
        "db_id", "created_at", "feature_schema", "target_schema",
        "feature_dim", "target_dim", "source_csvs", "source_csv_count",
        "db_sha256",
    )
    for key in required:
        if key not in metadata:
            errors.append(f"metadata missing required field '{key}'")

    db_path = Path(db_path).expanduser()
    if metadata.get("db_sha256"):
        actual_db_sha = sha256_file(db_path)
        if metadata["db_sha256"] != actual_db_sha:
            errors.append("db_sha256 does not match the loaded DB")
        if metadata.get("db_id") != f"sha256:{actual_db_sha}":
            errors.append("db_id does not match db_sha256")
    feature_schema = _list_field(metadata, "feature_schema", errors)
    target_schema = _list_field(metadata, "target_schema", errors)
    if metadata.get("feature_dim") != len(feature_schema):
        errors.append("feature_dim does not match feature_schema")
    if metadata.get("target_dim") != len(target_schema):
        errors.append("target_dim does not match target_schema")
    source_csvs = _list_field(metadata, "source_csvs", errors)
    if metadata.get("source_csv_count") != len(source_csvs):
        errors.append("source_csv_count does not match source_csvs")
    if expected_feature_schema is not None:
        if feature_schema != list(expected_feature_schema):
            errors.append(
                "feature_schema does not match runtime [q,dq] schema"
            )
    for key, expected in (("q_scale", q_scale), ("dq_scale", dq_scale)):
        recorded = metadata.get(key)
        if recorded is None and expected is not None and require_metadata:
            errors.append(f"metadata missing required field '{key}'")
            continue
        if recorded is not None and expected is not None:
            try:
                scale_matches = np.isclose(
                    float(recorded), float(expected), rtol=0.0, atol=1e-12
                )
            except (TypeError, ValueError):
                scale_matches = False
            if not scale_matches:
                errors.append(f"{key} does not match runtime scale")
    return _finite_optional_vector(metadata, "q_at_capture", 7, errors)


def _validate_session_binding(
    metadata, session_home_path, require_session_binding,
    q_metadata, errors,
):
    if require_session_binding and not metadata.get("session_home_sha256"):
        errors.append("metadata has no session_home_sha256 binding")
    if require_session_binding and not session_home_path:
        errors.append("runtime session_home_path is required for binding")
    if session_home_path and metadata.get("session_home_sha256"):
        session_path = Path(session_home_path).expanduser()
        if not session_path.is_file():
            errors.append("runtime session home file does not exist")
        elif metadata["session_home_sha256"] != sha256_file(session_path):
            errors.append(
                "session_home_sha256 does not match runtime session home"
            )
        else:
            try:
                payload = json.loads(session_path.read_text(encoding="utf-8"))
                q_runtime = _finite_optional_vector(
                    payload, "q_at_capture", 7, errors
                )
                if q_metadata is not None and q_runtime is not None:
                    if not np.array_equal(q_metadata, q_runtime):
                        errors.append(
                            "q_at_capture differs from runtime session home"
                        )
            except Exception as exc:
                errors.append(
                    f"failed to validate runtime session home: {exc}"
                )


def validate_historical_db_metadata(
    metadata, db_path, session_home_path="", expected_feature_schema=None,
    q_scale=None, dq_scale=None, require_metadata=False,
    require_session_binding=False,
):
    """Validate DB/schema/scale/session identity and return results."""
    errors = []
    warnings = []
    if not metadata:
        message = "historical DB metadata is missing"
        (errors if require_metadata else warnings).append(message)
        return {"valid": not errors, "errors": errors, "warnings": warnings}
    q_metadata = _validate_db_identity_and_schema(
        metadata, db_path, expected_feature_schema, q_scale, dq_scale,
        require_metadata, errors,
    )
    _validate_session_binding(
        metadata, session_home_path, require_session_binding,
        q_metadata, errors,
    )
    return {"valid": not errors, "errors": errors, "warnings": warnings}
