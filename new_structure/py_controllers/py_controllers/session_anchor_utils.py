"""Session-relative anchor / vec3 参数解析纯函数集合。

为什么抽出：
    cartesian_impedance.py（约 8000 行）与 trajectory_publisher.py 里各自
    重复了一份完全相同的 3-vector 参数解析逻辑，以及一份仅错误前缀不同的
    anchor JSON 字段读取逻辑。session_relative anchor / trajectory 是后续要
    持续维护的区域，重复实现容易在两个文件里被改歪（例如最近一次
    STRING vs DOUBLE_ARRAY 的类型修复只改了一处）。把这两个纯函数收敛到一个
    没有 ROS 依赖的小模块，方便统一维护和单独 import 测试。

边界（本模块只做纯数据校验，不碰任何控制/实时行为）：
    * 只依赖 numpy 和 json，不 import rclpy，不访问节点 self 状态；
    * 只负责“解析 + 校验恰好 3 个有限数”，不做几何平移、不做 anchor_delta
      计算、不决定是否发布力矩；
    * 兼容两种参数来源：JSON 字符串（STRING）与 list/DOUBLE_ARRAY；
    * 错误文案保持与原实现逐字节一致，错误语义（抛 ValueError）不变，
      由调用方决定失败后如何 fail-closed。
"""

import json
from pathlib import Path

import numpy as np


def load_session_home_payload(file_path):
    """Load a session-home JSON object with the controller's legacy errors."""
    file_path = Path(file_path)
    try:
        payload = json.loads(file_path.read_text())
    except Exception as e:
        raise ValueError(
            f"[SessionHome] failed to parse session home JSON "
            f"'{file_path}': {e}"
        )
    if not isinstance(payload, dict):
        raise ValueError(
            f"[SessionHome] session home JSON '{file_path}' is not an object."
        )
    return payload


def read_optional_q_at_capture(payload, error_prefix):
    """Return optional finite 7D q_at_capture, preserving schema errors."""
    q_at_capture = payload.get('q_at_capture')
    if q_at_capture is None:
        return None
    try:
        q_vec = np.asarray(q_at_capture, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"{error_prefix}q_at_capture must be null or 7 finite values."
        )
    if q_vec.shape != (7,) or not np.all(np.isfinite(q_vec)):
        raise ValueError(
            f"{error_prefix}q_at_capture must be null or 7 finite values, "
            f"got {q_at_capture!r}."
        )
    return q_vec


def parse_vec3_parameter(value, name):
    """把一个参数值解析为恰好 3 个有限数的 numpy 向量，失败抛 ValueError。

    兼容 STRING（JSON 风格字符串）与 list/DOUBLE_ARRAY 两种来源；错误文案与
    原 cartesian_impedance._get_vec3_parameter / trajectory_publisher.
    _parse_vec3_parameter 保持一致。
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Parameter '{name}' must be a JSON-style 3-vector, "
                f"got {value!r}: {e}"
            )
    try:
        vec = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"Parameter '{name}' must be 3 finite numeric values, "
            f"got {value!r}."
        )
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        raise ValueError(
            f"Parameter '{name}' must be 3 finite numeric values, "
            f"got {value!r}."
        )
    return vec


def read_anchor_vec3(payload, key, error_prefix):
    """从 anchor JSON 里读取一个恰好 3 个有限数的字段，失败抛 ValueError。

    error_prefix 由调用方给出，用来保持各文件原有的错误前缀逐字节一致
    （cartesian 用 ``[SessionAnchor] '<file>': ``，trajectory 用
    ``session anchor ``），因此不改变任何错误语义。
    """
    value = payload.get(key)
    try:
        vec = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{error_prefix}field '{key}' is not numeric.")
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        raise ValueError(
            f"{error_prefix}field '{key}' must be 3 finite values, "
            f"got {value!r}."
        )
    return vec


def compute_anchor_internal_residuals(
    ee_pose, session_start, nominal_start, nominal_center,
    shifted_center, anchor_delta,
):
    """计算 anchor JSON 内部自洽性的三个残差（设计 B 不变量），只算不判阈值。

    正常应全部 ~0：
        ee_pose_residual = |ee_pose - session_start|
        start_residual   = |session_start - (nominal_start + anchor_delta)|
        center_residual  = |shifted_center - (nominal_center + anchor_delta)|
    这三条编码设计 B：current pose == session_trajectory_start，start 与 center
    都按同一个 anchor_delta 从名义几何整体平移。这里只做纯 numpy 计算、不设阈值、
    不抛错；阈值与错误文案由各调用方保留（cartesian / trajectory 文案不同）。
    ``nominal_start`` / ``nominal_center`` 传入的是 anchor JSON 里记录的名义几何。
    """
    ee_pose_residual = float(np.linalg.norm(ee_pose - session_start))
    start_residual = float(np.linalg.norm(
        session_start - (nominal_start + anchor_delta)
    ))
    center_residual = float(np.linalg.norm(
        shifted_center - (nominal_center + anchor_delta)
    ))
    return ee_pose_residual, start_residual, center_residual


def validate_session_anchor_payload(
    payload, file_path, nominal_start, nominal_center,
):
    """校验 session-relative anchor JSON 的纯逻辑，返回 session_trajectory_start。

    从 cartesian_impedance._validate_session_anchor_payload 抽出的纯部分：模式/
    版本/必填字段检查、7 个 vec3 字段读取、q_at_capture 校验、内部自洽性残差、
    以及“采集时名义几何 vs 当前 run 名义几何”一致性检查。任何失败抛 ValueError，
    错误文案与原实现逐字节一致。

    刻意不包含依赖 node 配置/日志的 z 范围 + anchor_delta 上限/警告安全门
    （原 _validate_session_relative_start）——那部分仍由 node 侧在本函数返回后
    调用，避免把 realtime/日志行为搬进纯模块。

    ``nominal_start`` / ``nominal_center`` 是当前 run 配置的名义几何
    （node.session_relative_nominal_trajectory_start / _circle_center）。
    """
    file_mode = str(
        payload.get('trajectory_reference_mode', '')
    ).strip().lower()
    if file_mode != 'session_relative':
        raise ValueError(
            f"[SessionAnchor] '{file_path}': trajectory_reference_mode="
            f"'{file_mode}' does not match controller mode "
            "'session_relative'. Re-capture the anchor or switch modes."
        )
    for key in ('version', 'created_at', 'source', 'notes'):
        if key not in payload:
            raise ValueError(
                f"[SessionAnchor] '{file_path}': missing required field "
                f"'{key}'."
            )
    try:
        version = int(payload.get('version'))
    except (TypeError, ValueError):
        raise ValueError(
            f"[SessionAnchor] '{file_path}': field 'version' must be an "
            "integer."
        )
    if version < 2:
        raise ValueError(
            f"[SessionAnchor] '{file_path}': version={version} is too old "
            "for session_relative anchors; re-capture the anchor."
        )

    error_prefix = f"[SessionAnchor] '{file_path}': "
    session_start = read_anchor_vec3(
        payload, 'session_trajectory_start_xyz', error_prefix
    )
    ee_pose = read_anchor_vec3(payload, 'ee_pose_xyz', error_prefix)
    nominal_start_json = read_anchor_vec3(
        payload, 'nominal_trajectory_start_xyz', error_prefix
    )
    nominal_center_json = read_anchor_vec3(
        payload, 'nominal_circle_center_xyz', error_prefix
    )
    shifted_center = read_anchor_vec3(
        payload, 'shifted_circle_center_xyz', error_prefix
    )
    anchor_delta = read_anchor_vec3(
        payload, 'anchor_delta_xyz', error_prefix
    )
    read_anchor_vec3(payload, 'nominal_fixed_start_xyz', error_prefix)

    read_optional_q_at_capture(payload, error_prefix)

    # 文件内部一致性：session_start/shifted_center 必须与 anchor_delta 自洽。
    internal_tol = 1e-6
    ee_pose_residual, start_residual, center_residual = (
        compute_anchor_internal_residuals(
            ee_pose, session_start, nominal_start_json, nominal_center_json,
            shifted_center, anchor_delta,
        )
    )
    if (
        ee_pose_residual > internal_tol
        or start_residual > internal_tol
        or center_residual > internal_tol
    ):
        raise ValueError(
            f"[SessionAnchor] '{file_path}': internally inconsistent "
            f"anchor JSON (ee_pose_residual={ee_pose_residual:.2e} m, "
            f"start_residual={start_residual:.2e} m, "
            f"center_residual={center_residual:.2e} m)."
        )

    # 采集时的名义几何必须与当前 launch 的名义几何一致，否则平移后的
    # 轨迹会不同，跨 run 不可比。
    geometry_tol = 0.005
    nominal_start_mismatch = float(np.linalg.norm(
        nominal_start_json - nominal_start
    ))
    nominal_center_mismatch = float(np.linalg.norm(
        nominal_center_json - nominal_center
    ))
    if (
        nominal_start_mismatch > geometry_tol
        or nominal_center_mismatch > geometry_tol
    ):
        raise ValueError(
            f"[SessionAnchor] '{file_path}': nominal geometry in the "
            "anchor JSON does not match this run "
            f"(start mismatch {nominal_start_mismatch:.4f} m, center "
            f"mismatch {nominal_center_mismatch:.4f} m > "
            f"{geometry_tol:.3f} m). Re-capture the anchor for the "
            "current geometry."
        )

    return session_start
