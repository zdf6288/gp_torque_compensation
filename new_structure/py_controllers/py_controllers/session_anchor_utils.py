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

import numpy as np


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
