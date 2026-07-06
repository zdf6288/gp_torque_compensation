"""cartesian_impedance 控制器的 session-relative 参数声明 / 读取集合。

为什么抽出：
    cartesian_impedance.py 的 __init__ 超过 1600 行，session-relative 这组参数
    的 declare + read 混在里面，后续维护 session anchor / trajectory 时很难
    定位。把这组“纯配置声明与读取”收敛到一个 focused module，让 __init__ 只
    保留一次函数调用，参数组的增删改集中在这里。

边界（本模块只做 session-relative 配置的声明与读取，不做任何控制/几何/实时行为）：
    * 只声明 / 读取 session_relative_* 这一组参数，不碰 control gains、safety、
      torque、GP、logging、timing 等其它参数组；
    * 不改参数名、不改默认值、不改 ParameterDescriptor(dynamic_typing=True)
      的 STRING/DOUBLE_ARRAY 兼容语义；
    * read_session_relative_config 刻意复用 node 自己的 _get_*_parameter 访问器
      （node._get_bool_parameter 等），保证解析/类型/错误语义与原 __init__ 内联
      写法逐字节一致，只是把语句从 __init__ 平移到这里；
    * trajectory_reference_mode 的读取与校验、session_anchor_delta 的初始化仍留
      在 __init__，因为它们同时门控更广的 session_home 流程，不属于本参数组。
"""

from rcl_interfaces.msg import ParameterDescriptor

# anchor_delta norm 超过 session_relative_max_anchor_delta_m 时的处理策略。
# refuse 保持旧的 hard-refuse 行为（node/launch 默认）；warn 只告警并继续
# （floating-anchor session，从当前摆好的 pose 整体平移轨迹）；off 完全不做
# 该 norm 检查。z 范围 / stable-state / 内部自洽等其余安全门不受此策略影响。
SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODES = ('refuse', 'warn', 'off')
DEFAULT_SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODE = 'refuse'


def normalize_anchor_delta_limit_mode(value):
    """把 session_relative_anchor_delta_limit_mode 归一化为小写并校验合法值。

    非法值直接抛 ValueError（fail fast，构造期抛出 → 节点拒绝启动），错误
    信息包含参数名、允许值和实际值。cartesian 与 trajectory_publisher 共用
    本函数，保证两个节点对该参数的解析/默认语义一致。
    """
    mode = str(value).strip().lower()
    if mode not in SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODES:
        raise ValueError(
            "session_relative_anchor_delta_limit_mode must be one of "
            f"{list(SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODES)}, "
            f"got '{value}'."
        )
    return mode


def declare_session_relative_parameters(node):
    """声明 session_relative_* 参数组（默认值与原 __init__ 内联声明完全一致）。"""
    node.declare_parameter('session_relative_capture_enabled', False)
    node.declare_parameter('session_relative_max_anchor_delta_m', 0.250)
    node.declare_parameter('session_relative_warn_anchor_delta_m', 0.150)
    node.declare_parameter(
        'session_relative_anchor_delta_limit_mode',
        DEFAULT_SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODE,
    )
    node.declare_parameter('session_relative_min_z', 0.45)
    node.declare_parameter('session_relative_max_z', 0.85)
    node.declare_parameter('session_relative_requires_stable_state', True)
    node.declare_parameter('session_relative_stability_samples', 10)
    node.declare_parameter(
        'session_relative_stability_position_std_m', 0.003
    )
    node.declare_parameter(
        'session_relative_apply_to_startup_and_return', True
    )
    # 名义轨迹几何参考点。默认值必须与 trajectory_publisher 默认
    # goal1_spatial_multisine 几何在 t=0 的起点严格一致：
    #   x0 = 0.3 + 0.012*sin(0.7)
    #   y0 = 0.035*cos(0.4) + 0.012*sin(1.3)
    #   z0 = 0.65 + 0.030*sin(0.2) + 0.010*sin(1.1)
    # trajectory_publisher 加载 anchor 后会用自身几何做起点一致性检查
    # （容差 0.010 m），两边不一致会 fail closed。
    # launch/runner 以 STRING 传入这两个 vec3（ParameterValue value_type=str），
    # 但默认值是 list[float] 会让 rclpy 推断为 DOUBLE_ARRAY，导致 STRING 覆盖时
    # 抛 InvalidParameterTypeException。用 dynamic_typing 同时接受 STRING 与
    # DOUBLE_ARRAY；实际解析统一走 _get_vec3_parameter（支持 JSON 字符串与
    # list[float]，并校验恰好 3 个有限数）。
    node.declare_parameter(
        'session_relative_nominal_trajectory_start_xyz',
        [0.3077306122468523, 0.043799833015107294, 0.6648721535244662],
        ParameterDescriptor(dynamic_typing=True),
    )
    node.declare_parameter(
        'session_relative_nominal_circle_center_xyz',
        [0.3, 0.0, 0.65],
        ParameterDescriptor(dynamic_typing=True),
    )


def read_session_relative_config(node):
    """把 session_relative_* 参数读入 node.session_relative_* 属性。

    刻意复用 node 自己的 _get_*_parameter 访问器，读取/类型/错误语义与原
    __init__ 内联写法逐字节一致（只是把语句平移到这里）。
    """
    node.session_relative_capture_enabled = node._get_bool_parameter(
        'session_relative_capture_enabled'
    )
    node.session_relative_max_anchor_delta_m = (
        node._get_positive_float_parameter(
            'session_relative_max_anchor_delta_m', 0.250
        )
    )
    node.session_relative_warn_anchor_delta_m = (
        node._get_nonnegative_float_parameter(
            'session_relative_warn_anchor_delta_m', 0.150
        )
    )
    node.session_relative_anchor_delta_limit_mode = (
        normalize_anchor_delta_limit_mode(
            node.get_parameter(
                'session_relative_anchor_delta_limit_mode'
            ).value
        )
    )
    node.session_relative_min_z = node._get_nonnegative_float_parameter(
        'session_relative_min_z', 0.45
    )
    node.session_relative_max_z = node._get_positive_float_parameter(
        'session_relative_max_z', 0.85
    )
    node.session_relative_requires_stable_state = node._get_bool_parameter(
        'session_relative_requires_stable_state'
    )
    node.session_relative_stability_samples = node._get_bounded_int_parameter(
        'session_relative_stability_samples', 10, 2, 1000
    )
    node.session_relative_stability_position_std_m = (
        node._get_positive_float_parameter(
            'session_relative_stability_position_std_m', 0.003
        )
    )
    node.session_relative_apply_to_startup_and_return = (
        node._get_bool_parameter(
            'session_relative_apply_to_startup_and_return'
        )
    )
    node.session_relative_nominal_trajectory_start = (
        node._get_vec3_parameter(
            'session_relative_nominal_trajectory_start_xyz'
        )
    )
    node.session_relative_nominal_circle_center = (
        node._get_vec3_parameter('session_relative_nominal_circle_center_xyz')
    )
