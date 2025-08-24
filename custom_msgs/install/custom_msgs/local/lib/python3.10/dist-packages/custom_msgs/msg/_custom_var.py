# generated from rosidl_generator_py/resource/_idl.py.em
# with input from custom_msgs:msg/CustomVar.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_CustomVar(type):
    """Metaclass of message 'CustomVar'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('custom_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'custom_msgs.msg.CustomVar')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__custom_var
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__custom_var
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__custom_var
            cls._TYPE_SUPPORT = module.type_support_msg__msg__custom_var
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__custom_var

            from std_msgs.msg import Float64MultiArray
            if Float64MultiArray.__class__._TYPE_SUPPORT is None:
                Float64MultiArray.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CustomVar(metaclass=Metaclass_CustomVar):
    """Message class 'CustomVar'."""

    __slots__ = [
        '_q_des',
        '_dq_des',
        '_q_meas',
        '_dq_meas',
        '_ddq_computed',
        '_error_endo',
        '_error_rcm',
    ]

    _fields_and_field_types = {
        'q_des': 'std_msgs/Float64MultiArray',
        'dq_des': 'std_msgs/Float64MultiArray',
        'q_meas': 'std_msgs/Float64MultiArray',
        'dq_meas': 'std_msgs/Float64MultiArray',
        'ddq_computed': 'std_msgs/Float64MultiArray',
        'error_endo': 'std_msgs/Float64MultiArray',
        'error_rcm': 'std_msgs/Float64MultiArray',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Float64MultiArray
        self.q_des = kwargs.get('q_des', Float64MultiArray())
        from std_msgs.msg import Float64MultiArray
        self.dq_des = kwargs.get('dq_des', Float64MultiArray())
        from std_msgs.msg import Float64MultiArray
        self.q_meas = kwargs.get('q_meas', Float64MultiArray())
        from std_msgs.msg import Float64MultiArray
        self.dq_meas = kwargs.get('dq_meas', Float64MultiArray())
        from std_msgs.msg import Float64MultiArray
        self.ddq_computed = kwargs.get('ddq_computed', Float64MultiArray())
        from std_msgs.msg import Float64MultiArray
        self.error_endo = kwargs.get('error_endo', Float64MultiArray())
        from std_msgs.msg import Float64MultiArray
        self.error_rcm = kwargs.get('error_rcm', Float64MultiArray())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.q_des != other.q_des:
            return False
        if self.dq_des != other.dq_des:
            return False
        if self.q_meas != other.q_meas:
            return False
        if self.dq_meas != other.dq_meas:
            return False
        if self.ddq_computed != other.ddq_computed:
            return False
        if self.error_endo != other.error_endo:
            return False
        if self.error_rcm != other.error_rcm:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def q_des(self):
        """Message field 'q_des'."""
        return self._q_des

    @q_des.setter
    def q_des(self, value):
        if __debug__:
            from std_msgs.msg import Float64MultiArray
            assert \
                isinstance(value, Float64MultiArray), \
                "The 'q_des' field must be a sub message of type 'Float64MultiArray'"
        self._q_des = value

    @builtins.property
    def dq_des(self):
        """Message field 'dq_des'."""
        return self._dq_des

    @dq_des.setter
    def dq_des(self, value):
        if __debug__:
            from std_msgs.msg import Float64MultiArray
            assert \
                isinstance(value, Float64MultiArray), \
                "The 'dq_des' field must be a sub message of type 'Float64MultiArray'"
        self._dq_des = value

    @builtins.property
    def q_meas(self):
        """Message field 'q_meas'."""
        return self._q_meas

    @q_meas.setter
    def q_meas(self, value):
        if __debug__:
            from std_msgs.msg import Float64MultiArray
            assert \
                isinstance(value, Float64MultiArray), \
                "The 'q_meas' field must be a sub message of type 'Float64MultiArray'"
        self._q_meas = value

    @builtins.property
    def dq_meas(self):
        """Message field 'dq_meas'."""
        return self._dq_meas

    @dq_meas.setter
    def dq_meas(self, value):
        if __debug__:
            from std_msgs.msg import Float64MultiArray
            assert \
                isinstance(value, Float64MultiArray), \
                "The 'dq_meas' field must be a sub message of type 'Float64MultiArray'"
        self._dq_meas = value

    @builtins.property
    def ddq_computed(self):
        """Message field 'ddq_computed'."""
        return self._ddq_computed

    @ddq_computed.setter
    def ddq_computed(self, value):
        if __debug__:
            from std_msgs.msg import Float64MultiArray
            assert \
                isinstance(value, Float64MultiArray), \
                "The 'ddq_computed' field must be a sub message of type 'Float64MultiArray'"
        self._ddq_computed = value

    @builtins.property
    def error_endo(self):
        """Message field 'error_endo'."""
        return self._error_endo

    @error_endo.setter
    def error_endo(self, value):
        if __debug__:
            from std_msgs.msg import Float64MultiArray
            assert \
                isinstance(value, Float64MultiArray), \
                "The 'error_endo' field must be a sub message of type 'Float64MultiArray'"
        self._error_endo = value

    @builtins.property
    def error_rcm(self):
        """Message field 'error_rcm'."""
        return self._error_rcm

    @error_rcm.setter
    def error_rcm(self, value):
        if __debug__:
            from std_msgs.msg import Float64MultiArray
            assert \
                isinstance(value, Float64MultiArray), \
                "The 'error_rcm' field must be a sub message of type 'Float64MultiArray'"
        self._error_rcm = value
