import collections.abc
from typing import Any, get_origin, get_args, List, Dict, Tuple, Union


def check_type(value: Any, expected_type: Any, allow_none: bool = False) -> bool:
    """
    Checks if a value matches the expected type, supporting complex types such as
    List[Type], Dict[KeyType, ValueType], Tuple[Type1, Type2], etc.
    """
    if value is None:
        return allow_none

    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # Handle basic types and user-defined classes
    if origin is None:
        return isinstance(value, expected_type)

    # Handle generic types from typing module
    if origin is list or origin is List:
        # Ensure value is a list and all elements match the expected element type
        return isinstance(value, list) and all(check_type(item, args[0]) for item in value)

    if origin is dict or origin is Dict:
        # Ensure value is a dict and all keys and values match the expected key and value types
        return (
            isinstance(value, dict) and
            all(check_type(k, args[0]) for k in value.keys()) and
            all(check_type(v, args[1]) for v in value.values())
        )

    if origin is tuple or origin is Tuple:
        # Ensure value is a tuple and all elements match the expected types
        if len(args) == 2 and args[1] is Ellipsis:
            # Case for Tuple[Type, ...] (variable-length tuples of a single type)
            return isinstance(value, tuple) and all(check_type(item, args[0]) for item in value)
        else:
            # Case for Tuple[Type1, Type2, ...] (fixed-length tuples)
            return isinstance(value, tuple) and len(value) == len(args) and all(check_type(item, arg) for item, arg in zip(value, args))

    if origin is Union:
        # Check if the value matches any of the types in the Union
        return any(check_type(value, arg) for arg in args)

    if origin is collections.abc.Callable:
        # Check if the value is a callable (function or method)
        return callable(value)

    if origin is collections.abc.Iterable:
        # Handle any iterable type, ensuring all elements match the specified type
        if len(args) == 0:
            # Case for Iterable (no type specified)
            return isinstance(value, collections.abc.Iterable)
        else:
            return isinstance(value, collections.abc.Iterable) and all(check_type(item, args[0]) for item in value)


    return False