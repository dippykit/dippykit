"""Module of general functions intended to aid argument and return value
handling

This module contains an assortment of functions that perform simple
operations intended to simplify argument and return value handling.

"""

# Functional imports
import numpy as np

# General imports
from typing import List, Dict, Any, Union, Tuple
import warnings

__author__ = 'Brighton Ancelin'


NumericType = Union[int, float]
ShapeType = Union[Tuple[NumericType, NumericType], NumericType]


def resolve_arg_from_list(
        arg: str,
        allowed_arg_list: List[str],
        case_sensitive: bool=False,
        ) -> str:
    """Determines which argument a user intended to enter.

    Attempts to identify a string as a partially completed version of
    an element in a list of anticipated strings.

    :type arg: ``str``
    :param arg: A string which should be a partial completion of one of the
        strings in the allowed_arg_list list
    :type allowed_arg_list: ``List[str]``
    :param allowed_arg_list: A list of possible strings that the arg
        parameter could resolve to
    :type case_sensitive: ``bool``
    :param case_sensitive: (default=False) Should the string comparisons
        observe case
    :rtype: ``str``
    :return: The element in allowed_arg_list that is the completed version
        of arg, if only option exists
    :raises ValueError: If arg is a partial completion of multiple elements in
        allowed_arg_list *or* arg is not a partial completion of any
        elements in allowed_arg_list

    Examples:

    >>> resolve_arg_from_list('bla', ['red', 'green', 'blue', 'black'])
    'black'

    """
    if not case_sensitive:
        arg = arg.lower()
        allowed_arg_list = [x.lower() for x in allowed_arg_list]
    if arg in allowed_arg_list:
        return arg
    possible_args = list(allowed_arg_list.copy())
    prev_possible_args = possible_args.copy()
    for i in range(len(arg)):
        for possible_arg in possible_args.copy():
            if (i >= len(possible_arg)) or (arg[i] != possible_arg[i]):
                possible_args.remove(possible_arg)
        if 0 == len(possible_args):
            break
        prev_possible_args = possible_args.copy()
    if 1 == len(possible_args):
        return possible_args[0]
    raise ValueError("Argument '{}' not recognized. You may have meant one "
                     "of the following: {}".format(arg, prev_possible_args))


def resolve_arg_dict_from_list(
        keyword_args: Dict[str, Any],
        allowed_arg_list: List[str],
        case_sensitive: bool=False,
        warn_user_missing: bool=True,
        ) -> Dict[str, Any]:
    """Parses for arguments which a user intended to enter.

    For each key in keyword_args, this function attempts to identify the
    string key as a partially completed version of an element in a list of
    anticipated strings. Once a key is identified, the key's value is copied
    into the arg_dict dictionary with the completed string used as the new key.
    The arg_dict dictionary is then returned.

    If multiple keys in keyword_args could be completed as the same
    element, only the first key is evaluated; subsequent keys will send
    warnings. Unresolved keys will also send warnings.

    For each element in the allowed_arg_list list not identified, a warning
    will be sent if warn_user_missing is set to True.

    :param keyword_args: An iterable with string keys which should be
        partial completions of the strings in the allowed_arg_list list
    :type allowed_arg_list: ``List[str]``
    :param allowed_arg_list: A list of possible strings that the keys in the
        keyword_args parameter could resolve to
    :type case_sensitive: ``bool``
    :param case_sensitive: (default=False) Should the string comparisons
        observe case
    :type warn_user_missing: ``bool``
    :param warn_user_missing: (default=True) Should the user be warned for
        missing arguments
    :rtype: ``Dict[str, Any]``
    :return: A dictionary where keys are the elements in allowed_arg_list
        and the values are those identified in the keyword_args values.

    Examples:

    >>> resolve_arg_dict_from_list({'ti': 'noon', 'temp': 42}, ['time', \
'temperature'])
    {'time': 3, 'temperature': 42}

    """
    if not case_sensitive:
        allowed_arg_list = [x.lower() for x in allowed_arg_list]
    arg_dict = {}
    for key in keyword_args:
        try:
            if case_sensitive:
                arg_key = resolve_arg_from_list(key, allowed_arg_list)
            else:
                arg_key = resolve_arg_from_list(key.lower(), allowed_arg_list)
            if arg_key not in arg_dict:
                arg_dict[arg_key] = keyword_args[key]
            else:
                warnings.warn("Repeat keyword argument '{}' used. Only the "
                        "first instance will be evaluated.".format(arg_key))
        except ValueError:
            warnings.warn("Keyword argument '{}' unresolved.".format(key))

    if warn_user_missing:
        for key in allowed_arg_list:
            if key not in arg_dict.keys():
                warnings.warn("Keyword argument '{}' is missing.".format(key))
    return arg_dict

def get_arg_with_default(
        arg_dict: Dict[str, Any],
        key: str,
        default_value: Any
        ) -> Any:
    """Simple wrapper function for dictionary resolution with warnings

    Performs the operation arg_dict.get(key, default_value) on dictionary
    argument arg_dict, but first checks if key is in arg_dict. If the key is
    not found (and subsequently the default_value used), the user is warned.
    :type arg_dict: ``Dict``
    :param arg_dict: The dictionary
    :param key: The key
    :param default_value: The default return value
    :return: arg_dict.get(key, default_value)
    """
    if key not in arg_dict:
        warnings.warn("Argument '{}' is not defined. Using default value of "
                      "{} instead.".format(key, default_value))
    return arg_dict.get(key, default_value)


def resolve_shape_arg(
        shape_arg: ShapeType,
        max_shape: Tuple[int, int],
        arg_name: str,
        allow_larger_than_max: bool=True,
        ) -> Tuple[int, int]:
    """Determines the 2d tuple intent of a shape argument.

    Given an int, float, index-able int object of length 1 or 2, or index-able
    float object of length 1 or 2, this function will return the
    interpretation of this variable as a length 2 tuple of ints.

    For single-element or scalar elements, the shape is assumed to be
    square. If shape_arg does not adhere to the above specifications for
    acceptable type, assertion errors will be thrown from this function.

    For integer-based objects, the interpretation of the integer(s) is
    index/pixel size. For float-based objects, the interpretation of the
    float(s) is percentage of the max_shape dimensions.

    All values must be non-negative, and if argument allow_larger_than_max
    is passed as ``False``, integer values must not be greater than
    max_shape and float values must not be greater than 1. When argument
    allow_larger_than_max is set to ``True`` (default) however, warnings
    will be displayed when int values exceed max_shape or float values
    exceed 1.

    :type shape_arg: ``ShapeType``
    :param shape_arg: The argument to be interpreted as a 2d shape tuple
    :type max_shape: ``Tuple[int, int]``
    :param max_shape: The largest shape expected (or the largest shape
        possible if allow_larger_than_max is set to ``False``) to be
        interpreted from shape_arg
    :type arg_name: ``str``
    :param arg_name: The name to be used for referencing this argument if
        error or warnings are displayed
    :type allow_larger_than_max: ``bool``
    :param allow_larger_than_max: (default=``True``) Whether to allow the
        returned interpreted shape to exceed max_shape. This is useful for
        functions that may want to allow cropping **or** zero-padding with a
        single shape argument, for example.
    :rtype: ``Tuple[int, int]``
    :return: The interpreted shape in index/pixel domain
    :raises AssertionError: If shape_arg does not adhere to one of the
        allowed formats for shape or if shape_arg contains values outside
        allowed values.

    Examples:

    >>> resolve_shape_arg(8, (256, 256), 'block_size')
    (8, 8)
    >>> import numpy as np
    >>> resolve_shape_arg(np.array([8]), (256, 256), 'block_size')
    (8, 8)
    >>> resolve_shape_arg([8, 4], (256, 256), 'block_size')
    (8, 4)
    >>> resolve_shape_arg([256, 512], (512, 512), 'block_size')
    (256, 512)
    >>> resolve_shape_arg([400, 200], (200, 200), 'block_size')
    (400, 200)
    UserWarning: An integer block_size greater than the output image
    dimensions will result in zero padding
    >>> resolve_shape_arg(np.array([300, 128]), (512, 64), 'new_shape', False)
    AssertionError: An integer new_shape must be less than or equal to 1

    """
    if isinstance(shape_arg, np.ndarray):
        shape_arg = shape_arg.tolist()
    assert isinstance(shape_arg, int) or \
           isinstance(shape_arg, float) or \
           isinstance(shape_arg[0], int) or \
           isinstance(shape_arg[0], float), \
        "Argument '{}' must be either int or float".format(arg_name)
    assert isinstance(shape_arg, int) or \
           isinstance(shape_arg, float) or \
           2 == len(shape_arg) or 1 == len(shape_arg), \
        "Argument '{}' has a max size of 2".format(arg_name)

    if isinstance(shape_arg, float):
        assert 0 <= shape_arg, "float {} must be non-negative".format(arg_name)
        if allow_larger_than_max:
            if 1 < shape_arg:
                warnings.warn("A float {} greater than 1 will "
                              "result in zero padding".format(arg_name))
        else:
            assert not 1 < shape_arg, "A float {} must be less than or " \
                                      "equal to 1".format(arg_name)
    elif isinstance(shape_arg, int):
        assert 0 <= shape_arg, "int {} must be non-negative".format(arg_name)
        if allow_larger_than_max:
            if min(max_shape) < shape_arg:
                warnings.warn("An integer {} greater than the output image "
                              "dimensions will result in zero padding"
                              .format(arg_name))
        else:
            assert not min(max_shape) < shape_arg, "An integer {} must be " \
                    "less than or equal to 1".format(arg_name)
    elif isinstance(shape_arg[0], float):
        assert all(0 <= x for x in shape_arg), "float {} must be " \
                                               "non-negative".format(arg_name)
        if allow_larger_than_max:
            if any([1 < x for x in shape_arg]):
                warnings.warn("A float {} greater than 1 will "
                              "result in zero padding".format(arg_name))
        else:
            assert not any([1 < x for x in shape_arg]), "A float {} must be" \
                    "less than or equal to 1".format(arg_name)
    elif isinstance(shape_arg[0], int) and 1 == len(shape_arg):
        assert all(0 <= x for x in shape_arg), "int {} must be " \
                                               "non-negative".format(arg_name)
        if allow_larger_than_max:
            if min(max_shape) < shape_arg[0]:
                warnings.warn("An integer {} greater than the output image "
                              "dimensions will result in zero padding"
                              .format(arg_name))
        else:
            assert not min(max_shape) < shape_arg[0], "An integer {} must be" \
                    " less than or equal to 1".format(arg_name)
    elif isinstance(shape_arg[0], int) and 2 == len(shape_arg):
        assert all(0 <= x for x in shape_arg), "int {} must be " \
                                               "non-negative".format(arg_name)
        if allow_larger_than_max:
            if max_shape[0] < shape_arg[0] or max_shape[1] < shape_arg[1]:
                warnings.warn("An integer {} greater than the output image "
                              "dimensions will result in zero padding"
                              .format(arg_name))
        else:
            assert not (max_shape[0] < shape_arg[0] or
                        max_shape[1] < shape_arg[1]), "An integer {} must be" \
                        " less than or equal to 1".format(arg_name)


    if isinstance(shape_arg, int):
        shape_arg = (shape_arg, shape_arg)
    elif isinstance(shape_arg, float):
        shape_arg = ((shape_arg * max_shape[0]).astype(int),
                     (shape_arg * max_shape[1]).astype(int))
    elif isinstance(shape_arg[0], int) and 1 == len(shape_arg):
        shape_arg = (shape_arg[0], shape_arg[0])
    elif isinstance(shape_arg[0], int) and 2 == len(shape_arg):
        shape_arg = (shape_arg[0], shape_arg[1])
    elif isinstance(shape_arg[0], float) and 1 == len(shape_arg):
        shape_arg = ((shape_arg[0] * max_shape[0]).astype(int),
                     (shape_arg[0] * max_shape[1]).astype(int))
    elif isinstance(shape_arg[0], float) and 2 == len(shape_arg):
        shape_arg = ((shape_arg[0] * max_shape[0]).astype(int),
                     (shape_arg[1] * max_shape[1]).astype(int))

    return shape_arg


def resolve_shape_arg_no_max(
        shape_arg: Union[Tuple[int, int], int],
        arg_name: str
        ) -> Tuple[int, int]:
    """Determines the 2d tuple intent of a shape argument with no assumed 
    maximum shape.

    Given an int, or index-able int object of length 1 or 2, this function 
    will return the interpretation of this variable as a length 2 tuple of 
    ints.

    For single-element or scalar elements, the shape is assumed to be
    square. If shape_arg does not adhere to the above specifications for
    acceptable type, assertion errors will be thrown from this function. All 
    values must be non-negative

    :type shape_arg: ``int`` or ``Tuple[int, int]``
    :param shape_arg: The argument to be interpreted as a 2d shape tuple
    :type arg_name: ``str``
    :param arg_name: The name to be used for referencing this argument if
        error or warnings are displayed
    :rtype: ``Tuple[int, int]``
    :return: The interpreted shape in index/pixel domain
    :raises AssertionError: If shape_arg does not adhere to one of the
        allowed formats for shape or if shape_arg contains values outside
        allowed values.

    Examples:

    >>> resolve_shape_arg_no_max(5, 'window_size')
    (5, 5)
    >>> import numpy as np
    >>> resolve_shape_arg_no_max(np.array([8]), 'size_1')
    (8, 8)
    >>> resolve_shape_arg_no_max((3, 4), 'tri_dim')
    (3, 4)
    """
    if isinstance(shape_arg, np.ndarray):
        shape_arg = shape_arg.tolist()
    assert isinstance(shape_arg, int) or isinstance(shape_arg[0], int), \
        "Argument '{}' must be of type int".format(arg_name)
    assert isinstance(shape_arg, int) or \
           2 == len(shape_arg) or 1 == len(shape_arg), \
        "Argument '{}' has a max size of 2".format(arg_name)

    if isinstance(shape_arg, int):
        assert 0 <= shape_arg, "int {} must be non-negative".format(arg_name)
    elif isinstance(shape_arg[0], int):
        assert all(0 <= x for x in shape_arg), \
                "int {} must be non-negative".format(arg_name)

    if isinstance(shape_arg, int):
        shape_arg = (shape_arg, shape_arg)
    elif isinstance(shape_arg[0], int) and 1 == len(shape_arg):
        shape_arg = (shape_arg[0], shape_arg[0])

    return shape_arg

