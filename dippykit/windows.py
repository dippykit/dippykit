"""Module of window-generating functions

This module contains an assortment of functions that generate various windows.

"""

# This library was developed for the Georgia Tech graduate course ECE 6258:
# Digital Image Processing with Professor Ghassan AlRegib.
# For comments and feedback, please email dippykit[at]gmail.com

# Internal imports
from . import _utils

# Functional imports
import numpy as np

# General imports
import warnings
from typing import Union, Tuple

__author__ = 'Brighton Ancelin, Motaz Alfarraj, Ghassan AlRegib'

__all__ = ['window_2d']


def window_2d(
        support_size: Union[int, Tuple[int, int]],
        window_type: str='gaussian',
        **kwargs
        ) -> np.ndarray:
    """Generates a specified 2-dimensional window array.

    Returns a window with the specified parameters. The returned window
    is normalized such that the sum of all its elements is 1. When the window
    cannot be centered, the window will prefer top-left placement of pixels.

    :type support_size: ``int`` or ``Tuple[int, int]``
    :param support_size: Height and width of the window array in pixels.
    :type window_type: ``str``
    :param window_type: (default='gaussian') Type of window desired. Must be
        one of the following: *gaussian, rectangle, ellipse, circle*.
    :param kwargs: See below.
    :rtype: ``numpy.ndarray``
    :return: The desired, normalized 2-dimensional window.

    :Keyword Arguments:
        * *Gaussian windows*
            * **variance** (``float``) --
              (default=1.0) The sigma squared variance of the gaussian window.
        * *Rectangle windows*
            * **dimensions** (``int`` or ``Tuple[int, int]``) --
              (default= **support_size** ) The dimensions of the rectangle
              window. If a list of two integers is provided, the first
              element is the window height and the second element is the window
              width. If a single integer is provided, a square is generated.
        * *Ellipse windows*
            * **radii** (``int`` or ``Tuple[int, int]``) --
              (default= **support_size** /2) The radii of the ellipse window.
              If a list of two integers is provided, the first element is
              the window height-radius and the second element is the window
              width-radius. If a single integer is provided, a circle is
              generated.
        * *Circle windows*
            * **radius** (``int`` or ``Tuple[int, int]``) --
              (default= **support_size** /2) The radius of the circle window.

    Examples:

    >>> window_2d(5, 'rect', dim=(2,3))
    array([[0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.16666667, 0.16666667, 0.16666667, 0.        ],
           [0.        , 0.16666667, 0.16666667, 0.16666667, 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ]])
    >>> window_2d(5)
    array([[0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
           [0.01330621, 0.0596343 , 0.09832033, 0.0596343 , 0.01330621],
           [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
           [0.01330621, 0.0596343 , 0.09832033, 0.0596343 , 0.01330621],
           [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]])
    >>> window_2d(8, 'e', radii=(3,2))
    array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.05, 0.05, 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.05, 0.05, 0.05, 0.05, 0.  , 0.  ],
           [0.  , 0.  , 0.05, 0.05, 0.05, 0.05, 0.  , 0.  ],
           [0.  , 0.  , 0.05, 0.05, 0.05, 0.05, 0.  , 0.  ],
           [0.  , 0.  , 0.05, 0.05, 0.05, 0.05, 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.05, 0.05, 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]])

    """
    def gaussian():
        variance = _utils.get_arg_with_default(arg_dict, 'variance', 1)
        # Shift centers by -0.5 to accommodate the grid
        h_center = (support_size[0] / 2) - 0.5
        w_center = (support_size[1] / 2) - 0.5
        x, y = np.ogrid[(-h_center):(support_size[0] - h_center),
                (-w_center):(support_size[1] - w_center)]
        window_array = np.exp(-0.5 * (x**2 + y**2) / variance)
        # Normalize the window
        window_array /= np.sum(window_array)
        return window_array

    def rectangle():
        dimensions = _utils.get_arg_with_default(arg_dict, 'dimensions',
                                                 support_size)
        dimensions = _utils.resolve_shape_arg(dimensions, support_size,
                                                   'dimensions', False)
        (height, width) = dimensions
        # Should never happen thanks to resolve_shape_arg, but added as a
        # final precaution
        assert height <= support_size[0] and width <= support_size[1], \
            "Rectangle dimensions must be less than support_size"
        window_array = np.zeros(support_size)
        h_begin = int((support_size[0] - height) / 2)
        h_end = h_begin + height
        w_begin = int((support_size[1] - width) / 2)
        w_end = w_begin + width
        weight = 1/(height * width)
        window_array[h_begin:h_end, w_begin:w_end] = weight
        return window_array

    def ellipse():
        radii = _utils.get_arg_with_default(arg_dict, 'radii',
                                            [int(x/2) for x in support_size])
        radii = _utils.resolve_shape_arg(radii,
                                         [int(x/2) for x in support_size], 'radii', True)
        (h_radius, w_radius) = radii
        window_array = np.zeros(support_size)
        h_center = (support_size[0] / 2) - 0.5
        w_center = (support_size[1] / 2) - 0.5
        if ((h_center + 0.5) - h_radius) < 0 or \
                ((w_center + 0.5) - w_radius) < 0:
            warnings.warn("Figure extends beyond the support size and will "
                          "be clipped.")
        x, y = np.ogrid[(-h_center):(support_size[0]-h_center),
                (-w_center):(support_size[1]-w_center)]
        # Create a mask in the shape of the ellipse
        mask = (x/h_radius)**2 + (y/w_radius)**2 < 1
        weight = 1/np.sum(mask)
        window_array[mask] = weight
        return window_array

    def circle():
        radius = _utils.get_arg_with_default(arg_dict, 'radius',
                                             int(min(support_size)/2))
        # Assert that the kwargs are valid
        assert isinstance(radius, int) or isinstance(radius, float) or \
               1 == len(radius), "Keyword argument '{}' has a max size of " \
               "1".format('radius')
        # Add an appropriately-named entry in the arg_dict and let ellipse()
        # handle the rest
        arg_dict['radii'] = radius
        return ellipse()

    # Dictionary of window_type names and their associated functions and
    # function parameter names
    func_dict = {
        'gaussian': (gaussian, ['variance']),
        'rectangle': (rectangle, ['dimensions']),
        'ellipse': (ellipse, ['radii']),
        'circle': (circle, ['radius']),
    }
    # Ensure that the support_size is valid
    support_size = _utils.resolve_shape_arg_no_max(support_size,
                                                        'support_size')
    window_type = _utils.resolve_arg_from_list(window_type,
                                               list(func_dict.keys()))
    arg_dict = _utils.resolve_arg_dict_from_list(kwargs, func_dict[
                                                      window_type][1])
    return func_dict[window_type][0]()

