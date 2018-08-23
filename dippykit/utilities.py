"""Module of image processing utility functions

This module contains an assortment of general purpose or otherwise
ill-categorized functions that are useful in image processing.

"""

# This library was developed for the Georgia Tech graduate course ECE 6258:
# Digital Image Processing with Professor Ghassan AlRegib.
# For comments and feedback, please email dippykit[at]gmail.com

# Internal imports
from . import _utils

# Functional imports
import numpy as np
from scipy import signal

# General imports
import warnings
from typing import Union, Tuple, Callable, Any

__author__ = 'Brighton Ancelin, Motaz Alfarraj, Ghassan AlRegib'

__all__ = ['block_process', 'zigzag_indices', 'conv2d_grid', 'rgb2ycbcr',
           'rgb2gray', 'convolve2d']


def block_process(
        im: np.ndarray,
        func: Callable[[np.ndarray], Any],
        block_size: _utils.ShapeType=(8, 8),
        stride: _utils.ShapeType=None,
        ) -> np.ndarray:
    """Performs an operation on an array/image in smaller blocks and
    concatenates the results

    Given an array/image, a function, and a block size, this function will
    iterate through all non-overlapping blocks of size block_size and
    perform the specified function on each block. The return values of the
    specified function for each block are then concatenated together and
    returned as a new array.

    Blocks begin with array index [0, 0] and continue until all values in
    the input array have been used. If a full block can't be created from
    the area a block would occupy, a partial block will be evaluated (e.g.
    using a block size of (8, 8) on an image that is of size (9, 9) will
    result in a block of size (8, 8), a block of size (8,1), a block of size
    (1, 8), and a block of size (1, 1)).

    The function argument, func, can return anything that numpy is able to
    convert to an ``ndarray``. This value may be a scalar, vector, matrix,
    or anything that will consistently concatenate as blocks progress along
    each axis.

    If a stride is specified, then blocks edges will be separated by the
    stride argument instead of by the block size itself. This can be used to
    process overlapping blocks or to skip over regions of an array/image.

    :type im: ``numpy.ndarray``
    :param im: The array or image to be processed.
    :type func: ``Callable[[np.ndarray], Any]``
    :param func: The function to apply to each block. The return value must
        be something that numpy can convert to an ndarray.
    :type block_size: ``ShapeType``
    :param block_size: (default=(8, 8)) The size of each block.
    :type stride: ``ShapeType``
    :param stride: (default=block_size) The stride to take when moving to
        the next block.
    :rtype: ``numpy.ndarray``
    :return: The concatenated output of the function func for each block.

    Examples:

    >>> import numpy as np
    >>> im = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
    ...                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ...                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    ...                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    ...                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    ...                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    ...                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
    ...                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
    ...                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
    ...                [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
    >>> block_process(im, np.transpose, (2, 2))
    array([[ 0, 10,  2, 12,  4, 14,  6, 16,  8, 18],
           [ 1, 11,  3, 13,  5, 15,  7, 17,  9, 19],
           [20, 30, 22, 32, 24, 34, 26, 36, 28, 38],
           [21, 31, 23, 33, 25, 35, 27, 37, 29, 39],
           [40, 50, 42, 52, 44, 54, 46, 56, 48, 58],
           [41, 51, 43, 53, 45, 55, 47, 57, 49, 59],
           [60, 70, 62, 72, 64, 74, 66, 76, 68, 78],
           [61, 71, 63, 73, 65, 75, 67, 77, 69, 79],
           [80, 90, 82, 92, 84, 94, 86, 96, 88, 98],
           [81, 91, 83, 93, 85, 95, 87, 97, 89, 99]])
    >>> block_process(im, np.flipud, (3, 5))
    array([[20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
           [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
           [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
           [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
           [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
           [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
           [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
           [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
           [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
    >>> block_process(im, np.mean, (5, 5))
    array([[22., 27.],
           [72., 77.]])
    >>> block_process(im, np.mean, (2, 2), stride=(4, 4))
    array([[16.5, 20.5, 23.5],
           [56.5, 60.5, 63.5],
           [86.5, 90.5, 93.5]])
    """
    block_size = _utils.resolve_shape_arg(block_size, im.shape,
                                               'block_size', False)
    if stride is None:
        row_stride = block_size[0]
        col_stride = block_size[1]
    else:
        (row_stride, col_stride) = _utils.resolve_shape_arg(stride,
                                                            im.shape, 'stride',
                                                            False)
    height = im.shape[0]
    width = im.shape[1]
    result = None
    for row in range(0, height, row_stride):
        row_result = None
        for col in range(0, width, col_stride):
            cur_result = func(im[row:min(row + row_stride, height),
                                 col:min(col + col_stride, width)])
            cur_result = np.atleast_2d(cur_result)
            if row_result is not None:
                row_result = np.concatenate((row_result, cur_result), axis=1)
            else:
                row_result = cur_result
        if result is not None:
            result = np.concatenate((result, row_result), axis=0)
        else:
            result = row_result
    return result


def zigzag_indices(
        shape: Union[Tuple[int, int], int],
        length: int=-1,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Generates indices for a zigzag pattern in a matrix

    Given a matrix shape, this function will determine the indices in such a
    matrix of consecutive elements in a zigzag pattern. If an integer *length*
    is specified, the indices of only *length* elements will be returned.
    *It is almost always preferable to specify a length if one does not need
    every element's indices. This saves computation time.*

    The zigzag pattern used here is akin to the one used in JPEG.

    :type shape: ``Tuple[int, int]`` or ``int``
    :param shape: The shape of the matrix for which the zigzag indices are
        applicable.
    :type length: ``int``
    :param length: (default=shape[0]*shape[1]) The number of indices to be
        returned; equivalently, the length of each ndarray returned.
    :rtype: ``Tuple[numpy.ndarray, numpy.ndarray]``
    :return: A length 2 tuple of indices of zigzag values.

    Examples:

    >>> import numpy as np
    >>> im = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
    ...                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ...                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    ...                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    ...                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    ...                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    ...                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
    ...                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
    ...                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
    ...                [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
    >>> im[zigzag_indices(im.shape, 10)]
    array([ 0,  1, 10, 20, 11,  2,  3, 12, 21, 30])
    >>> im_2 = np.array([[ 0,  1,  2],
    ...                  [10, 11, 12],
    ...                  [20, 21, 22]])
    >>> ind = zigzag_indices(im_2.shape)
    >>> ind
    (array([0, 0, 1, 2, 1, 0, 1, 2, 2]), array([0, 1, 0, 0, 1, 2, 2, 1, 2]))
    >>> im_2[ind]
    array([ 0,  1, 10, 20, 11,  2, 12, 21, 22])
    """
    shape = _utils.resolve_shape_arg_no_max(shape, 'shape')
    if -1 == length:
        length = shape[0] * shape[1]
    indices = np.zeros((2, length), dtype=int)
    i = 0
    for antidiag_sum in range(shape[0] + shape[1] - 1):
        row_start = max(0, antidiag_sum - shape[1] + 1)
        col_start = max(0, antidiag_sum - shape[0] + 1)
        row_stop = min(shape[0], antidiag_sum + 1)
        col_stop = min(shape[1], antidiag_sum + 1)
        antidiag_length = row_stop - row_start
        step_overshoot = i + antidiag_length - length
        if 0 >= step_overshoot:
            if 0 == antidiag_sum % 2:
                indices[0, i:(i + antidiag_length)] = \
                        np.arange(row_stop - 1, row_start - 1, -1)
                indices[1, i:(i + antidiag_length)] = \
                    np.arange(col_start, col_stop)
            else:
                indices[0, i:(i + antidiag_length)] = \
                    np.arange(row_start, row_stop)
                indices[1, i:(i + antidiag_length)] = \
                    np.arange(col_stop - 1, col_start - 1, -1)
        else:
            if 0 == antidiag_sum % 2:
                row_stop = min(shape[0], antidiag_sum + 1)
                col_start = max(0, antidiag_sum - shape[0] + 1)
                row_start = row_stop - (antidiag_length - step_overshoot)
                col_stop = col_start + (antidiag_length - step_overshoot)
                indices[0, i:] = np.arange(row_stop - 1, row_start - 1, -1)
                indices[1, i:] = np.arange(col_start, col_stop)
            else:
                row_start = max(0, antidiag_sum - shape[1] + 1)
                col_stop = min(shape[1], antidiag_sum + 1)
                row_stop = row_start + (antidiag_length - step_overshoot)
                col_start = col_stop - (antidiag_length - step_overshoot)
                indices[0, i:] = np.arange(row_start, row_stop)
                indices[1, i:] = np.arange(col_stop - 1, col_start - 1, -1)
        i += antidiag_length
        if 0 <= step_overshoot:
            break

    return tuple(indices)


def conv2d_grid(
        f_lower: Tuple[_utils.NumericType, _utils.NumericType],
        g_lower: Tuple[_utils.NumericType, _utils.NumericType],
        f: np.ndarray,
        g: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a meshgrid for the domain of a convolution result

    Given the domain values of the [0, 0] index in the
    two-dimensional signals f and g and the signals f and g themselves,
    this function returns a meshgrid of the domain of f convolved with g.

    A step size of 1 is assumed uniformly for both signal domains. For
    example, if the domain value of the [0, 0] index in f is
    [x=\ **3**\ , y=\ **1**\ ], then this function assumes the domain value
    of the [1, 7] index in f is [x=(3+1)=\ **4**\ , y=(1+7)=\ **8**\ ].

    A bit of background:

    A common point of obscurity in image processing is the treatment of
    arrays as actual arrays vs. two-dimensional signals. When dealing with
    the latter, an underlying domain is implied. For convolution operations
    on two-dimensional signals in particular, keeping track of the changes
    in domain can be annoying and tricky. This function exists to be a
    simple, reliable way to track the domains of your signals after
    convolution operations.

    :type f_lower: ``Tuple[int or float, int or float]``
    :param f_lower: The domain value of the point f[0, 0].
    :type g_lower: ``Tuple[int or float, int or float]``
    :param g_lower: The domain value of the point g[0, 0].
    :type f: ``numpy.ndarray``
    :param f: An ndarray interpreted as a two-dimensional signal.
    :type g: ``numpy.ndarray``
    :param g: An ndarray interpreted as a two-dimensional signal.
    :rtype: ``Tuple[numpy.ndarray, numpy.ndarray]``
    :return: The domain of the convolution of f and g in meshgrid format.

    Examples:

    >>> import numpy as np
    >>> import dippykit as dip
    >>> f_domain_x, f_domain_y = np.meshgrid(np.arange(-5, 5), np.arange(0, 7))
    >>> f = np.zeros(f_domain_x.shape)
    >>> f[(0 == f_domain_x) & (0 == f_domain_y)] = 1
    >>> f
    array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    >>> g_domain_x, g_domain_y = np.meshgrid(np.arange(0, 5), np.arange(-5, 2))
    >>> g = np.zeros(g_domain_x.shape)
    >>> g[(0 == g_domain_x) & (0 == g_domain_y)] = 1
    >>> g
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])
    >>> fg = dip.convolve2d(f, g)
    >>> fg
    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    >>> f_lower = (f_domain_x[0, 0], f_domain_y[0, 0])
    >>> g_lower = (g_domain_x[0, 0], g_domain_y[0, 0])
    >>> fg_domain_x, fg_domain_y = conv2d_grid(f_lower, g_lower, f, g)
    >>> fg[(0 == fg_domain_x) & (0 == fg_domain_y)]
    array([1.])
    """
    f_lower = np.array(f_lower)
    g_lower = np.array(g_lower)
    f_upper = f_lower + f.shape[0:2] - 1
    g_upper = g_lower + g.shape[0:2] - 1
    xy_start = f_lower + g_lower
    xy_end = f_upper + g_upper
    x_grid, y_grid = np.meshgrid(np.arange(xy_start[1], xy_end[1]+1),
                                 np.arange(xy_start[0], xy_end[0]+1))
    return x_grid, y_grid


def rgb2ycbcr(
        rgb_im: np.ndarray,
        ) -> np.ndarray:
    """Converts an RGB image to YCbCr, in exactly the same manner as Matlab

    This function will return the same output as Matlab's rgb2ycbcr function
    for images with a dtype of ``uint8``.

    :type rgb_im: ``numpy.ndarray``
    :param rgb_im: The RGB image to be converted.
    :rtype: ``numpy.ndarray``
    :return: A YCbCr image, matching the Matlab conversion metric for dtypes
        of ``numpy.uint8``.

    Examples:

    >>> import numpy as np
    >>> im = np.array([[[255, 255, 255], [128, 128, 128]],
    ...                [[  0,   0,   0], [ 64,  32, 224]]], dtype=np.uint8)
    >>> rgb2ycbcr(im)
    array([[[235, 128, 128],
            [126, 128, 128]],
           [[ 16, 128, 128],
            [ 70, 208, 128]]], dtype=uint8)

    """
    assert 3 == len(rgb_im.shape), "rgb2ycbcr must have a 3-channel rgb " \
            "image as input"
    if not ('uint8' == rgb_im.dtype.name):
        warnings.warn("Function rgb2ycbcr may not work exactly as Matlab for "
                      "input images with a dtype of {}"
                      .format(rgb_im.dtype.name))
    rgb_im_float = rgb_im.astype(np.float)
    weight_tensor = np.array([[ 65.481, 128.553,  24.966],
                              [-37.797, -74.203, 112.   ],
                              [112.   , -93.786, -18.214]])
    weight_tensor /= 255
    bias_tensor = np.array([16, 128, 128]).reshape(1, 1, 3)
    yuv_primitive = np.tensordot(rgb_im_float, weight_tensor, axes=(2, 1))
    yuv_u = yuv_primitive + bias_tensor
    if rgb_im.dtype == np.uint8:
        return np.round(yuv_u).astype(np.uint8)
    return np.round(yuv_u)


def rgb2gray(
        rgb_im: np.ndarray,
        ) -> np.ndarray:
    """Converts a 3-channel RGB image to 1-channel gray image

    Given a 3-channel RGB image, this function converts the image to a
    1-channel gray image with particular weights for each channel. These
    weights are the same as those used in the Matlab rgb2gray function.

    :type rgb_im: ``numpy.ndarray``
    :param rgb_im: The input 3-channel RGB image.
    :rtype: ``numpy.ndarray``
    :return: The output 1-channel gray image.

    Examples:

    >>> import numpy as np
    >>> im = np.array([[[255, 255, 255], [128, 128, 128]],
    ...                [[  0,   0,   0], [ 64,  32, 224]]], dtype=np.uint8)
    >>> rgb2gray(im)
    array([[255, 128],
           [  0,  63]], dtype=uint8)

    """
    assert 3 == len(rgb_im.shape), "rgb2ycbcr must have a 3-channel rgb " \
                                   "image as input"
    weight_tensor = np.array([0.299, 0.5870, 0.1140])
    gray_im = np.tensordot(rgb_im, weight_tensor, axes=(2, 0))
    if rgb_im.dtype.kind == 'u' or rgb_im.dtype.kind == 'i':
        gray_im = np.round(gray_im)
    gray_im = gray_im.astype(rgb_im.dtype)
    return gray_im


def convolve2d(
        sig_1: np.ndarray,
        sig_2: np.ndarray,
        mode: str='full',
        boundary: str='fill',
        fillvalue: _utils.NumericType=0,
        like_matlab: bool=False,
        ) -> np.ndarray:
    """Convolves two arrays in two dimensions

    Given two input arrays, this function computes their 2D convolution. The
    input arrays must be either 2-dimensional or 3-dimensional, and the
    second input array must have less than or equal dimensions to the first.

    Convolution is always computed across the first and second dimensions of
    the input arrays. In the case that both input arrays are 3-dimensional,
    the result is the sum of 2D convolutions for each corresponding 2D
    slice of the input arrays. This result is 2-dimensional. In the case that
    only the first input array is 3-dimensional, the result is the stack of
    2D convolutions for each 2D slice of the first input array with the
    second input array. This result is 3-dimensional.

    In cases where either the first or second dimension of the second input
    array is an even integer, and where the mode is 'same', this function
    will prefer lesser indices in centering by default. Setting the
    'like_matlab' keyword argument to True will alter this behavior to
    prefer greater indices in centering. This is better understood through
    demonstration in the examples below.

    This function is essentially a wrapper for `scipy.signal.convolve2d`_,
    so more detailed documentation may be found there.

    :type sig_1: ``numpy.ndarray``
    :param sig_1: The first input array.
    :type sig_2: ``numpy.ndarray``
    :param sig_2: The second input array.
    :type mode: ``str``
    :param mode: (default='full') The mode to be used for the convolution.
        If set to 'full', a full convolution will be performed. If set to
        'same', the output will be the same size (in the first two
        dimensions) as the first input array. If set to valid, only values
        generated without padding will be retained. For more information,
        see `scipy.signal.convolve2d`_.
    :type boundary: ``str``
    :param boundary: (default='fill') A string representing how boundaries
        will be handled. If set to 'fill', the first input array is padded
        with fillvalue values. If set to 'wrap', the first input array is
        padded with copies of itself in a tile configuration. If set to
        'symm', the first input array is padded with reflected copies of
        itself. For more information, see `scipy.signal.convolve2d`_.
    :type fillvalue: ``NumericType``
    :param fillvalue: (default=0) The numeric value to pad the first input
        array with (assuming boundary='fill').
    :type like_matlab: ``bool``
    :param like_matlab: (default=False) If set to True, this function will
        return arrays in the same manner that Matlab would.
    :rtype: ``numpy.ndarray``
    :return: The convolution of the two arrays.

    .. note::
        This function wraps around functions from other packages. Reading
        these functions' documentations may be useful. See the **See also**
        section for more information.

    .. seealso::
        `scipy.signal.convolve2d`_
            Documentation of the convolve2d function from Scipy

    .. _scipy.signal.convolve2d: https://docs.scipy.org/doc/scipy/reference
        /generated/scipy.signal.convolve2d.html

    Examples:

    >>> a = np.array([[1, 0, 2],
    ...               [0, 3, 0],
    ...               [4, 0, 5]])
    >>> b = np.array([[ 1,  1],
    ...               [-1, -1]])
    >>> convolve2d(a, b)
    array([[ 1,  1,  2,  2],
           [-1,  2,  1, -2],
           [ 4,  1,  2,  5],
           [-4, -4, -5, -5]])
    >>> convolve2d(a, b, mode='same')
    array([[ 1,  1,  2],
           [-1,  2,  1],
           [ 4,  1,  2]])
    >>> convolve2d(a, b, mode='same', like_matlab=True)
    array([[ 2,  1, -2],
           [ 1,  2,  5],
           [-4, -5, -5]])
    >>> c = np.stack(([[1, 1],
    ...                [1, 1]],
    ...               [[2, 2],
    ...                [2, 2]],
    ...               [[3, 3],
    ...                [3, 3]]), axis=2)
    >>> d = convolve2d(c, b)
    >>> d[:, :, 0]
    array([[ 1,  2,  1],
           [ 0,  0,  0],
           [-1, -2, -1]])
    >>> d[:, :, 1]
    array([[ 2,  4,  2],
           [ 0,  0,  0],
           [-2, -4, -2]])
    >>> d[:, :, 2]
    array([[ 3,  6,  3],
           [ 0,  0,  0],
           [-3, -6, -3]])

    """
    def do_convolve2d():
        if 3 == sig_1.ndim:
            if 2 == sig_2.ndim:
                sig_out_0 = signal.convolve2d(sig_1[:, :, 0], sig_2,
                        mode=mode, boundary=boundary, fillvalue=fillvalue)
                sig_out = np.zeros((*sig_out_0.shape, sig_1.shape[2]),
                                   dtype=sig_out_0.dtype)
                sig_out[:, :, 0] = sig_out_0
                for i in range(1, sig_1.shape[2]):
                    sig_out[:, :, i] = signal.convolve2d(sig_1[:, :, i],
                            sig_2, mode=mode, boundary=boundary,
                            fillvalue=fillvalue)
            else:
                assert sig_1.shape[2] == sig_2.shape[2], \
                        "3 dimensional signal inputs must have same 3rd " \
                        "dimension shape"
                sig_out = signal.convolve2d(sig_1[:, :, 0], sig_2[:, :, 0],
                        mode=mode, boundary=boundary, fillvalue=fillvalue)
                for i in range(1, sig_1.shape[2]):
                    sig_out += signal.convolve2d(sig_1[:, :, i],
                            sig_2[:, :, i], mode=mode, boundary=boundary,
                            fillvalue=fillvalue)
        else:
            sig_out = signal.convolve2d(sig_1, sig_2, mode=mode,
                    boundary=boundary, fillvalue=fillvalue)
        return sig_out
    assert (2 == sig_1.ndim) or (3 == sig_1.ndim), \
            "Signal 1 must be 2 or 3 dimensional"
    assert (2 == sig_2.ndim) or (3 == sig_2.ndim), \
            "Signal 2 must be 2 or 3 dimensional"
    assert (sig_1.ndim >= sig_2.ndim), \
            "Signal 1 must be equal or greater than signal 2 in terms of " \
            "dimensions"
    if like_matlab:
        if 'same' == mode:
            if (0 == sig_2.shape[0] % 2) or (0 == sig_2.shape[1] % 2):
                row_start = sig_2.shape[0] // 2
                col_start = sig_2.shape[1] // 2
                row_end = row_start + sig_1.shape[0]
                col_end = col_start + sig_1.shape[1]
                mode = 'full'
                sig_out = do_convolve2d()
                return sig_out[row_start:row_end, col_start:col_end]
    return do_convolve2d()

