"""Module of image adjustment functions

This module contains an assortment of functions for use in adjusting the
specifics of images. This includes noise, domain expansion/contraction,
shifts, etc.

"""

# This library was developed for the Georgia Tech graduate course ECE 6258:
# Digital Image Processing with Professor Ghassan AlRegib.
# For comments and feedback, please email dippykit[at]gmail.com

# Internal imports
from . import _utils
from . import image_io

# Functional imports
import numpy as np
from skimage.util import random_noise

# General imports
import warnings

__author__ = 'Brighton Ancelin, Motaz Alfarraj, Ghassan AlRegib'

__all__ = ['image_noise', 'image_adjust', 'image_translate', 'image_shift']


def image_noise(
        im: np.ndarray,
        mode: str='gaussian',
        **kwargs
        ) -> np.ndarray:
    """Adds random noise to an image

    Adds random noise to an input image. For images with an integer dtype,
    the returned image will retain the original image's dtype. In all other
    cases, the returned image will have a float dtype and be within the
    normalized range from 0 to 1.

    This function is essentially a wrapper for `skimage.util.random_noise`_,
    so more detailed documentation may be found there.

    :type im: ``numpy.ndarray``
    :param im: The original image.
    :type mode: ``str``
    :param mode: (default='gaussian') The type of noise to add to the image.
        May be one of the following: *'gaussian', 'localvar', 'poisson',
        'salt', 'pepper', 's&p', 'speckle'*. Passed as the mode argument to
        `skimage.util.random_noise`_.
    :param kwargs: See below.
    :rtype: ``numpy.ndarray``
    :return: The noisy image.

    :Keyword Arguments:
        * **mean** (``int`` or ``float``) --
          The mean of the randomly distributed noise. This value's meaning
          is dependent on the dtype of the image (e.g. mean=127 for an image
          with dtype=uint8 is analogous to mean=(127 / 255) for an image with
          dtype=float).
        * **var** (``int`` or ``float``) --
          The variance of the randomly distributed noise. This value's
          meaning is dependent on the dtype of the image (e.g. var=(0.01 * (
          255 ** 2)) for an image with dtype=uint8 is analogous to var=0.01
          for an image with dtype=float).
        * **local_vars** (``numpy.ndarray``) --
          The array of local variances for the randomly distributed noise.
          This value's meaning is dependent on the dtype of the image. Each
          element in the array is treated in the same way the var keyword
          argument would be.
        * A full list of keyword arguments along with descriptions can be
          found at `skimage.util.random_noise`_.

    .. note::
        This function wraps around functions from other packages. Reading
        these functions' documentations may be useful. See the **See also**
        section for more information.

    .. seealso::
        `skimage.util.random_noise`_
            Documentation of the random_noise function from Scikit Image

    .. _skimage.util.random_noise: http://scikit-image.org/docs/dev/api/
        skimage.util.html#skimage.util.random_noise

    Examples:

    >>> import numpy as np
    >>> im = np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)
    >>> im
    array([[238, 208, 243, 228, 198, 186,  72, 181],
           [ 92, 247,  98, 197, 203, 210, 219, 175],
           [124,  66, 231, 193, 154, 153, 136,  59],
           [218, 162,  65, 196, 181, 160,  13, 148],
           [164,  26, 150, 202,  55, 138,  40, 116],
           [ 46, 151,  97,  78, 169, 249, 181, 167],
           [ 79,  72,  31,  98, 152,  80, 220, 152],
           [ 19, 233, 111, 126, 176,  68,  40,  49]], dtype=uint8)
    >>> image_noise(im)
    array([[246, 220, 211, 216, 216, 157,  51, 176],
           [ 79, 230,  68, 217, 211, 175, 244, 180],
           [ 86,  75, 184, 178, 123, 183, 106,  61],
           [201, 154,  73, 193, 163, 180,  33, 180],
           [192,  19, 144, 197,  64,  89,  36,  99],
           [ 58, 182,  81,  53, 154, 255, 193, 145],
           [ 60,  62,  55, 122, 132,  97, 244, 167],
           [ 33, 181, 132, 163, 190,   9,  41,  45]], dtype=uint8)

    """
    try:
        im_float = image_io.im_to_float(im)
        success = True
        dtype_max_val = np.iinfo(im.dtype).max
        if 'mean' in kwargs:
            kwargs['mean'] /= dtype_max_val
        if 'var' in kwargs:
            kwargs['var'] /= (dtype_max_val ** 2)
        if 'local_vars' in kwargs:
            kwargs['local_vars'] /= (dtype_max_val ** 2)
    except ValueError:
        im_float = im
        success = False
        if np.any(0 > im) or np.any(1 < im):
            warnings.warn('Image passed to image_noise() was not able to be '
                      'converted to a normalized floating point range. This '
                      'is most likely due to having the image in a '
                      'non-integer dtype. Returned noisy image will have a '
                      'normalized floating point range.')
    if success:
        return image_io.float_to_im(random_noise(im_float, mode=mode,
                **kwargs), np.iinfo(im.dtype).bits)
    else:
        return random_noise(im_float, mode=mode, **kwargs)


def image_adjust(
        im: np.ndarray,
        lower_in: _utils.NumericType=None,
        upper_in: _utils.NumericType=None,
        lower_out: _utils.NumericType=None,
        upper_out: _utils.NumericType=None,
        keep_dtype: bool=True,
        ) -> np.ndarray:
    """Adjusts a given image by clipping and scaling its range

    Given an image and parameters, this function clips the values in the
    image to within a specific range and subsequently scales the image to a
    new range.

    :type im: ``numpy.ndarray``
    :param im: The original image.
    :type lower_in: ``NumericType``
    :param lower_in: The lower clipping bound for the input image. All
        values less than this will be set to this in the image before
        scaling. By default, this value will be set to 1st percentile of the
        image.
    :type upper_in: ``NumericType``
    :param upper_in: The upper clipping bound for the input image. All
        values greater than this will be set to this in the image before
        scaling. By default, this value will be set to 99th percentile of the
        image.
    :type lower_out: ``NumericType``
    :param lower_out: The lower bound for the range of the scaled output
        image. By default, this value is the minimum value of the dtype for
        integer dtypes or 0.0 for float dtypes.
    :type upper_out: ``NumericType``
    :param upper_out: The upper bound for the range of the scaled output
        image. By default, this value is the maximum value of the dtype for
        integer dtypes or 1.0 for float dtypes.
    :type keep_dtype: ``bool``
    :param keep_dtype: (default=True) Whether or not to cast the output
        image back to the dtype of the input image
    :rtype: ``numpy.ndarray``
    :return: The adjusted image.

    Examples:

    >>> import numpy as np
    >>> im = np.array([[  0,  32,  64],
    ...                [ 96, 128, 160],
    ...                [192, 224, 255]], dtype=np.uint8)
    >>> image_adjust(im, 64, 192)
    array([[  0,   0,   0],
           [ 63, 127, 191],
           [255, 255, 255]], dtype=uint8)
    >>> image_adjust(im, 64, 192, 64, 192)
    array([[ 64,  64,  64],
           [ 96, 128, 160],
           [192, 192, 192]], dtype=uint8)

    """
    sorted_im_ravel = None
    if keep_dtype:
        orig_dtype = im.dtype
    if lower_in is None:
        # Set to the 1st percentile
        sorted_im_ravel = np.sort(im.ravel())
        lower_in = sorted_im_ravel[im.size // 100]
    if upper_in is None:
        # Set to the 99th percentile
        if sorted_im_ravel is not None:
            upper_in = sorted_im_ravel[im.size - (im.size // 100) - 1]
        else:
            upper_in = np.sort(im.ravel())[im.size - (im.size // 100) - 1]
    if lower_out is None:
        if im.dtype.kind in 'iu':
            lower_out = np.iinfo(im.dtype).min
        elif im.dtype.kind == 'f':
            lower_out = 0.0
        else:
            # Should never occur
            lower_out = 0.0
    if upper_out is None:
        if im.dtype.kind in 'iu':
            upper_out = np.iinfo(im.dtype).max
        elif im.dtype.kind == 'f':
            upper_out = 1.0
        else:
            # Should never occur
            upper_out = 1.0
    im[im < lower_in] = lower_in
    im[im > upper_in] = upper_in
    im = ((im - lower_in).astype(float) * (upper_out - lower_out) /
          (upper_in - lower_in)) + lower_out
    if keep_dtype:
        im = im.astype(orig_dtype)
    return im


def image_translate(
        im: np.ndarray,
        dist_vec: _utils.ShapeType,
        pad_value: _utils.NumericType=0,
        ) -> np.ndarray:
    """Translates an image

    Given an image and a two-dimensional vector of distances to translate,
    this function translates the original image. Values left in the wake of
    the translation are padded with pad_value, which is by default set to 0.

    :type im: ``numpy.ndarray``
    :param im: The original image.
    :type dist_vec: ``ShapeType``
    :param dist_vec: A vector of distances to translate.
    :type pad_value: ``NumericType``
    :param pad_value: The value to pad for elements left in the wake of the
        translation.
    :rtype: ``numpy.ndarray``
    :return: The translated image.

    Examples:

    >>> import numpy as np
    >>> im = np.array(array([[ 1,  2,  3,  4,  5],
    ...                      [ 6,  7,  8,  9, 10],
    ...                      [11, 12, 13, 14, 15],
    ...                      [16, 17, 18, 19, 20],
    ...                      [21, 22, 23, 24, 25]])
    >>> image_translate(im, (0, 2))
    array([[ 0,  0,  1,  2,  3],
           [ 0,  0,  6,  7,  8],
           [ 0,  0, 11, 12, 13],
           [ 0,  0, 16, 17, 18],
           [ 0,  0, 21, 22, 23]])
    >>> image_translate(im, (-3, 2), pad_value=-1)
    array([[-1, -1, 16, 17, 18],
           [-1, -1, 21, 22, 23],
           [-1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1]])

    """
    try:
        dist_vec_abs = [abs(dist) for dist in dist_vec]
    except TypeError:
        dist_vec = (dist_vec, dist_vec)
        dist_vec_abs = [abs(dist) for dist in dist_vec]
    dist_vec_abs_px = _utils.resolve_shape_arg(dist_vec_abs, im.shape,
                                               arg_name='dist_vec',
                                               allow_larger_than_max=False)
    im_out = np.ones_like(im) * pad_value
    if 0 == dist_vec_abs_px[0]:
        if 0 == dist_vec_abs_px[1]:
            im_out[:, :] = im[:, :]
        elif dist_vec[1] < 0:
            im_out[:, :-dist_vec_abs_px[1]] = im[:, dist_vec_abs_px[1]:]
        else:
            im_out[:, dist_vec_abs_px[1]:] = im[:, :-dist_vec_abs_px[1]]
    elif dist_vec[0] < 0:
        if 0 == dist_vec_abs_px[1]:
            im_out[:-dist_vec_abs_px[0], :] = im[dist_vec_abs_px[0]:, :]
        elif dist_vec[1] < 0:
            im_out[:-dist_vec_abs_px[0], :-dist_vec_abs_px[1]] = \
                    im[dist_vec_abs_px[0]:, dist_vec_abs_px[1]:]
        else:
            im_out[:-dist_vec_abs_px[0], dist_vec_abs_px[1]:] = \
                    im[dist_vec_abs_px[0]:, :-dist_vec_abs_px[1]]
    else:
        if 0 == dist_vec_abs_px[1]:
            im_out[dist_vec_abs_px[0]:, :] = im[:-dist_vec_abs_px[0], :]
        elif dist_vec[1] < 0:
            im_out[dist_vec_abs_px[0]:, :-dist_vec_abs_px[1]] = \
                    im[:-dist_vec_abs_px[0], dist_vec_abs_px[1]:]
        else:
            im_out[dist_vec_abs_px[0]:, dist_vec_abs_px[1]:] = \
                    im[:-dist_vec_abs_px[0], :-dist_vec_abs_px[1]]
    return im_out


def image_shift(
        im: np.ndarray,
        dist_vec: _utils.ShapeType,
        ) -> np.ndarray:
    """Shifts an image by consecutively applying and summing translations

    Given an image and a two-dimensional vector of distances to shift,
    this function shifts the image by weighting and summing many
    intermediate translations. This simulates an image taken with a shaky
    camera.

    :type im: ``numpy.ndarray``
    :param im: The original image.
    :type dist_vec: ``ShapeType``
    :param dist_vec: A vector of distances to shift.
    :rtype: ``numpy.ndarray``
    :return: The translated image.

    Examples:

    >>> import numpy as np
    >>> im = np.array(array([[  1,   2,   4],
    ...                      [  8,  16,  32],
    ...                      [ 64, 128, 256]])
    >>> image_translate(im, (0, 1))
    array([[  1,   1,   3],
           [  8,  12,  24],
           [ 64,  96, 192]])
    >>> image_translate(im, (-1, 1))
    array([[  1,   5,  10],
           [  8,  40,  80],
           [ 64, 128, 256]])

    """
    try:
        dist_vec_abs = [abs(dist) for dist in dist_vec]
    except TypeError:
        dist_vec = (dist_vec, dist_vec)
        dist_vec_abs = [abs(dist) for dist in dist_vec]
    dist_vec_abs_px = _utils.resolve_shape_arg(dist_vec_abs, im.shape,
                                               arg_name='dist_vec',
                                               allow_larger_than_max=False)
    im_out = im.copy().astype(float)
    im_ones = np.ones_like(im, dtype=np.uint16)
    weights = im_ones.copy()
    if 0 == dist_vec_abs_px[0]:
        if 0 == dist_vec_abs_px[1]:
            pass
        elif dist_vec[1] < 0:
            for j in range(1, dist_vec_abs_px[1] + 1):
                weights += image_translate(im_ones, (0, -j), pad_value=0)
                im_out += image_translate(im, (0, -j), pad_value=0)
        else:
            for j in range(1, dist_vec_abs_px[1] + 1):
                weights += image_translate(im_ones, (0, j), pad_value=0)
                im_out += image_translate(im, (0, j), pad_value=0)
    elif dist_vec[0] < 0:
        if 0 == dist_vec_abs_px[1]:
            for i in range(1, dist_vec_abs_px[0] + 1):
                weights += image_translate(im_ones, (-i, 0), pad_value=0)
                im_out += image_translate(im, (-i, 0), pad_value=0)
        elif dist_vec[1] < 0:
            iterations = min(dist_vec_abs_px)
            for k in range(1, iterations + 1):
                i = (k * dist_vec_abs_px[0]) // iterations
                j = (k * dist_vec_abs_px[1]) // iterations
                weights += image_translate(im_ones, (-i, -j), pad_value=0)
                im_out += image_translate(im, (-i, -j), pad_value=0)
        else:
            iterations = min(dist_vec_abs_px)
            for k in range(1, iterations + 1):
                i = (k * dist_vec_abs_px[0]) // iterations
                j = (k * dist_vec_abs_px[1]) // iterations
                weights += image_translate(im_ones, (-i, j), pad_value=0)
                im_out += image_translate(im, (-i, j), pad_value=0)
    else:
        if 0 == dist_vec_abs_px[1]:
            for i in range(1, dist_vec_abs_px[0] + 1):
                weights += image_translate(im_ones, (i, 0), pad_value=0)
                im_out += image_translate(im, (i, 0), pad_value=0)
        elif dist_vec[1] < 0:
            iterations = min(dist_vec_abs_px)
            for k in range(1, iterations + 1):
                i = (k * dist_vec_abs_px[0]) // iterations
                j = (k * dist_vec_abs_px[1]) // iterations
                weights += image_translate(im_ones, (i, -j), pad_value=0)
                im_out += image_translate(im, (i, -j), pad_value=0)
        else:
            iterations = min(dist_vec_abs_px)
            for k in range(1, iterations + 1):
                i = (k * dist_vec_abs_px[0]) // iterations
                j = (k * dist_vec_abs_px[1]) // iterations
                weights += image_translate(im_ones, (i, j), pad_value=0)
                im_out += image_translate(im, (i, j), pad_value=0)
    im_out /= weights
    return im_out.astype(im.dtype)

