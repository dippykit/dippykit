"""Module of various image transform functions

This module contains an assortment of functions that perform transform
operations on images in various useful manners.

"""

# This library was developed for the Georgia Tech graduate course ECE 6258:
# Digital Image Processing with Professor Ghassan AlRegib.
# For comments and feedback, please email dippykit[at]gmail.com

# Functional imports
import numpy as np
from scipy import fftpack
from skimage.feature import canny
from skimage.filters import sobel, sobel_h, sobel_v, scharr, prewitt, \
    roberts, laplace

__author__ = 'Brighton Ancelin, Motaz Alfarraj, Ghassan AlRegib'

__all__ = ['dct_2d', 'idct_2d', 'edge_detect']


def dct_2d(
        im: np.ndarray,
        ) -> np.ndarray:
    """Computes the Discrete Cosine Transform of an image

    Given an input image, this function computes its discrete cosine
    transform. This result is identical to the result of the dct2() Matlab
    function. Internally, this function is computed using
    ``scipy.fftpack.dctn(im, norm='ortho')``.

    This function is essentially a wrapper for `scipy.fftpack.dctn`_,
    so more detailed documentation may be found there.

    :type im: ``numpy.ndarray``
    :param im: An image to be processed.
    :rtype: ``numpy.ndarray``
    :return: The discrete cosine transform of the input image.

    .. note::
        This function wraps around functions from other packages. Reading
        these functions' documentations may be useful. See the **See also**
        section for more information.

    .. seealso::
        `scipy.fftpack.dctn`_
            Documentation of the dctn function from Scipy

    .. _scipy.fftpack.dctn: https://docs.scipy.org/doc/scipy/reference
        /generated/scipy.fftpack.dctn.html

    Examples:

    >>> import numpy as np
    >>> freq_1 = 1/4
    >>> freq_2 = 1/8
    >>> domain_1, domain_2 = np.meshgrid(np.arange(8), np.arange(8))
    >>> domain_1
    array([[0, 1, 2, 3, 4, 5, 6, 7],
           [0, 1, 2, 3, 4, 5, 6, 7],
           [0, 1, 2, 3, 4, 5, 6, 7],
           [0, 1, 2, 3, 4, 5, 6, 7],
           [0, 1, 2, 3, 4, 5, 6, 7],
           [0, 1, 2, 3, 4, 5, 6, 7],
           [0, 1, 2, 3, 4, 5, 6, 7],
           [0, 1, 2, 3, 4, 5, 6, 7]])
    >>> domain_2
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4, 4, 4, 4],
           [5, 5, 5, 5, 5, 5, 5, 5],
           [6, 6, 6, 6, 6, 6, 6, 6],
           [7, 7, 7, 7, 7, 7, 7, 7]])
    >>> im_1 = 10*np.cos(2 * np.pi * freq_1 * (domain_1 + 0.5))
    >>> im_2 = 10*np.cos(2 * np.pi * freq_2 * (domain_2 + 0.5))
    >>> im = im_1 + im_2
    >>> np.set_printoptions(precision=2, suppress=True)
    >>> im
    array([[ 16.31,   2.17,   2.17,  16.31,  16.31,   2.17,   2.17,  16.31],
           [ 10.9 ,  -3.24,  -3.24,  10.9 ,  10.9 ,  -3.24,  -3.24,  10.9 ],
           [  3.24, -10.9 , -10.9 ,   3.24,   3.24, -10.9 , -10.9 ,   3.24],
           [ -2.17, -16.31, -16.31,  -2.17,  -2.17, -16.31, -16.31,  -2.17],
           [ -2.17, -16.31, -16.31,  -2.17,  -2.17, -16.31, -16.31,  -2.17],
           [  3.24, -10.9 , -10.9 ,   3.24,   3.24, -10.9 , -10.9 ,   3.24],
           [ 10.9 ,  -3.24,  -3.24,  10.9 ,  10.9 ,  -3.24,  -3.24,  10.9 ],
           [ 16.31,   2.17,   2.17,  16.31,  16.31,   2.17,   2.17,  16.31]])
    >>> dct_2d(im)
    array([[-0.  ,  0.  , -0.  ,  0.  , 56.57,  0.  , -0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  , -0.  , -0.  ,  0.  , -0.  , -0.  ],
           [56.57,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [-0.  ,  0.  , -0.  , -0.  ,  0.  ,  0.  ,  0.  , -0.  ],
           [-0.  ,  0.  , -0.  ,  0.  , -0.  ,  0.  , -0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  , -0.  ,  0.  ,  0.  , -0.  , -0.  ],
           [-0.  ,  0.  , -0.  ,  0.  ,  0.  ,  0.  , -0.  , -0.  ],
           [ 0.  ,  0.  ,  0.  , -0.  , -0.  ,  0.  , -0.  , -0.  ]])

    """
    return fftpack.dctn(im, norm='ortho')


def idct_2d(
        im: np.ndarray,
        ) -> np.ndarray:
    """Computes the Inverse Discrete Cosine Transform of an image

    Given an input image, this function computes its inverse discrete cosine
    transform. This result is identical to the result of the idct2() Matlab
    function. Internally, this function is computed using
    ``scipy.fftpack.idctn(im, norm='ortho')``.

    This function is essentially a wrapper for `scipy.fftpack.idctn`_,
    so more detailed documentation may be found there.

    :type im: ``numpy.ndarray``
    :param im: An image to be processed.
    :rtype: ``numpy.ndarray``
    :return: The inverse discrete cosine transform of the input image.

    .. note::
        This function wraps around functions from other packages. Reading
        these functions' documentations may be useful. See the **See also**
        section for more information.

    .. seealso::
        `scipy.fftpack.idctn`_
            Documentation of the dctn function from Scipy

    .. _scipy.fftpack.idctn: https://docs.scipy.org/doc/scipy/reference
        /generated/scipy.fftpack.idctn.html

    Examples:

    >>> import numpy as np
    >>> im_DCT = np.zeros((8, 8))
    >>> im_DCT[0, 2] = 1
    >>> np.set_printoptions(precision=2, suppress=True)
    >>> idct_2d(im_DCT)
    array([[ 0.16,  0.07, -0.07, -0.16, -0.16, -0.07,  0.07,  0.16],
           [ 0.16,  0.07, -0.07, -0.16, -0.16, -0.07,  0.07,  0.16],
           [ 0.16,  0.07, -0.07, -0.16, -0.16, -0.07,  0.07,  0.16],
           [ 0.16,  0.07, -0.07, -0.16, -0.16, -0.07,  0.07,  0.16],
           [ 0.16,  0.07, -0.07, -0.16, -0.16, -0.07,  0.07,  0.16],
           [ 0.16,  0.07, -0.07, -0.16, -0.16, -0.07,  0.07,  0.16],
           [ 0.16,  0.07, -0.07, -0.16, -0.16, -0.07,  0.07,  0.16],
           [ 0.16,  0.07, -0.07, -0.16, -0.16, -0.07,  0.07,  0.16]])

    """
    return fftpack.idctn(im, norm='ortho')


def edge_detect(
        im: np.ndarray,
        mode: str='sobel',
        as_bool: bool=False,
        **kwargs
        ) -> np.ndarray:
    """Performs an edge detection operation on an image

    Given an input image and an edge detection mode, this function applies
    an edge detection operation to the image and returns the result.
    Optionally, this result can be returned as a binary/boolean image.

    This function is essentially a wrapper for several external functions,
    so more detailed documentation may be found there.

    :type im: ``numpy.ndarray``
    :param im: The input image.
    :type mode: ``str``
    :param mode: The edge detection mode/algorithm to be used. Acceptable
        values are: 'sobel', 'sobel_h', 'sobel_v', 'canny', 'scharr',
        'prewitt', 'roberts', and 'laplace'.
    :type as_bool: ``bool``
    :param as_bool: (default=False) If set to True, this function will
        return the result as a binary/boolean image. This binary image is
        built by taking the original edge image and substituting any
        values below *threshold* as False, and all other values as True. The
        value of *threshold* is defined as the 95th percentile of values in
        the original edge image.
    :param kwargs: The keyword arguments to be passed to the edge detecting
        functions. If mode='canny', and there is no keyword argument
        'sigma', then this function will set the 'sigma' keyword argument to
        the square root of 2 by default.
    :rtype: ``numpy.ndarray``
    :return: The edge image.

    .. note::
        This function wraps around functions from other packages. Reading
        these functions' documentations may be useful. See the **See also**
        section for more information.

    .. seealso::
        `skimage.feature.canny`_
            Documentation of the canny function from Scikit Image
        `skimage.filters.sobel`_
            Documentation of the sobel function from Scikit Image
        `skimage.filters.sobel_h`_
            Documentation of the sobel_h function from Scikit Image
        `skimage.filters.sobel_v`_
            Documentation of the sobel_v function from Scikit Image
        `skimage.filters.scharr`_
            Documentation of the scharr function from Scikit Image
        `skimage.filters.prewitt`_
            Documentation of the prewitt function from Scikit Image
        `skimage.filters.roberts`_
            Documentation of the roberts function from Scikit Image
        `skimage.filters.laplace`_
            Documentation of the laplace function from Scikit Image

    .. _skimage.feature.canny: http://scikit-image.org/docs/dev/api
        /skimage.feature.html#skimage.feature.canny
    .. _skimage.filters.sobel: http://scikit-image.org/docs/dev/api
        /skimage.filters.html#skimage.filters.sobel
    .. _skimage.filters.sobel_h: http://scikit-image.org/docs/dev/api
        /skimage.filters.html#skimage.filters.sobel_h
    .. _skimage.filters.sobel_v: http://scikit-image.org/docs/dev/api
        /skimage.filters.html#skimage.filters.sobel_v
    .. _skimage.filters.scharr: http://scikit-image.org/docs/dev/api
        /skimage.filters.html#skimage.filters.scharr
    .. _skimage.filters.prewitt: http://scikit-image.org/docs/dev/api
        /skimage.filters.html#skimage.filters.prewitt
    .. _skimage.filters.roberts: http://scikit-image.org/docs/dev/api
        /skimage.filters.html#skimage.filters.roberts
    .. _skimage.filters.laplace: http://scikit-image.org/docs/dev/api
        /skimage.filters.html#skimage.filters.laplace

    Examples:

    >>> import numpy as np
    >>> im = np.array(array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
    ...                      [0. , 0. , 0. , 0.5, 0.5, 0. , 0. , 0. ],
    ...                      [0. , 0. , 1. , 1. , 1. , 1. , 0. , 0. ],
    ...                      [0. , 0.5, 1. , 1. , 1. , 1. , 0. , 0. ],
    ...                      [0. , 0.5, 1. , 1. , 1. , 1. , 0. , 0. ],
    ...                      [0. , 0. , 1. , 1. , 1. , 1. , 0. , 0. ],
    ...                      [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
    ...                      [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ]])
    >>> np.round(edge_detect(im), 2)
    array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.25, 0.64, 0.73, 0.73, 0.64, 0.25, 0.  ],
           [0.  , 0.64, 0.75, 0.45, 0.45, 0.76, 0.56, 0.  ],
           [0.  , 0.73, 0.45, 0.  , 0.  , 0.71, 0.71, 0.  ],
           [0.  , 0.73, 0.45, 0.  , 0.  , 0.71, 0.71, 0.  ],
           [0.  , 0.64, 0.76, 0.71, 0.71, 0.75, 0.56, 0.  ],
           [0.  , 0.25, 0.56, 0.71, 0.71, 0.56, 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]])
    >>> edge_detect(im, as_bool=True)
    array([[False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False],
           [False, False,  True, False, False,  True, False, False],
           [False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False],
           [False, False,  True, False, False,  True, False, False],
           [False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False]])
    >>> np.round(dip.edge_detect(im, 'sobel_h'), 2)
    array([[ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.25,  0.75,  1.  ,  1.  ,  0.75,  0.25,  0.  ],
           [ 0.  ,  0.5 ,  0.75,  0.62,  0.62,  0.62,  0.25,  0.  ],
           [ 0.  ,  0.25,  0.12,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  , -0.25, -0.12,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  , -0.5 , -0.88, -1.  , -1.  , -0.75, -0.25,  0.  ],
           [ 0.  , -0.25, -0.75, -1.  , -1.  , -0.75, -0.25,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]])

    """
    if 'sobel' == mode:
        edges = sobel(im, **kwargs)
    elif 'sobel_h' == mode:
        edges = sobel_h(im, **kwargs)
    elif 'sobel_v' == mode:
        edges = sobel_v(im, **kwargs)
    elif 'canny' == mode:
        if 'sigma' not in kwargs:
            kwargs['sigma'] = np.sqrt(2)
        edges = canny(im, **kwargs)
    elif 'scharr' == mode:
        edges = scharr(im, **kwargs)
    elif 'prewitt' == mode:
        edges = prewitt(im, **kwargs)
    elif 'roberts' == mode:
        edges = roberts(im, **kwargs)
    elif 'laplace' == mode:
        edges = laplace(im, **kwargs)
    else:
        raise ValueError("Unrecognized mode '{}'".format(mode))
    if as_bool:
        if edges.dtype.kind == 'b':
            return edges
        else:
            sorted_edges_ravel = np.sort(edges.ravel())
            # Set threshold to 95th percentile
            threshold = sorted_edges_ravel[(edges.size * 19) // 20]
            return np.where(edges < threshold, False, True)
    return edges

