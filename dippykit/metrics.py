"""Module of various metrics-based functions

This module contains an assortment of functions that provide insight into
image and signal data. They provide a means for high-level analysis.

"""

# This library was developed for the Georgia Tech graduate course ECE 6258:
# Digital Image Processing with Professor Ghassan AlRegib.
# For comments and feedback, please email dippykit[at]gmail.com

# Internal imports
from . import _utils
from . import utilities

# Functional imports
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from skimage.measure import structural_similarity

# General imports
from typing import Dict, Tuple, Any

__author__ = 'Brighton Ancelin, Motaz Alfarraj, Ghassan AlRegib'


__all__ = ['PSNR', 'contrast', 'MSE', 'energy', 'MAD', 'MADev', 'entropy',
           'SSIM', 'SSIM_luminance', 'SSIM_contrast', 'SSIM_structure']


def PSNR(
        im_1: np.ndarray,
        im_2: np.ndarray,
        max_signal_value: _utils.NumericType=None,
        ) -> float:
    """Calculates the **peak signal-to-noise ratio** between two images in dB

    Calculates the peak signal-to-noise ratio in decibels between two input
    images. The ratio is defined as -20 * log\ :subscript:`10`\ 
    (max_signal_value / X) where max_signal_value is the maximum possible 
    value in either image and X is the mean of all elements in the ndarray 
    (im_1 - im_2)\ :superscript:`2`\ . The two images must have the same 
    dtype.
    
    If max_signal_value is not specified and the two images have unsigned 
    integer dtype, then max_signal_value is assumed to be the largest value 
    possible in the dtype. If max_signal_value is not specified and the two 
    images have float dtype and all their values are between 0.0 and 1.0 
    inclusive, then max_signal_value is assumed to be 1.0.

    :type im_1: ``numpy.ndarray``
    :param im_1: An image to be compared.
    :type im_2: ``numpy.ndarray``
    :param im_2: An image to be compared.
    :type max_signal_value: ``int`` or ``float``
    :param max_signal_value: (Optional) The maximum *possible* signal value in 
        either image. (This is not necessarily the maximum value of either 
        image).
    :rtype: ``float``
    :return: The decibel peak signal-to-noise ratio.

    Examples:

    >>> import numpy as np
    >>> im = np.array([[0.5  , 0.125],
    ...                [0.25 , 0.   ]])
    >>> im_2 = im / 2
    >>> PSNR(im, im_2)
    16.880806619058927
    >>> im = np.array([[100,  20],
    ...                [ 40,   0]])
    >>> im_2 = im / 2
    >>> PSNR(im, im_2, 255)
    19.380190974762105
    """
    assert im_1.dtype == im_2.dtype, "The two images must have the same dtype"
    if max_signal_value is None:
        float_cond = 'f' == im_1.dtype.kind and \
                     1.0 >= max(np.max(im_1), np.max(im_2)) and \
                     0.0 <= min(np.min(im_1), np.min(im_2))
        assert float_cond or 'u' == im_1.dtype.kind, \
                "Without a specified max_signal_value, the PSNR can only be " \
                "calculated for images with unsigned integer dtypes or " \
                "float dtypes with values normalized between 0.0 and 1.0. " \
                "Please either specify a max_signal_value or convert the " \
                "images to an unsigned integer dtype."
        if float_cond:
            max_signal_value = 1.0
        else:
            max_signal_value = np.iinfo(im_1.dtype).max
    error_diff = im_1 - im_2
    if np.all(0 == error_diff):
        return float('inf')
    return 10 * np.log10(max_signal_value ** 2 / np.mean(error_diff ** 2))


def contrast(
        im: np.ndarray,
        ) -> float:
    """Calculates the **contrast value** of an image

    Calculates the contrast value of an image. The contrast value of the
    image is defined as the difference between the maximum pixel value and
    the minimum pixel value divided by the sum of the same values.
    Therefore, the return value of this function is a float between 0.0 and
    1.0.

    :type im: ``numpy.ndarray``
    :param im: An image to be examined.
    :rtype: ``float``
    :return: The contrast value of the image (float between 0.0 and 1.0).

    Examples:

    >>> import numpy as np
    >>> im_1 = np.array([[0.5  , 0.125],
    ...                  [0.25 , 0.   ]])
    >>> contrast(im_1)
    1.0
    >>> im_2 = np.array([[125, 125],
    ...                  [125, 125]])
    >>> contrast(im_2)
    0.0

    """
    max_pixel = np.max(im)
    min_pixel = np.min(im)
    return (max_pixel - min_pixel) / (max_pixel + min_pixel)


def MSE(
        im: np.ndarray,
        im_ref: np.ndarray=None,
        ) -> float:
    """Calculates the **mean squared error** of an image or images

    * **For one image argument:**
        Calculates the mean squared error of an error image. The mean squared
        error is defined as the mean of all squared elements in the error
        image.

    * **For two image arguments:**
        Calculates the mean squared error between two images. The mean squared
        error is defined as the mean of all squared elements in the ndarray
        (im - im_ref).

    :type im: ``numpy.ndarray``
    :param im: An image to be examined.
    :type im_ref: ``numpy.ndarray``
    :param im_ref: (Optional) A reference image for im to be compared against.
    :rtype: ``float``
    :return: The mean squared error of the image(s).

    Examples:

    >>> import numpy as np
    >>> im_1 = np.array([[0.5  , 0.125],
    ...                  [0.25 , 0.   ]])
    >>> MSE(im_1)
    0.08203125
    >>> im_2 = np.array([[0.   , 0.125],
    ...                  [0.25 , 0.   ]])
    >>> MSE(im_1, im_2)
    0.0625

    """
    if im_ref is not None:
        return np.mean((im - im_ref) ** 2)
    return np.mean(im ** 2)


def energy(
        im: np.ndarray,
        ) -> float:
    """Calculates the **energy** of an image

    Calculates the energy of an image. The energy is defined as the sum of
    all squared elements in the image.

    :type im: ``numpy.ndarray``
    :param im: An image to be examined.
    :rtype: ``float``
    :return: The energy of the image.

    Examples:

    >>> import numpy as np
    >>> im_1 = np.array([[0.5  , 0.125],
    ...                  [0.25 , 0.   ]])
    >>> energy(im_1)
    0.328125
    >>> im_2 = np.array([[125, 125],
    ...                  [125, 125]])
    >>> energy(im_2)
    62500

    """
    return np.sum(im ** 2)


def MAD(
        im: np.ndarray,
        im_ref: np.ndarray=None,
        ) -> float:
    """Calculates the **mean absolute difference** of an image or images

    * **For one image argument:**
        Calculates the mean absolute difference of an error image. The mean
        absolute difference is defined as the mean of the absolute value of
        all elements in the error image.

    * **For two image arguments:**
        Calculates the mean absolute difference between two images. The mean
        absolute difference is defined as the mean of the absolute value of
        all elements in the ndarray (im - im_ref).

    :type im: ``numpy.ndarray``
    :param im: An image to be examined.
    :type im_ref: ``numpy.ndarray``
    :param im_ref: (Optional) A reference image for im to be compared against.
    :rtype: ``float``
    :return: The mean absolute difference of the image(s).

    Examples:

    >>> import numpy as np
    >>> im_1 = np.array([[0.5  , 0.125],
    ...                  [0.25 , 0.   ]])
    >>> MAD(im_1)
    0.21875
    >>> im_2 = np.array([[0.   , 0.125],
    ...                  [0.25 , 0.   ]])
    >>> MAD(im_1, im_2)
    0.125

    """
    assert all(x == y for x, y in zip(im.shape, im_ref.shape)), \
        "MAD calculations require that images be the same shape"
    if im_ref is not None:
        if 'u' == im.dtype.kind:
            im = im.astype(np.int32)
        if 'u' == im_ref.dtype.kind:
            im_ref = im_ref.astype(np.int32)
        return np.mean(np.abs(im - im_ref))
    return np.mean(np.abs(im))


def MADev(
        im: np.ndarray,
        ) -> float:
    """Calculates the **mean absolute deviation** of an image

    Calculates the mean absolute deviation of an image. The mean absolute
    deviation is defined as the mean of the absolute value of all
    differences between elements in the image and the mean value of all
    elements in the image.

    :type im: ``numpy.ndarray``
    :param im: An image to be examined.
    :rtype: ``float``
    :return: The mean absolute deviation of the image.

    Examples:

    >>> import numpy as np
    >>> im_1 = np.array([[15, 15],
    ...                  [15, 15]])
    >>> MADev(im_1)
    0.0
    >>> im_2 = np.array([[15, 20],
    ...                  [10, 15]])
    >>> MADev(im_2)
    2.5

    """
    return np.mean(np.abs(im - np.mean(im)))


def entropy(
        im: np.ndarray,
        ) -> float:
    """Calculates the **entropy** of an image

    Calculates the entropy of an image. The entropy is equivalent to the
    expected value of the number of bits used to encode a single pixel in
    the optimal encoding case. It is a lower bound on the amount of
    information required to represent the image (per pixel). The entropy is
    defined by the sum of all pixel value probabilities times the
    logarithm-base-2 of the inverse pixel probability.

    This function requires that the input image have either an integer or
    unsigned integer dtype.

    :type im: ``numpy.ndarray``
    :param im: An image to be examined.
    :rtype: ``float``
    :return: The entropy of the image.

    Examples:

    >>> import numpy as np
    >>> im_1 = np.array([[15, 15],
    ...                  [15, 15]])
    >>> entropy(im_1)
    -0.0
    >>> im_2 = np.array([[15, 20],
    ...                  [10, 15]])
    >>> entropy(im_2)
    1.5

    """
    assert 'i' == im.dtype.kind or 'u' == im.dtype.kind, \
            "Image dtype must be integer for entropy to be calculated"
    bins = np.bincount(im.reshape(-1))
    bin_probs = bins[0 != bins]
    bin_probs = bin_probs / np.sum(bin_probs)
    return -np.sum(bin_probs * np.log2(bin_probs))


def _SSIM_preprocess(
        im_1: np.ndarray,
        im_2: np.ndarray,
        K1: float=0.01,
        K2: float=0.03,
        use_gaussian_window: bool=None,
        window_size: int=None,
        data_range: float=None,
        sigma: float=None,
        auto_downsample: bool=None,
        use_sample_covariance: bool=None,
        like_matlab: bool=False,
        **kwargs
        ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if window_size is not None:
        kwargs['win_size'] = window_size
    if data_range is not None:
        kwargs['data_range'] = data_range
    if sigma is not None:
        kwargs['sigma'] = sigma
    kwargs['full'] = True
    kwargs['K1'] = K1
    kwargs['K2'] = K2
    if use_gaussian_window is not None:
        kwargs['gaussian_weights'] = use_gaussian_window
    elif not like_matlab:
        kwargs['gaussian_weights'] = False
    if use_sample_covariance is not None:
        kwargs['use_sample_covariance'] = use_sample_covariance
    elif not like_matlab:
        kwargs['use_sample_covariance'] = False
    if like_matlab:
        if 'win_size' not in kwargs:
            kwargs['win_size'] = 11
        if 'data_range' not in kwargs:
            # Be careful; the Matlab-like implementation always uses
            # data_range=255 regardless of dtype. This may not be what you
            # want!
            kwargs['data_range'] = 255
        if 'sigma' not in kwargs:
            kwargs['sigma'] = 1.5
        if 'full' not in kwargs:
            kwargs['full'] = True
        if 'K1' not in kwargs:
            kwargs['K1'] = 0.01
        if 'K2' not in kwargs:
            kwargs['K2'] = 0.03
        if 'gaussian_weights' not in kwargs:
            kwargs['gaussian_weights'] = True
        if 'use_sample_covariance' not in kwargs:
            kwargs['use_sample_covariance'] = False
        if auto_downsample is None:
            auto_downsample = False
    if auto_downsample is not None and auto_downsample:
        scale_factor = round((min(im_1.shape[0], im_1.shape[1]) / 256) + 1e-9)
        if 1 < scale_factor:
            low_pass_filter = np.ones((scale_factor, scale_factor))
            low_pass_filter /= scale_factor ** 2
            im_1 = utilities.convolve2d(im_1, low_pass_filter, mode='same',
                    boundary='symmetric', like_matlab=True)
            im_1 = im_1[::scale_factor, ::scale_factor]
            im_2 = utilities.convolve2d(im_2, low_pass_filter, mode='same',
                    boundary='symmetric', like_matlab=True)
            im_2 = im_2[::scale_factor, ::scale_factor]
    return im_1, im_2, kwargs


def _SSIM_like_matlab_crop(
        ssim_map: np.ndarray,
        **kwargs
        ) -> np.ndarray:
    pad = kwargs['win_size'] // 2
    return ssim_map[pad:(ssim_map.shape[0] - pad),
                    pad:(ssim_map.shape[1] - pad)]


def SSIM(
        im_1: np.ndarray,
        im_2: np.ndarray,
        K1: float=0.01,
        K2: float=0.03,
        use_gaussian_window: bool=None,
        window_size: int=None,
        data_range: float=None,
        sigma: float=None,
        auto_downsample: bool=None,
        use_sample_covariance: bool=None,
        like_matlab: bool=False,
        **kwargs
        ) -> Tuple[float, np.ndarray]:
    """Returns the mean SSIM index and SSIM image for a comparison of two 
    images
    
    Given two images, this function will return the mean SSIM index and the 
    full SSIM image.

    This function is essentially a wrapper for
    `skimage.measure.structural_similarity`_, so more detailed documentation may be
    found there.
    
    :type im_1: ``numpy.ndarray``
    :param im_1: The first image in the comparison
    :type im_2: ``numpy.ndarray``
    :param im_2: The second image in the comparison
    :type K1: ``float``
    :param K1: (default=0.01) The K1 parameter used in the SSIM calculation
    :type K2: ``float``
    :param K2: (default=0.03) The K2 parameter used in the SSIM calculation
    :type use_gaussian_window: ``bool``
    :param use_gaussian_window: (default=False) If set to True, 
        this function will use a gaussian window for the SSIM calculation 
    :type window_size: ``int``
    :param window_size: (Optional) The size of the window to be used in the 
        SSIM calculation. If **use_gaussian_window** is set to True, 
        then this argument is ignored. 
    :type data_range: ``float``
    :param data_range: (Optional) The range of values that the image can span. 
        If this parameter is not set, then the range will be determined 
        from the minimum and maximum values of the images. For uint8 
        images, this value should be 255.
    :type sigma: ``float``
    :param sigma: (Optional) If **use_gaussian_window** is set to True, 
        this parameter determines the sigma used for the gaussian window.
    :type auto_downsample: ``bool``
    :param auto_downsample: (default=True) If set to True, this function will
        automatically downsample the inputs. The downsampling consists of
        finding the smallest of the images' first two dimensions (number of
        rows and number of columns), dividing this value by 256, rounding
        the output, and setting this final value as the scaling factor (sf). A
        square (sf x sf) lowpass averaging filter is then applied to both
        images with mode 'same'. Finally, the two images are downsampled by
        sf in each dimension.
    :type use_sample_covariance: ``bool``
    :param use_sample_covariance: (default=False) If set to True,
        this function will use sample covariances in its calculations.
        Otherwise, covariances of 1 will be used.
    :type like_matlab: ``bool``
    :param like_matlab: If set to True, this function will act like the
        Matlab function ssim_index. This is merely a specific configuration
        of the above parameters.
    :param kwargs: For a full list of keyword arguments, see
        `skimage.measure.structural_similarity`_.
    :rtype: ``(float, numpy.ndarray)``
    :return: A tuple of two elements. The first element is the mean SSIM 
        index. The second element is the full SSIM image.

    .. note::
        This function wraps around functions from other packages. Reading
        these functions' documentations may be useful. See the **See also**
        section for more information.

    .. seealso::
        `skimage.measure.structural_similarity`_
            Documentation of the structural_similarity function from Scikit Image
        `SSIM Wikipedia Page`_
            Wikipedia page describing the formulae for SSIM calculations

    .. _skimage.measure.structural_similarity: http://scikit-image.org/docs/dev/api/
        skimage.measure.html#skimage.measure.structural_similarity

    .. _SSIM Wikipedia Page: https://en.wikipedia.org/wiki/
        Structural_similarity#Formula_components

    """
    im_1, im_2, kwargs = _SSIM_preprocess(im_1=im_1, im_2=im_2, K1=K1,
            K2=K2, use_gaussian_window=use_gaussian_window,
            window_size=window_size, data_range=data_range, sigma=sigma,
            auto_downsample=auto_downsample,
            use_sample_covariance=use_sample_covariance,
            like_matlab=like_matlab, **kwargs)
    if not like_matlab:
        return structural_similarity(im_1, im_2, **kwargs)
    else:
        mssim, ssim_map = structural_similarity(im_1, im_2, **kwargs)
        return mssim, _SSIM_like_matlab_crop(ssim_map, **kwargs)


def _compare_ssim_with_callback(
        X,
        Y,
        callback_func,
        win_size=None,
        gradient=False,
        data_range=None,
        multichannel=False,
        gaussian_weights=False,
        full=False,
        **kwargs
        ):
    """
    The following code was copied and modified from
    skimage.measure.structural_similarity. The modifications were adding a callback
    feature to the algorithm. This allows for targeted SSIM calculations.
    """
    if not X.dtype == Y.dtype:
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if multichannel:
        # loop over channels
        args = dict(win_size=win_size,
                    gradient=gradient,
                    data_range=data_range,
                    multichannel=False,
                    gaussian_weights=gaussian_weights,
                    full=full)
        args.update(kwargs)
        nch = X.shape[-1]
        mssim = np.empty(nch)
        if gradient:
            G = np.empty(X.shape)
        if full:
            S = np.empty(X.shape)
        for ch in range(nch):
            ch_result = _compare_ssim_with_callback(X[..., ch], Y[..., ch],
                    callback_func=callback_func, **args)
            if gradient and full:
                mssim[..., ch], G[..., ch], S[..., ch] = ch_result
            elif gradient:
                mssim[..., ch], G[..., ch] = ch_result
            elif full:
                mssim[..., ch], S[..., ch] = ch_result
            else:
                mssim[..., ch] = ch_result
        mssim = mssim.mean()
        if gradient and full:
            return mssim, G, S
        elif gradient:
            return mssim, G
        elif full:
            return mssim, S
        else:
            return mssim

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if win_size is None:
        if gaussian_weights:
            win_size = 11  # 11 to match Wang et. al. 2004
        else:
            win_size = 7   # backwards compatibility

    if np.any((np.asarray(X.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent.  If the input is a multichannel "
            "(color) image, set multichannel=True.")

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        if X.dtype.type is np.bool_:
            dmin, dmax = False, True
        elif X.dtype.kind in 'iu':
            dinfo = np.iinfo(X.dtype.type)
            dmin = dinfo.min
            dmax = dinfo.max
        elif X.dtype.kind == 'f':
            dmin, dmax = -1, 1
        else:
            raise ValueError('dtype not recognized and data_range is None is '
                             'not allowed')
        data_range = dmax - dmin

    ndim = X.ndim

    if gaussian_weights:
        # sigma = 1.5 to approximately match filter in Wang et. al. 2004
        # this ends up giving a 13-tap rather than 11-tap Gaussian
        filter_func = gaussian_filter
        filter_args = {'sigma': sigma}

    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(X * X, **filter_args)
    uyy = filter_func(Y * Y, **filter_args)
    uxy = filter_func(X * Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    # Have to include the non-negative assertion (rounding errors) because
    # some metrics may want to use standard deviation (sqrt of variance),
    # which would otherwise be an issue
    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vxy[vxy < 0] = 0

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2

    S = callback_func(ux, uy, uxx, uyy, uxy, vx, vy, vxy, C1, C2)

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim
    mssim = S[pad:(S.shape[0]-pad), pad:(S.shape[1]-pad)].mean()

    if gradient:
        # The following is Eqs. 7-8 of Avanaki 2009.
        grad = filter_func(A1 / D, **filter_args) * X
        grad += filter_func(-S / B2, **filter_args) * Y
        grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D,
                            **filter_args)
        grad *= (2 / X.size)

        if full:
            return mssim, grad, S
        else:
            return mssim, grad
    else:
        if full:
            return mssim, S
        else:
            return mssim


def SSIM_luminance(
        im_1: np.ndarray,
        im_2: np.ndarray,
        K1: float=0.01,
        K2: float=0.03,
        use_gaussian_window: bool=None,
        window_size: int=None,
        data_range: float=None,
        sigma: float=None,
        auto_downsample: bool=None,
        use_sample_covariance: bool=None,
        like_matlab: bool=False,
        **kwargs
        ) -> Tuple[float, np.ndarray]:
    """Returns the mean luminance component of the SSIM index and luminance
    component of the SSIM image for a comparison of two images

    Given two images, this function will return the mean luminance component
    of the SSIM index and luminance component of the full SSIM image.

    This function is heavily based on `skimage.measure.structural_similarity`_,
    so more detailed documentation may be found there.

    :type im_1: ``numpy.ndarray``
    :param im_1: The first image in the comparison
    :type im_2: ``numpy.ndarray``
    :param im_2: The second image in the comparison
    :type K1: ``float``
    :param K1: (default=0.01) The K1 parameter used in the SSIM calculation
    :type K2: ``float``
    :param K2: (default=0.03) The K2 parameter used in the SSIM calculation
    :type use_gaussian_window: ``bool``
    :param use_gaussian_window: (default=False) If set to True,
        this function will use a gaussian window for the SSIM calculation
    :type window_size: ``int``
    :param window_size: (Optional) The size of the window to be used in the
        SSIM calculation. If **use_gaussian_window** is set to True,
        then this argument is ignored.
    :type data_range: ``float``
    :param data_range: (Optional) The range of values that the image can span.
        If this parameter is not set, then the range will be determined
        from the minimum and maximum values of the images. For uint8
        images, this value should be 255.
    :type sigma: ``float``
    :param sigma: (Optional) If **use_gaussian_window** is set to True,
        this parameter determines the sigma used for the gaussian window.
    :type auto_downsample: ``bool``
    :param auto_downsample: (default=True) If set to True, this function will
        automatically downsample the inputs. The downsampling consists of
        finding the smallest of the images' first two dimensions (number of
        rows and number of columns), dividing this value by 256, rounding
        the output, and setting this final value as the scaling factor (sf). A
        square (sf x sf) lowpass averaging filter is then applied to both
        images with mode 'same'. Finally, the two images are downsampled by
        sf in each dimension.
    :type use_sample_covariance: ``bool``
    :param use_sample_covariance: (default=False) If set to True,
        this function will use sample covariances in its calculations.
        Otherwise, covariances of 1 will be used.
    :type like_matlab: ``bool``
    :param like_matlab: If set to True, this function will act like the
        Matlab function ssim_index. This is merely a specific configuration
        of the above parameters.
    :param kwargs: For a full list of keyword arguments, see
        `skimage.measure.structural_similarity`_.
    :rtype: ``(float, numpy.ndarray)``
    :return: A tuple of two elements. The first element is the mean SSIM
        index. The second element is the full SSIM image.

    .. seealso::
        `skimage.measure.structural_similarity`_
            Documentation of the structural_similarity function from Scikit Image
        `SSIM Wikipedia Page`_
            Wikipedia page describing the formulae for SSIM calculations

    .. _skimage.measure.structural_similarity: http://scikit-image.org/docs/dev/api/
        skimage.measure.html#skimage.measure.structural_similarity

    .. _SSIM Wikipedia Page: https://en.wikipedia.org/wiki/
        Structural_similarity#Formula_components

    """
    im_1, im_2, kwargs = _SSIM_preprocess(im_1=im_1, im_2=im_2, K1=K1,
            K2=K2, use_gaussian_window=use_gaussian_window,
            window_size=window_size, data_range=data_range, sigma=sigma,
            auto_downsample=auto_downsample,
            use_sample_covariance=use_sample_covariance,
            like_matlab=like_matlab, **kwargs)
    def callback_func(ux, uy, uxx, uyy, uxy, vx, vy, vxy, C1, C2):
        return ((2 * ux * uy) + C1) / ((ux ** 2) + (uy ** 2) + C1)
    if not like_matlab:
        return _compare_ssim_with_callback(im_1, im_2,
                callback_func=callback_func, **kwargs)
    else:
        mssim, ssim_map = _compare_ssim_with_callback(im_1, im_2,
                callback_func=callback_func, **kwargs)
        return mssim, _SSIM_like_matlab_crop(ssim_map, **kwargs)


def SSIM_contrast(
        im_1: np.ndarray,
        im_2: np.ndarray,
        K1: float=0.01,
        K2: float=0.03,
        use_gaussian_window: bool=None,
        window_size: int=None,
        data_range: float=None,
        sigma: float=None,
        auto_downsample: bool=None,
        use_sample_covariance: bool=None,
        like_matlab: bool=False,
        **kwargs
        ) -> Tuple[float, np.ndarray]:
    """Returns the mean contrast component of the SSIM index and contrast
    component of the SSIM image for a comparison of two images

    Given two images, this function will return the mean contrast component
    of the SSIM index and contrast component of the full SSIM image.

    This function is heavily based on `skimage.measure.structural_similarity`_,
    so more detailed documentation may be found there.

    :type im_1: ``numpy.ndarray``
    :param im_1: The first image in the comparison
    :type im_2: ``numpy.ndarray``
    :param im_2: The second image in the comparison
    :type K1: ``float``
    :param K1: (default=0.01) The K1 parameter used in the SSIM calculation
    :type K2: ``float``
    :param K2: (default=0.03) The K2 parameter used in the SSIM calculation
    :type use_gaussian_window: ``bool``
    :param use_gaussian_window: (default=False) If set to True,
        this function will use a gaussian window for the SSIM calculation
    :type window_size: ``int``
    :param window_size: (Optional) The size of the window to be used in the
        SSIM calculation. If **use_gaussian_window** is set to True,
        then this argument is ignored.
    :type data_range: ``float``
    :param data_range: (Optional) The range of values that the image can span.
        If this parameter is not set, then the range will be determined
        from the minimum and maximum values of the images. For uint8
        images, this value should be 255.
    :type sigma: ``float``
    :param sigma: (Optional) If **use_gaussian_window** is set to True,
        this parameter determines the sigma used for the gaussian window.
    :type auto_downsample: ``bool``
    :param auto_downsample: (default=True) If set to True, this function will
        automatically downsample the inputs. The downsampling consists of
        finding the smallest of the images' first two dimensions (number of
        rows and number of columns), dividing this value by 256, rounding
        the output, and setting this final value as the scaling factor (sf). A
        square (sf x sf) lowpass averaging filter is then applied to both
        images with mode 'same'. Finally, the two images are downsampled by
        sf in each dimension.
    :type use_sample_covariance: ``bool``
    :param use_sample_covariance: (default=False) If set to True,
        this function will use sample covariances in its calculations.
        Otherwise, covariances of 1 will be used.
    :type like_matlab: ``bool``
    :param like_matlab: If set to True, this function will act like the
        Matlab function ssim_index. This is merely a specific configuration
        of the above parameters.
    :param kwargs: For a full list of keyword arguments, see
        `skimage.measure.structural_similarity`_.
    :rtype: ``(float, numpy.ndarray)``
    :return: A tuple of two elements. The first element is the mean SSIM
        index. The second element is the full SSIM image.

    .. seealso::
        `skimage.measure.structural_similarity`_
            Documentation of the structural_similarity function from Scikit Image
        `SSIM Wikipedia Page`_
            Wikipedia page describing the formulae for SSIM calculations

    .. _skimage.measure.structural_similarity: http://scikit-image.org/docs/dev/api/
        skimage.measure.html#skimage.measure.structural_similarity

    .. _SSIM Wikipedia Page: https://en.wikipedia.org/wiki/
        Structural_similarity#Formula_components

    """
    im_1, im_2, kwargs = _SSIM_preprocess(im_1=im_1, im_2=im_2, K1=K1,
            K2=K2, use_gaussian_window=use_gaussian_window,
            window_size=window_size, data_range=data_range, sigma=sigma,
            auto_downsample=auto_downsample,
            use_sample_covariance=use_sample_covariance,
            like_matlab=like_matlab, **kwargs)
    def callback_func(ux, uy, uxx, uyy, uxy, vx, vy, vxy, C1, C2):
        return ((2 * ((vx * vy) ** 0.5)) + C2) / (vx + vy + C2)
    if not like_matlab:
        return _compare_ssim_with_callback(im_1, im_2,
                callback_func=callback_func, **kwargs)
    else:
        mssim, ssim_map = _compare_ssim_with_callback(im_1, im_2,
                callback_func=callback_func, **kwargs)
        return mssim, _SSIM_like_matlab_crop(ssim_map, **kwargs)


def SSIM_structure(
        im_1: np.ndarray,
        im_2: np.ndarray,
        K1: float=0.01,
        K2: float=0.03,
        use_gaussian_window: bool=None,
        window_size: int=None,
        data_range: float=None,
        sigma: float=None,
        auto_downsample: bool=None,
        use_sample_covariance: bool=None,
        like_matlab: bool=False,
        **kwargs
        ) -> Tuple[float, np.ndarray]:
    """Returns the mean structure component of the SSIM index and structure
    component of the SSIM image for a comparison of two images

    Given two images, this function will return the mean structure component
    of the SSIM index and structure component of the full SSIM image.

    This function is heavily based on `skimage.measure.structural_similarity`_,
    so more detailed documentation may be found there.

    :type im_1: ``numpy.ndarray``
    :param im_1: The first image in the comparison
    :type im_2: ``numpy.ndarray``
    :param im_2: The second image in the comparison
    :type K1: ``float``
    :param K1: (default=0.01) The K1 parameter used in the SSIM calculation
    :type K2: ``float``
    :param K2: (default=0.03) The K2 parameter used in the SSIM calculation
    :type use_gaussian_window: ``bool``
    :param use_gaussian_window: (default=False) If set to True,
        this function will use a gaussian window for the SSIM calculation
    :type window_size: ``int``
    :param window_size: (Optional) The size of the window to be used in the
        SSIM calculation. If **use_gaussian_window** is set to True,
        then this argument is ignored.
    :type data_range: ``float``
    :param data_range: (Optional) The range of values that the image can span.
        If this parameter is not set, then the range will be determined
        from the minimum and maximum values of the images. For uint8
        images, this value should be 255.
    :type sigma: ``float``
    :param sigma: (Optional) If **use_gaussian_window** is set to True,
        this parameter determines the sigma used for the gaussian window.
    :type auto_downsample: ``bool``
    :param auto_downsample: (default=True) If set to True, this function will
        automatically downsample the inputs. The downsampling consists of
        finding the smallest of the images' first two dimensions (number of
        rows and number of columns), dividing this value by 256, rounding
        the output, and setting this final value as the scaling factor (sf). A
        square (sf x sf) lowpass averaging filter is then applied to both
        images with mode 'same'. Finally, the two images are downsampled by
        sf in each dimension.
    :type use_sample_covariance: ``bool``
    :param use_sample_covariance: (default=False) If set to True,
        this function will use sample covariances in its calculations.
        Otherwise, covariances of 1 will be used.
    :type like_matlab: ``bool``
    :param like_matlab: If set to True, this function will act like the
        Matlab function ssim_index. This is merely a specific configuration
        of the above parameters.
    :param kwargs: For a full list of keyword arguments, see
        `skimage.measure.structural_similarity`_.
    :rtype: ``(float, numpy.ndarray)``
    :return: A tuple of two elements. The first element is the mean SSIM
        index. The second element is the full SSIM image.

    .. seealso::
        `skimage.measure.structural_similarity`_
            Documentation of the structural_similarity function from Scikit Image
        `SSIM Wikipedia Page`_
            Wikipedia page describing the formulae for SSIM calculations

    .. _skimage.measure.structural_similarity: http://scikit-image.org/docs/dev/api/
        skimage.measure.html#skimage.measure.structural_similarity

    .. _SSIM Wikipedia Page: https://en.wikipedia.org/wiki/
        Structural_similarity#Formula_components

    """
    im_1, im_2, kwargs = _SSIM_preprocess(im_1=im_1, im_2=im_2, K1=K1,
            K2=K2, use_gaussian_window=use_gaussian_window,
            window_size=window_size, data_range=data_range, sigma=sigma,
            auto_downsample=auto_downsample,
            use_sample_covariance=use_sample_covariance,
            like_matlab=like_matlab, **kwargs)
    def callback_func(ux, uy, uxx, uyy, uxy, vx, vy, vxy, C1, C2):
        return (vxy + (C2 / 2)) / (((vx * vy) ** 0.5) + (C2 /2))
    if not like_matlab:
        return _compare_ssim_with_callback(im_1, im_2,
                callback_func=callback_func, **kwargs)
    else:
        mssim, ssim_map = _compare_ssim_with_callback(im_1, im_2,
                callback_func=callback_func, **kwargs)
        return mssim, _SSIM_like_matlab_crop(ssim_map, **kwargs)

