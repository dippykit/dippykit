"""Module of various image sampling functions

This module contains an assortment of functions that sample images in various
useful manners.

"""

# This library was developed for the Georgia Tech graduate course ECE 6258:
# Digital Image Processing with Professor Ghassan AlRegib.
# For comments and feedback, please email dippykit[at]gmail.com

# Internal imports
from . import _utils

# Functional imports
import numpy as np
import cv2

__author__ = 'Brighton Ancelin, Motaz Alfarraj, Ghassan AlRegib'

__all__ = ['_downsample', '_upsample', 'resample']


def _downsample(
        image: np.ndarray,
        ds_matrix: np.ndarray,
        ) -> np.ndarray:
    """Downsamples an image based on a given downsampling matrix.

    **This function should seldom ever be used. Instead, one should often
    use :func:`resample`. The :func:`resample` function can do everything
    that this function can do and more. This function is faster than the
    :func:`resample` function for downsampling without interpolation.**

    Downsamples an image by the given downsampling matrix argument,
    ds_matrix. For an input domain vector **mn** = [**m**, **n**] and an input
    image **f** (**mn**), the output image **g** (**.**) is defined by **g**
    (**mn**) = **f** (**M** @ **mn**) where **M** is the downsampling matrix
    ds_matrix.

    :type image: ``numpy.ndarray``
    :param image: An image to be downsampled.
    :type ds_matrix: ``numpy.ndarray`` (dtype must be integer)
    :param ds_matrix: The downsampling matrix **M** to be used.
    :rtype: ``numpy.ndarray``
    :return: The downsampled image

    Examples:

    >>> image = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7],
    ...                   [10, 11, 12, 13, 14, 15, 16, 17],
    ...                   [20, 21, 22, 23, 24, 25, 26, 27],
    ...                   [30, 31, 32, 33, 34, 35, 36, 37],
    ...                   [40, 41, 42, 43, 44, 45, 46, 47],
    ...                   [50, 51, 52, 53, 54, 55, 56, 57],
    ...                   [60, 61, 62, 63, 64, 65, 66, 67],
    ...                   [70, 71, 72, 73, 74, 75, 76, 77]])
    >>> M = np.array([[2, 0], [0, 2]])
    >>> downsampled_image = downsample(image, M)
    >>> downsampled_image
    array([[ 0.,  2.,  4.,  6.],
           [20., 22., 24., 26.],
           [40., 42., 44., 46.],
           [60., 62., 64., 66.]])
    >>> M2 = np.array([[1, 2], [2, 1]])
    >>> downsample(downsampled_image, M2)
    array([[ 0.,  0.,  0., 60.],
           [ 0.,  0., 42.,  0.],
           [ 0., 24., 66.,  0.],
           [ 6.,  0.,  0.,  0.]])

    """
    assert (2, 2) == ds_matrix.shape, "Argument 'ds_matrix' must be an " \
            "ndarray with shape (2, 2)"
    assert np.issubdtype(ds_matrix.dtype, np.integer), "Argument " \
            "'ds_matrix' must be an ndarray with an integer dtype"
    ds_matrix_det = ds_matrix[0, 0] * ds_matrix[1, 1] \
            - ds_matrix[0, 1] * ds_matrix[1, 0]
    assert 0 != ds_matrix_det, "Argument 'ds_matrix' must be nonsingular"
    height = image.shape[0]
    width = image.shape[1]
    ds_matrix_inv_scaled = np.array(
            [[ds_matrix[1, 1], -ds_matrix[0, 1]],
            [-ds_matrix[1, 0], ds_matrix[0, 0]]])
    kl_extrema = np.array(
            np.meshgrid([0, height-1], [0, width-1])).reshape(2, -1)
    mn_extrema = (1/ds_matrix_det) * (ds_matrix_inv_scaled @ kl_extrema)
    m_min = np.min(mn_extrema[0])
    m_max = np.max(mn_extrema[0])
    n_min = np.min(mn_extrema[1])
    n_max = np.max(mn_extrema[1])
    kl = np.array(np.meshgrid(np.arange(0, height),
                              np.arange(0, width))).reshape(2, -1)
    mn = ds_matrix_inv_scaled @ kl
    mn_lattice_indices = np.all(0 == mn % ds_matrix_det, axis=0)
    mn = ((1/ds_matrix_det) * mn[:, mn_lattice_indices]).astype(int)
    kl = kl[:, mn_lattice_indices]
    ds_image = np.zeros((np.floor(m_max - m_min + 1).astype(int),
                         np.floor(n_max - n_min + 1).astype(int)))
    mn_offset = np.fix(np.array([[m_min], [n_min]])).astype(int)
    ds_image[tuple(mn - mn_offset)] = image[tuple(kl)]
    return ds_image


def _upsample(
        image: np.ndarray,
        us_matrix: np.ndarray,
        ) -> np.ndarray:
    """Upsamples an image based on a given upsampling matrix.

    **This function should seldom ever be used. Instead, one should often
    use :func:`resample`. The :func:`resample` function can do everything
    that this function can do and more. This function is faster than the
    :func:`resample` function for upsampling without interpolation.**

    Upsamples an image by the given upsampling matrix argument, us_matrix.
    For an input domain vector **mn** = [**m**, **n**] and an input
    image **f** (**mn**), the output image **g** (**.**) is defined by **g**
    (**mn**) = **f** (**M** @ **mn**) where **M** is the upsampling matrix
    us_matrix.

    :type image: ``numpy.ndarray``
    :param image: An image to be upsampled.
    :type us_matrix: ``numpy.ndarray``
    :param us_matrix: The upsampling matrix **M** to be used.
    :rtype: ``numpy.ndarray``
    :return: The upsampled image

    Examples:

    >>> image = np.array([[ 0,  1,  2,  3],
    ...                   [10, 11, 12., 13],
    ...                   [20, 21, 22., 23],
    ...                   [30, 31, 32., 33]])
    >>> M = np.array([[1/2, 0], [0, 1/2]])
    >>> upsample(image, M)
    array([[ 0.,  0.,  1.,  0.,  2.,  0.,  3.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [10.,  0., 11.,  0., 12.,  0., 13.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [20.,  0., 21.,  0., 22.,  0., 23.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [30.,  0., 31.,  0., 32.,  0., 33.]])
    >>> M2 = np.array([[-1/3, -2/3], [-2/3, -1/3]])
    >>> upsample(image, M2)
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0., 13.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0., 23.,  0.,  0.,  2.,  0.],
           [ 0.,  0.,  0., 33.,  0.,  0., 12.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0., 22.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0., 32.,  0.,  0., 11.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0., 21.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0., 31.,  0.,  0., 10.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0., 20.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [30.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    """
    assert (2, 2) == us_matrix.shape, "Argument 'us_matrix' must be an " \
            "ndarray with shape (2, 2)"
    us_matrix_det = us_matrix[0, 0] * us_matrix[1, 1] \
            - us_matrix[0, 1] * us_matrix[1, 0]
    assert 0 != us_matrix_det, "Argument 'us_matrix' must be nonsingular"
    height = image.shape[0]
    width = image.shape[1]
    us_matrix_inv = ((1/us_matrix_det) * np.array(
            [[us_matrix[1, 1], -us_matrix[0, 1]],
            [-us_matrix[1, 0], us_matrix[0, 0]]])).astype(int)
    kl_extrema = np.array(
            np.meshgrid([0, height-1], [0, width-1])).reshape(2, -1)
    mn_extrema = us_matrix_inv @ kl_extrema
    m_min = np.min(mn_extrema[0])
    m_max = np.max(mn_extrema[0])
    n_min = np.min(mn_extrema[1])
    n_max = np.max(mn_extrema[1])
    kl = np.array(np.meshgrid(np.arange(0, height),
                              np.arange(0, width))).reshape(2, -1)
    mn = us_matrix_inv @ kl
    us_image = np.zeros((m_max - m_min + 1, n_max - n_min + 1))
    mn_offset = np.array([[m_min], [n_min]])
    us_image[tuple(mn - mn_offset)] = image[tuple(kl)]
    return us_image


def resample(
        image: np.ndarray,
        rs_matrix: np.ndarray,
        **kwargs
        ) -> np.ndarray:
    """Resamples an image based on a given resampling matrix.

    Resamples an image by the given resampling matrix argument, rs_matrix.
    For an input domain vector **mn** = [**m**, **n**] and an input
    image **f** (**mn**), the output image **g** (**.**) is defined by **g**
    (**mn**) = **f** (**M** @ **mn**) where **M** is the resampling matrix
    (rs_matrix) and the @ symbol denotes matrix multiplication.

    :type image: ``numpy.ndarray``
    :param image: An image to be resampled.
    :type rs_matrix: ``numpy.ndarray``
    :param rs_matrix: The resampling matrix **M** to be used.
    :rtype: ``numpy.ndarray``
    :return: The resampled image.

    :Keyword Arguments:
        * **interpolation** (``str``) --
          (default=None) The interpolation method used during the
          resampling. The lack of an interpolation method will lead to a
          resampling where only pixels that perfectly map across the input
          and output domains are included. This typically only works well
          for resampling matrices that contain only integer entries (which
          will downsample), or the inverses of such matrices (which will
          upsample).

          The interpolation keyword argument can take on any of the
          following values (For more information, see `cv2.warpAffine`_):

            * '**nearest**' (See cv2.INTER_NEAREST)
            * '**linear**' or '**bilinear**' (See cv2.INTER_LINEAR)
            * '**area**' (See cv2.INTER_AREA)
            * '**cubic**' or '**bicubic**' (See cv2.INTER_CUBIC)
            * '**lanczos4**' (See cv2.INTER_LANCZOS4)
        * **crop** (``bool``) --
          (default=False) Whether or not to crop the output image. If this
          keyword argument is specified to be ``True`` without specifying a
          crop_size keyword argument, then the crop_size will be set to the
          original size of the input image by default.
        * **crop_size** (``ShapeType``) --
          (default=image.shape) The dimensions of the output image after
          cropping. An integer argument signifies dimension in pixels,
          a floating point argument signifies dimension in proportion to the
          output image size (before cropping). A single value will yield a
          square output image.

    .. _cv2.warpAffine: https://docs.opencv.org/2.4/modules/imgproc/doc
        /geometric_transformations.html?#void%20warpAffine(InputArray%20src,
        %20OutputArray%20dst,%20InputArray%20M,%20Size%20dsize,%20int
        %20flags,%20int%20borderMode,%20const%20Scalar&%20borderValue)

    Examples:

    >>> import numpy as np
    >>> image = np.array([[ 0,  1,  2,  3],
    ...                   [10, 11, 12., 13],
    ...                   [20, 21, 22., 23],
    ...                   [30, 31, 32., 33]])
    >>> M = np.array([[1/2, 0], [0, 1/2]])
    >>> resample(image, M)
    array([[ 0.,  0.,  1.,  0.,  2.,  0.,  3.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [10.,  0., 11.,  0., 12.,  0., 13.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [20.,  0., 21.,  0., 22.,  0., 23.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [30.,  0., 31.,  0., 32.,  0., 33.]])
    >>> resample(image, M, interp='lin')
    array([[ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ],
           [ 5. ,  5.5,  6. ,  6.5,  7. ,  7.5,  8. ],
           [10. , 10.5, 11. , 11.5, 12. , 12.5, 13. ],
           [15. , 15.5, 16. , 16.5, 17. , 17.5, 18. ],
           [20. , 20.5, 21. , 21.5, 22. , 22.5, 23. ],
           [25. , 25.5, 26. , 26.5, 27. , 27.5, 28. ],
           [30. , 30.5, 31. , 31.5, 32. , 32.5, 33. ]])
    >>> resample(image, M, interp='lin', crop=True, crop_size=(4,3))
    array([[ 6. ,  6.5,  7. ],
           [11. , 11.5, 12. ],
           [16. , 16.5, 17. ],
           [21. , 21.5, 22. ]])
    >>> M2 = np.array([[2, 0], [0, 2]])
    >>> resample(image, M2)
    array([[ 0.,  2.],
           [20., 22.]])

    """
    def resample_no_interp():
        rs_matrix_scaled_int = (resolution_factor * rs_matrix).astype(int)
        height = image.shape[0]
        width = image.shape[1]
        rs_matrix_inv = np.linalg.inv(rs_matrix)
        # Create a matrix where each column is a coordinate of one of the
        # corners in the domain of the input image (kl-space)
        kl_extrema = np.array(np.meshgrid([0, height - 1], [0, width - 1])) \
            .reshape(2, -1)
        # Convert the extrema in input image domain (kl-space) to output image
        # domain (mn-space)
        mn_extrema = rs_matrix_inv @ kl_extrema
        # Determine the minimum and maximum values of each coordinate possible.
        # We fix the values (round towards 0) because only integer values of mn
        # will be actual pixels
        m_min = np.fix(np.min(mn_extrema[0])).astype(int)
        m_max = np.fix(np.max(mn_extrema[0])).astype(int)
        n_min = np.fix(np.min(mn_extrema[1])).astype(int)
        n_max = np.fix(np.max(mn_extrema[1])).astype(int)
        # Create a matrix where each column is a coordinate in output image
        # domain (mn-space) and the set of all columns is the set of all pixels
        # in the output image
        mn = np.array(np.meshgrid(np.arange(m_min, m_max + 1),
                                  np.arange(n_min, n_max + 1))).reshape(2, -1)
        # Convert the output image domain (mn-space) to input image domain,
        # but have every value scaled by resolution_factor (this keeps
        # everything as integers)
        kl_scaled = rs_matrix_scaled_int @ mn
        # The only output image coordinates that will retain a pixel value from
        # the input image are those that perfectly map from integer kl values
        # to integer mn values. The mask of these valid lattice coordinates
        # in the mn and kl coordinate matrices is created below.
        lattice_mask = np.all(0 == kl_scaled % resolution_factor, axis=0)
        # Mask and scale down kl_scaled to create kl for next masking
        kl = (kl_scaled[:, lattice_mask] / resolution_factor).astype(int)
        # Only non-negative kl values within the kl max values (height and
        # width) will correspond to pixels in the input image.
        domain_mask = np.all((0 <= kl) & (np.array([[height], [width]]) > kl),
                             axis=0)
        # Mask the domains to only valid values
        kl = kl[:, domain_mask]
        mn = (mn[:, lattice_mask])[:, domain_mask]
        # Create an image of 0s of the correct size
        rs_image_height = m_max - m_min + 1
        rs_image_width = n_max - n_min + 1
        rs_image = np.zeros((rs_image_height, rs_image_width)) \
                .astype(image.dtype)
        # Create a column vector of the offsets in m and n
        mn_offset = np.array([[m_min], [n_min]])
        # Assign values from the input image to the output image accordingly
        rs_image[tuple(mn - mn_offset)] = image[tuple(kl)]
        return rs_image

    def resample_interp():
        height = image.shape[0]
        width = image.shape[1]
        rs_matrix_inv = np.linalg.inv(rs_matrix)
        # Create a matrix where each column is a coordinate of one of the
        # corners in the domain of the input image (kl-space)
        kl_extrema = np.array(np.meshgrid([0, height - 1], [0, width - 1])) \
            .reshape(2, -1)
        # Convert the extrema in input image domain (kl-space) to output image
        # domain (mn-space)
        mn_extrema = rs_matrix_inv @ kl_extrema
        # Determine the minimum and maximum values of each coordinate possible.
        # We fix the values (round towards 0) because only integer values of mn
        # will be actual pixels
        m_min = np.fix(np.min(mn_extrema[0])).astype(int)
        m_max = np.fix(np.max(mn_extrema[0])).astype(int)
        n_min = np.fix(np.min(mn_extrema[1])).astype(int)
        n_max = np.fix(np.max(mn_extrema[1])).astype(int)
        # Restructure the rs_matrix to fit the format desired by
        # cv2.warpAffine()
        aff_resample = np.linalg.inv(rs_matrix)
        aff_resample[[0, 0, 1, 1], [0, 1, 0, 1]] = \
                aff_resample[[1, 1, 0, 0], [1, 0, 1, 0]]
        aff_translation = np.array([[-n_min], [-m_min]])
        rs_image_dims = (n_max - n_min + 1, m_max - m_min + 1)
        affine_mat = np.concatenate((aff_resample, aff_translation), axis=1)
        # Return the cv2.warpAffine() transformation of the image
        return cv2.warpAffine(image, affine_mat.astype(float), rs_image_dims,
                              flags=arg_dict['interpolation'])

    possible_arg_list = ['crop', 'crop_size', 'interpolation']
    possible_interp_list = ['none', 'nearest', 'linear', 'bilinear', 'area',
                            'cubic', 'bicubic', 'lanczos4']
    arg_dict = _utils.resolve_arg_dict_from_list(kwargs,
                                                 possible_arg_list, warn_user_missing=False)
    if 'interpolation' in arg_dict:
        if isinstance(arg_dict['interpolation'], str):
            arg_dict['interpolation'] = _utils.resolve_arg_from_list(
                    arg_dict['interpolation'], possible_interp_list)
            if 'bilinear' == arg_dict['interpolation']:
                arg_dict['interpolation'] = 'linear'
            elif 'bicubic' == arg_dict['interpolation']:
                arg_dict['interpolation'] = 'cubic'
            interp_dict = {
                'none': None,
                'nearest': cv2.INTER_NEAREST,
                'linear': cv2.INTER_LINEAR,
                'area': cv2.INTER_AREA,
                'cubic': cv2.INTER_CUBIC,
                'lanczos4': cv2.INTER_LANCZOS4,
            }
            arg_dict['interpolation'] = interp_dict[arg_dict['interpolation']]
    assert (2, 2) == rs_matrix.shape, "Argument 'rs_matrix' must be an " \
            "ndarray with shape (2, 2)"
    # The rs_matrix must contain rational entries. We define a
    # resolution_factor of 100 to signify that every element in rs_matrix
    # can be expressed as an integer divided by 100. To ensure this,
    # we round rs_matrix to 2 decimal places.
    resolution_factor = 100
    rs_matrix = np.round(rs_matrix, 2)
    assert 0 != np.linalg.det(rs_matrix), \
            "Argument 'rs_matrix' must be nonsingular"
    if 'interpolation' not in arg_dict or arg_dict['interpolation'] is None:
        rs_image = resample_no_interp()
    else:
        rs_image = resample_interp()
    (rs_image_height, rs_image_width) = np.array(rs_image.shape)

    if arg_dict.get('crop', False):
        if 'crop_size' in arg_dict:
            crop_size = _utils.resolve_shape_arg(arg_dict['crop_size'],
                                                 (rs_image_height, rs_image_width), 'crop_size')
        else:
            # Default cropping: the input image dimensions
            crop_size = image.shape

        if crop_size[0] <= rs_image_height:
            m_idx = np.floor((rs_image_height - crop_size[0])/2).astype(int) \
                    + np.arange(0, crop_size[0])
            rs_image = rs_image[m_idx, :]
        else:
            height_diff = crop_size[0] - rs_image_height
            pad_top = np.floor(height_diff/2).astype(int)
            pad_bottom = height_diff - pad_top
            rs_image = np.pad(rs_image, ((pad_top, pad_bottom), (0, 0)),
                              'constant', constant_values=0)
        if crop_size[1] <= rs_image_width:
            n_idx = np.floor((rs_image_width - crop_size[1]) / 2).astype(int) \
                    + np.arange(0, crop_size[1])
            rs_image = rs_image[:, n_idx]
        else:
            width_diff = crop_size[1] - rs_image_width
            pad_left = np.floor(width_diff/2).astype(int)
            pad_right = width_diff - pad_left
            rs_image = np.pad(rs_image, ((0, 0), (pad_left, pad_right)),
                              'constant', constant_values=0)
    return rs_image

