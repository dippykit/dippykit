"""Module of image I/O related functions

This module contains an assortment of functions that make the input and
output of images much simpler. The syntax is similar to that of Matlab.

"""

# Functional imports
import numpy as np
from PIL import Image

__author__ = 'Brighton Ancelin'

__all__ = ['im_read', 'im_write', 'im_to_float', 'float_to_im']


def im_read(
        filepath: str,
        ) -> np.ndarray:
    """Reads an image from a file

    Using Pillow, attempts to open the image at the given filepath argument
    and subsequently converts the image into a numpy array.

    This function is essentially a wrapper for `PIL.Image.open`_,
    so more detailed documentation may be found there.

    :type filepath: ``str``
    :param filepath: A filepath to the desired image.
    :rtype: ``numpy.ndarray``
    :return: The image as a numpy array.
    :exception IOError: *"If the file cannot be found, or the image cannot be
        opened and identified." -Pillow*

    .. note::
        This function wraps around functions from other packages. Reading
        these functions' documentations may be useful. See the **See also**
        section for more information.

    .. seealso::
        `PIL.Image.open`_
            Documentation of the open function from Pillow

    .. _PIL.Image.open: https://pillow.readthedocs.io/en/3.1.x/reference
        /Image.html#functions

    Examples:

    >>> im_read('black_image.tif')
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)

    """
    return np.array(Image.open(filepath))


def im_write(
        image: np.ndarray,
        filepath: str,
        quality: int=75,
        ) -> None:
    """Writes an image to a file

    Using Pillow, attempts to write the image to a given filepath argument.

    This function is essentially a wrapper for `PIL.Image.save`_,
    so more detailed documentation may be found there.

    :type image: ``numpy.ndarray``
    :param image: The image to be written to a file.
    :type filepath: ``str``
    :param filepath: A filepath for the output image.
    :type quality: ``int``
    :param quality: (default=75) The quality level at which you'd like to
        save the image. This value should range from 1 (worst) to 95 (best).
        This value is primarily used when saving JPEG files.
    :exception KeyError: *"If the output format could not be determined
        from the file name." -Pillow*
    :exception IOError: *"If the file could not be written.  The file
        may have been created, and may contain partial data." -Pillow*

    .. note::
        This function wraps around functions from other packages. Reading
        these functions' documentations may be useful. See the **See also**
        section for more information.

    .. seealso::
        `PIL.Image.save`_
            Documentation of the save function from Pillow

    .. _PIL.Image.save: https://pillow.readthedocs.io/en/3.1.x/reference
        /Image.html#PIL.Image.Image.save

    Examples:

    >>> import numpy as np
    >>> image = np.array([[0, 255], [255, 0]], dtype='uint8')
    >>> im_write(image, 'image.tif')
    >>> # There is now a file 'image.tif' in the current directory

    """
    if 'u' != image.dtype.kind:
        raise ValueError("Argument 'image' must be an ndarray with an "
                         "unsigned integer dtype")
    Image.fromarray(image).save(filepath, quality=quality)


def im_to_float(
        image: np.ndarray,
        ) -> np.ndarray:
    """Converts an image from unsigned integer format to normalized floating
    point format

    Given an image in an unsigned integer format (values between 0 and
    2\ :superscript:`N`\  - 1), this function converts the image to a
    floating point format where each value is normalized between
    0.0 and 1.0. Images in this format are more easily processed reliably.

    The dtype of the input image must be one of the following: ``uint8``,
    ``uint16``, ``uint32``, or ``uint64``. The values in the image are
    assumed to have a range of 0~(2\ :superscript:`N`\  - 1) inclusive,
    where N is the number of bits used to represent each unsigned integer.
    This means that if one wants to convert an image with dtype ``uint8``,
    every instance of the value 0 in the input image will become a 0.0 in the
    output image and every instance of the value 255 in the input image will
    become a 1.0 in the output image.

    :type image: ``numpy.ndarray`` (dtype must be unsigned integer)
    :param image: The image in unsigned integer format.
    :rtype: ``numpy.ndarray``
    :return: The image in normalized floating point format (values between
        0.0 and 1.0).
    :exception ValueError: If the dtype of the image argument is not
        unsigned integer.

    Examples:

    >>> import numpy as np
    >>> image = np.array([[0, 64], [128, 255]], dtype='uint8')
    >>> im_to_float(image)
    array([[0.        , 0.25098039],
           [0.50196078, 1.        ]])

    """
    if 'u' != image.dtype.kind:
        raise ValueError("Argument 'image' must be an ndarray with an "
                         "unsigned integer dtype")
    return image.astype('float64') / np.iinfo(image.dtype).max


def float_to_im(
        image: np.ndarray,
        bit_depth: int=8,
        ) -> np.ndarray:
    """Converts an image from normalized floating point space to integer space

    Given an image in normalized floating point format (values between 0.0
    and 1.0), this function converts the image to an unsigned integer format
    normalized to the range of values of the format (e.g. for ``uint8`` this
    range is 0~(2\ :superscript:`N`\  - 1) = 0~255 inclusive). Images in
    this format are more easily stored or written to files.

    If any values in the image argument are less than 0.0 or greater than 1.0,
    they will be replaced with 0.0s and 1.0s, respectively. This allows for
    normalized floating point images to "saturate" in processing.

    If the bit_depth argument is specified, the image will be converted with
    the specified bit depth. The number of levels in the image will be
    2\ :superscript:`bit_depth`\ .

    The dtype of the returned image is dependent on the bit depth specified.
    By default, the bit depth is set to 8, meaning that the returned image
    will have a dtype of ``uint8``. For a given bit depth, the returned
    dtype will be the following:

        * 1 <= bit_depth <= 8: ``uint8`` (default)
        * 9 <= bit_depth <= 16: ``uint16``
        * 17 <= bit_depth <= 32: ``uint32``
        * 33 <= bit_depth <= 64: ``uint64``

    :type image: ``numpy.ndarray`` (dtype must be float)
    :param image: The image in normalized floating point format (0.0
        represents the minimum value and 1.0 represents the maximum value).
    :type bit_depth: ``int``
    :param bit_depth: (default=8) Bit depth for the converted image (between 1
        and 64 inclusive).
    :rtype: ``numpy.ndarray``
    :return: The image in unsigned integer format.
    :exception ValueError: If the bit_depth is not between 1 and 64.

    Examples:

    >>> import numpy as np
    >>> image = np.array([[0, 64], [128, 255]], dtype='uint8')
    >>> image_in_float = im_to_float(image)
    >>> image_in_float
    array([[0.        , 0.25098039],
           [0.50196078, 1.        ]])
    >>> float_to_im(image_in_float)
    array([[  0,  64],
           [128, 255]], dtype=uint8)
    >>> float_to_im(image_in_float, 1)
    array([[  0,   0],
           [128, 128]], dtype=uint8)

    """
    if 1 > bit_depth or 64 < bit_depth:
        raise ValueError("Argument 'bit_depth' must be between 1 and 64 ("
                         "inclusive)")
    image_copy = image.copy()
    image_copy[image_copy > 1] = 1
    image_copy[image_copy < 0] = 0
    if 8 >= bit_depth:
        scale = 2 ** (8 - bit_depth)
        new_dtype = np.uint8
    elif 16 >= bit_depth:
        scale = 2 ** (16 - bit_depth)
        new_dtype = np.uint16
    elif 32 >= bit_depth:
        scale = 2 ** (32 - bit_depth)
        new_dtype = np.uint32
    elif 64 >= bit_depth:
        scale = 2 ** (64 - bit_depth)
        new_dtype = np.uint64
    r_image = (scale * np.floor(image_copy * (np.iinfo(new_dtype).max /
            scale))).astype(new_dtype)
    return r_image

