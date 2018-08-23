"""Common import for all dippykit functions

Importing
^^^^^^^^^

Recommended techniques for importing this library are the following:

    * For a default import, use:
        >>> import dippykit as dip
        >>> # Refer to all functions using the dip.function() syntax, e.g.:
        >>> dip.window_2d(4, 'r', dim=2)
        array([[0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.25, 0.25, 0.  ],
               [0.  , 0.25, 0.25, 0.  ],
               [0.  , 0.  , 0.  , 0.  ]])

    * For a static import, use:
        >>> from dippykit import *
        >>> # Refer to all functions using the function() syntax, e.g.:
        >>> window_2d(4, 'r', dim=2)
        array([[0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.25, 0.25, 0.  ],
               [0.  , 0.25, 0.25, 0.  ],
               [0.  , 0.  , 0.  , 0.  ]])

New Types
^^^^^^^^^

In the documentation for this library there are two new types used. These
are ``NumericType`` and ``ShapeType``.

``NumericType`` is an alias for the union of ``int`` and ``float``, and as
such represents values that can be either integers or floating point
numbers.

``ShapeType`` is an alias for the union of ``NumericType`` and
``Tuple[NumericType, NumericType]``. In effect, ``ShapeType`` is something
that could be logically construed as the dimensions of some rectangle. For
single-element or scalar arguments, the shape is square. Integer arguments
describe the absolute size of the rectangle, whereas floating point
arguments describe the size of the rectangle relative to some other
rectangle. For example, when specifying the to crop a (256, 256) region from
an image of size (512, 512), one could specify a shape of 256, 0.5, (256,
256), or (0.5, 0.5).

Aliases to External Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some functions available through this library are merely aliases for
functions provided by external packages. This is done for simplicity and
ease of student experience. These aliases, along with links to their
referenced functions' documentation pages, are listed below:

    * numpy.fft
        * `fft`_
        * `fft2`_
        * `fftshift`_
        * `ifft`_
        * `ifft2`_
    * scipy.signal
        * `convolve`_
    * scipy.linalg
        * `dft`_
    * matplotlib.pyplot
        * `axis`_
        * `bar`_
        * `colorbar`_
        * `contour`_
        * `figure`_
        * `hist`_
        * `legend`_
        * `loglog`_
        * `plot`_
        * `semilogx`_
        * `semilogy`_
        * `show`_
        * `subplot`_
        * `suptitle`_
        * `title`_
        * `xlabel`_
        * `ylabel`_
    * cv2
        * `medianBlur`_
        * `resize`_
    * scipy.io
        * `loadmat`_
        * `savemat`_

    .. _fft: https://docs.scipy.org/doc/numpy/reference/generated
        /numpy.fft.fft.html
    .. _fft2: https://docs.scipy.org/doc/numpy/reference/generated
        /numpy.fft.fft2.html
    .. _fftshift: https://docs.scipy.org/doc/numpy/reference/generated
        /numpy.fft.fftshift.html
    .. _ifft: https://docs.scipy.org/doc/numpy/reference/generated
        /numpy.fft.ifft.html
    .. _ifft2: https://docs.scipy.org/doc/numpy/reference/generated
        /numpy.fft.ifft2.html

    .. _convolve: https://docs.scipy.org/doc/scipy/reference/generated
        /scipy.signal.convolve.html

    .. _dft: https://docs.scipy.org/doc/scipy/reference/generated
        /scipy.linalg.dft.html

    .. _axis: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axis.html
    .. _bar: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.bar.html
    .. _colorbar: https://matplotlib.org/api/_as_gen
        /matplotlib.pyplot.colorbar.html
    .. _contour: https://matplotlib.org/api/_as_gen
        /matplotlib.pyplot.contour.html
    .. _figure: https://matplotlib.org/api/_as_gen
        /matplotlib.pyplot.figure.html
    .. _hist: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
    .. _legend: https://matplotlib.org/api/_as_gen
        /matplotlib.pyplot.legend.html
    .. _loglog: https://matplotlib.org/api/_as_gen
        /matplotlib.pyplot.loglog.html
    .. _plot: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    .. _semilogx: https://matplotlib.org/api/_as_gen
        /matplotlib.pyplot.semilogx.html
    .. _semilogy: https://matplotlib.org/api/_as_gen
        /matplotlib.pyplot.semilogy.html
    .. _show: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.show.html
    .. _subplot: https://matplotlib.org/api/_as_gen
        /matplotlib.pyplot.subplot.html
    .. _suptitle: https://matplotlib.org/api/_as_gen
        /matplotlib.pyplot.suptitle.html
    .. _title: https://matplotlib.org/api/_as_gen
        /matplotlib.pyplot.title.html
    .. _xlabel: https://matplotlib.org/api/_as_gen
        /matplotlib.pyplot.xlabel.html
    .. _ylabel: https://matplotlib.org/api/_as_gen
        /matplotlib.pyplot.ylabel.html

    .. _medianBlur: https://docs.opencv.org/2.4/modules/imgproc/doc
        /filtering.html#void%20medianBlur(InputArray%20src,%20OutputArray
        %20dst,%20int%20ksize)
    .. _resize: https://docs.opencv.org/2.4/modules/imgproc/doc
        /geometric_transformations.html#void%20resize(InputArray%20src,
        %20OutputArray%20dst,%20Size%20dsize,%20double%20fx,%20double%20fy,
        %20int%20interpolation)

    .. _loadmat: https://docs.scipy.org/doc/scipy/reference/generated
        /scipy.io.loadmat.html
    .. _savemat: https://docs.scipy.org/doc/scipy/reference/generated
        /scipy.io.savemat.html

"""

# This library was developed for the Georgia Tech graduate course ECE 6258:
# Digital Image Processing with Professor Ghassan AlRegib.
# For comments and feedback, please email dippykit[at]gmail.com

__author__ = 'Brighton Ancelin, Motaz Alfarraj, Ghassan AlRegib'

from .windows import *
from .image_io import *
from .visualization import *
from .sampling import *
from .transforms import *
from .metrics import *
from .utilities import *
from .coding import *
from .adjustments import *

from numpy.fft import fft, fft2, fftshift, ifft, ifft2

from scipy.signal import convolve
from scipy.linalg import dft

from matplotlib.pyplot import axis, bar, colorbar, contour, figure, hist, \
    legend, loglog, plot, semilogx, semilogy, show, subplot, suptitle, \
    title, xlabel, ylabel

from cv2 import medianBlur, resize

from scipy.io import loadmat, savemat

