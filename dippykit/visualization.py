"""Module of image visualization functions

This module contains an assortment of functions relevant to the plotting and
visualization of various image-relevant data

"""

# Functional imports
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from multiprocessing import Queue, Process
from threading import Thread
from tkinter import Tk, TOP, BOTH

# General imports
from typing import Callable, Any, Tuple

__author__ = 'Brighton Ancelin'

__all__ = ['imshow', 'quiver', 'surf', 'setup_continuous_rendering', 'zlabel']

def imshow(
        im: np.ndarray,
        *args,
        **kwargs
        ) -> None:
    """Displays an image

    Displays the argument image with optional parameters. If the image has a
    dtype of uint8, then by default the vmin and vmax parameters will be set
    to 0 and 255 respectively. This is to provided accurate depictions of
    otherwise dark images.

    This function is essentially a wrapper for `matplotlib.pyplot.imshow`_,
    so more detailed documentation may be found there.

    :type im: ``numpy.ndarray``
    :param im: The image to be displayed.
    :return: None

    .. note::
        This function wraps around functions from other packages. Reading
        these functions' documentations may be useful. See the **See also**
        section for more information.

    .. seealso::
        `matplotlib.pyplot.imshow`_
            Documentation of the random_noise function from Scikit Image

    .. _matplotlib.pyplot.imshow: https://matplotlib.org/api/_as_gen
        /matplotlib.pyplot.imshow.html

    """
    if im.dtype == np.uint8:
        info = np.iinfo(im.dtype)
        if 'vmin' not in kwargs:
            kwargs['vmin'] = info.min
        if 'vmax' not in kwargs:
            kwargs['vmax'] = info.max
        plt.imshow(im, *args, **kwargs)
    else:
        plt.imshow(im, *args, **kwargs)
    plt.axis('off')


def surf(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        **kwargs
        ) -> None:
    """Plots a surface

    Plots the x, y, and z values as a surface in 3D.

    This function is essentially a wrapper for
    `mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface`_, so more detailed
    documentation may be found there.

    :type x: ``numpy.ndarray``
    :param x: The array of x coordinates for the surface plot.
    :type y: ``numpy.ndarray``
    :param y: The array of y coordinates for the surface plot.
    :type z: ``numpy.ndarray``
    :param z: The array of z coordinates for the surface plot.

    .. note::
        This function wraps around functions from other packages. Reading
        these functions' documentations may be useful. See the **See also**
        section for more information.

    .. seealso::
        `mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface`_
            Documentation of the plot_surface function from Matplotlib

    .. _mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface: https://matplotlib
        .org/mpl_toolkits/mplot3d/tutorial.html#surface-plots

    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    if (1 == x.ndim) and (1 == y.ndim):
        x = x.reshape(-1, 1)
        y = y.reshape(1, -1)
    ax = plt.gca(projection='3d')
    ax.plot_surface(x, y, z, **kwargs)


def zlabel(
        s: str,
        *args,
        **kwargs
        ) -> None:
    """Writes a string to the z axis label

    Provided that the current axes has a z axis, this function will write
    the given string to the axis label.

    This function is essentially a wrapper for
    `mpl_toolkits.mplot3d.axes3d.Axes3D.set_zlabel`_, so more detailed
    documentation may be found there.

    :type s: ``str``
    :param s: The string to write to the z axis label.
    :return: None

    .. note::
        This function wraps around functions from other packages. Reading
        these functions' documentations may be useful. See the **See also**
        section for more information.

    .. seealso::
        `mpl_toolkits.mplot3d.axes3d.Axes3D.set_zlabel`_
            Documentation of the set_ylabel function (set_zlabel has no
            formal documentation) from Matplotlib

    .. _mpl_toolkits.mplot3d.axes3d.Axes3D.set_zlabel: https://matplotlib.org
        /api/_as_gen/matplotlib.axes.Axes.set_ylabel.html

    """
    ax = plt.gca()
    assert 'set_zlabel' in dir(ax), \
            "Can't set z label if the current axes don't have a z axis."
    ax.set_zlabel(s, *args, **kwargs)


def quiver(
        *args,
        **kwargs
        ) -> None:
    """Plots a field of arrows

    Plots a field of arrow on the current axes. If the following keyword
    arguments are not set, then they will take on the following default values:
        * 'units': 'xy'
        * 'angles': 'xy'
        * 'scale_units': 'xy'
        * 'scale': The mean of the magnitudes of the U and V vectors.

    This function is essentially a wrapper for
    `matplotlib.axes.Axes.quiver`_, so more detailed documentation may be
    found there.

    :return: None

    .. note::
        This function wraps around functions from other packages. Reading
        these functions' documentations may be useful. See the **See also**
        section for more information.

    .. seealso::
        `matplotlib.axes.Axes.quiver`_
            Documentation of the quiver function from Matplotlib

    .. _matplotlib.axes.Axes.quiver: https://matplotlib.org/api/_as_gen
        /matplotlib.axes.Axes.quiver.html

    """
    if 'units' not in kwargs:
        kwargs['units'] = 'xy'
    if 'angles' not in kwargs:
        kwargs['angles'] = 'xy'
    if 'scale_units' not in kwargs:
        kwargs['scale_units'] = 'xy'
    if 'scale' not in kwargs:
        if 2 == len(args) or 3 == len(args):
            scale = np.mean((args[0] ** 2 + args[1] ** 2) ** 0.5)
        elif 4 == len(args) or 5 == len(args):
            scale = np.mean((args[2] ** 2 + args[3] ** 2) ** 0.5)
        kwargs['scale'] = scale
    plt.quiver(*args, **kwargs)


def setup_continuous_rendering(
        render: Callable[[Axes, Any], None],
        update: Callable[[Queue], None],
        delay: int=100,
        auto_play: bool=True,
        ) -> None:
    """Sets up a continuous renderer

    This function sets up a window to display data that can continuously
    change. To best understand this function, try copying the example code
    below into a python file, running it, and then observing the results.

    :type render: ``Callable[[Axes, Any], None]``
    :param render: A function that takes a matplotlib Axes object and any
        data as arguments. This function returns nothing. This function should
        use the Axes object to update the rendering each time a new datum is
        received. These data are supplied through the *update* function.
    :type update: ``Callable[[Queue], None]``
    :param update: A function that takes a Queue as an argument and returns
        nothing. This function should update the rendering by putting each new
        datum into its queue via the ``Queue.put()`` function. These data are
        then subsequently rendered by the *render* function. Once this
        function places ``None`` into the queue, the rendering will cease to
        update.
    :type delay: ``int``
    :param delay: (default=100) The time (in milliseconds) between calling
        the *render* function to update the rendering. Also known as the
        refresh rate.
    :type auto_play: ``bool``
    :param auto_play: (default=True) If set to false, the rendering will
        prompt the user before each update.
    :return: None

    Examples:

    .. code-block:: python

        # This file will generate a rendering of a square moving in an image

        import numpy as np
        import dippykit as dip

        def render_square(ax, data):
            # Show the image without any axis
            ax.imshow(data, 'gray')
            ax.axis('off')

        def update_square(queue):
            # Create an arbitrary animation of a square progressively moving
            # through the image
            for i in range(9):
                square = np.zeros(9)
                square[i] = 1
                square = square.reshape((3, 3))
                queue.put(square)
            # Putting None into the queue tells the renderer to cease updating
            queue.put(None)

        if __name__ == '__main__':
            # Sets up a continuous rendering using the functions above.
            # The rendering will update every 1000 milliseconds (1 second).
            dip.setup_continuous_rendering(render_square, update_square, 1000)

    .. code-block:: python

        # This file will generate renderings of happy and sad faces

        import numpy as np
        import dippykit as dip

        def render_face(ax, data):
            # Break the data into more manageable variable names
            mouth, left_eye, right_eye, is_happy = data
            mouth_x, mouth_y = mouth
            left_eye_x, left_eye_y = left_eye
            right_eye_x, right_eye_y = right_eye
            # First, clear the axes
            ax.clear()
            # Draw the face
            ax.plot(mouth_x, mouth_y)
            ax.plot(left_eye_x, left_eye_y)
            ax.plot(right_eye_x, right_eye_y)
            # Set the appropriate title to the axes
            if is_happy:
                ax.set_title('Happy Face - Press ENTER to toggle')
            else:
                ax.set_title('Sad Face - Press ENTER to toggle')

        def update_face(queue):
            # Defining all the arrays
            mouth_x = np.array(range(31)) - 15
            happy_mouth_y = 20 * (mouth_x/15) ** 2
            sad_mouth_y = 20 - happy_mouth_y
            u = np.linspace(0, 1, 21)
            left_eye_x = np.cos(2 * np.pi * u) - 9
            right_eye_x = np.cos(2 * np.pi * u) + 9
            eye_y = np.sin(2 * np.pi * u) + 29
            # Aggregating the data into single tuples
            happy_mouth = (mouth_x, happy_mouth_y)
            sad_mouth = (mouth_x, sad_mouth_y)
            left_eye = (left_eye_x, eye_y)
            right_eye = (right_eye_x, eye_y)
            happy_face = (happy_mouth, left_eye, right_eye, True)
            sad_face = (sad_mouth, left_eye, right_eye, False)
            is_happy = True
            while True:
                # If there are less than 10 items in the queue
                if queue.qsize() < 10:
                    # Alternate with happy and sad faces
                    if is_happy:
                        queue.put(happy_face)
                        is_happy = False
                    else:
                        queue.put(sad_face)
                        is_happy = True

        if __name__ == '__main__':
            # Sets up a continuous rendering using the functions above.
            # This rendering will await user input before updating the display.
            dip.setup_continuous_rendering(render_face, update_face,
                                           auto_play=False)

    """
    window = Tk()
    queue = Queue()
    update_process = Process(target=update, args=(queue,))
    update_process.start()
    ax, canvas = _setup_window(window)
    _update_window(render, queue, window, ax, canvas, delay, auto_play)
    window.mainloop()
    update_process.terminate()


def _setup_window(
        window: Tk
        ) -> Tuple[Axes, FigureCanvasTkAgg]:
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(1, 1, 1)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.show()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
    return ax, canvas


def _update_window(
        render: Callable[[Axes, Any], None],
        queue: Queue,
        window: Tk,
        ax: Axes,
        canvas: FigureCanvasTkAgg,
        delay: int,
        auto_play: bool,
        wait_thread: Thread=None,
        ) -> None:
    do_update = False
    if wait_thread is None:
        do_update = True
    elif not wait_thread.is_alive():
        wait_thread = None
        do_update = True
    if do_update and (not queue.empty()):
        val = queue.get_nowait()
        if val is not None:
            render(ax, val)
            canvas.draw()
            if not auto_play:
                wait_thread = Thread(
                    target=lambda: input('Press ENTER to continue...'))
                wait_thread.setDaemon(True)
                wait_thread.start()
            window.after(delay, _update_window, render, queue, window, ax,
                         canvas, delay, auto_play, wait_thread)
    else:
        window.after(delay, _update_window, render, queue, window, ax,
                     canvas, delay, auto_play, wait_thread)

