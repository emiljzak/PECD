
#function
"""Summary line.

    Extended description of function.

    .. math::
        ax^2 + bx + c = 0
        :label: my_label
    Args:
        grid_type : str
            type of time grid

    Returns: tuple
        fdsf :math:`\mu`: fdsfsd numpy.ndarray
            times at which to evaluate `Hamiltonian` in :py:func:`Map.genmap_femlist` function in :eq:`my_label`    

        dt: numpy.ndarray
            Something to write about :py:class:`Map` class

    .. note:: This is a **note** box
    .. warning:: This is a **warning** box.

    Example:

    .. code-block:: python

        def roots(a, b, c):
            q = b**2 - 4*a*c
            root1 = -b + sqrt(q)/float(2*a)
            root2 = -b - sqrt(q)/float(2*a)
            return root1, root2

    .. code-block:: python

        [i for t in self.tgrid for a in tgrid[i]]


    Raises: `NotImplementedError` because it is an error

    .. figure:: _images/ham_nat.png
        :height: 200
        :width: 200
        :align: center

    Some text


"""

#module
"""A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

  Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""


#class
"""Class contains methods related to the time-propagation of the wavefunction.

Args:
    filename : str
        fdsfsd
Kwargs:
    thresh : float
        fdsf

Attrs:
    params : dict
        Keeps relevant parameters.
    irun : int
        id of the run over the Euler grid
"""