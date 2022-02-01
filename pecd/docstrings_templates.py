
#function
"""Summary line.

    Extended description of function.

    .. math::
        
        ax^2 + bx + c = 0
    Args:
        grid_type : str
            type of time grid

    Returns: tuple
        t_c : numpy.ndarray
            times at which to evaluate `Hamiltonian` in :py:func:`Map.genmap_femlist` function

        dt: numpy.ndarray
            Something to write about :py:class:`Map` class

    Example:

    .. code-block:: python

        [sym for J in self.Jlist1 for sym in symlist1[J]]

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