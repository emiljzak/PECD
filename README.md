# CMI Python-project template

Template for new CMI Python-software projects.

This includes the basic setup of a Python project with library, scripts, pytest testing, and [sphinx
documentation](https://cmi-python-project-template.readthedocs.io).


## Installation

Standard setuptool installation, run
```
python setup.py install
```
or advanced versions using `develop`, `--user`, etc. – see the
[documentation](https://cmi-python-project-template.readthedocs.io) for more details.


## Documentation

See this very [sphinx-generated
documentation](https://cmi-python-project-template.readthedocs.io) for further
details; it can be produced by running
```
python setup.py build_sphinx
```

See also the [license](./LICENSE.md).

## Testing

Standard tests are implemented in `tests/`, run by executing
```
pytest
```



<!-- Put Emacs local variables into HTML comment
Local Variables:
coding: utf-8
fill-column: 100
End:
-->

#Code speed up (/feature-parallel): efficient matrix elements and parallelization on CPU and GPU with numba and cuPy


Optimization of the code for better memory and time performance.
1) We need to calculate radial integrals only for a single bin. The only bin-dependent term is the centrifugal energy. This is going to reduce the computation time significantly.
2) Time and memory diagnostic tools
3) Code levels:
LEVEL 0: a) direct eigensolver (numpy, full/sparse)
b) direct eigensolver (scipy, full/sparse)
 
LEVEL 1: a) Lanczos eigensolver (full matrix repr.)
b) Lanczos eigensolver (CSR)
 
LEVEL 2: a) Lanczos eigensolver (full matrix repr. + @jit(CPU))
b) Lanczos eigensolver (CSR+ @jit(CPU))
 
LEVEL 3: a) Lanczos eigensolver (full matrix repr. + cuPy(GPU))
b) Lanczos eigensolver (full matrix repr. + numba(GPU))
 
LEVEL 4: a) Lanczos eigensolver (CSR+ cuPy(GPU))
b) Lanczos eigensolver (CSR+ numba(GPU))
 
LEVEL 5: Both MVP and matmul parallelized . In levels 2-4 only MVP is parallelized
