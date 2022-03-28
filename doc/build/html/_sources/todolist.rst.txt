ToDo list
=========

.. toctree::
   :maxdepth: 2

.. todolist::


1. Use jit for KEO, split filling up the sparse matrix into a separate function
2. Consider using SciPy.sparse LinearOperator class with custom matrix-vector product (NUMBA?) in expm_multiply in prop_wf. Can it give speed-up?
3. Parallelize matrix-exponential-vector products in prop_wf. Consider for the start quspin.tools.expm_multiplu_parallel,https://pypi.org/project/sparse-dot-mkl/ or quimb.linalg.base_linalg.expm_multiply or an implementation invoking CuPy or Numba written MVP with homemade lanczos propagator.
4. Implement variable bin sizes and numbers of basis functions in each bin based on femlist. Presently each bin must have equal size and equal number of basis functions.
5. Move 'params' syntax into a secondary layer, leave pure keywords in the input file.
6. Use scipy.sparse dia_matrix for constructing inflated KD and KC.
7. Try numba jit (parallel = True) with prange when constructing the KEO with (indptr,indices,data)->csr. Loop over bins, loop over points. Any gain vs scipy.sparse.block_diag? Consider cupyx.scipy.sparse for the same purpose.
8. Parallelize calc_vxi part in potmat (loop over grid points).
9. Cleanup the Hamiltonian class
10. alpha averaging of Flm and direct calculation of b coefficients from F_l0