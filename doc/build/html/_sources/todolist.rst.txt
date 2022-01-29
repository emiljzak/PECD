ToDo list
=========

.. toctree::
   :maxdepth: 2

.. todolist::


1. Use jit for KEO, split filling up the sparse matrix into a separate function
2. Consider using SciPy.sparse LinearOperator class with custom matrix-vector product (NUMBA?) in expm_multiply in prop_wf. Can it give speed-up?
3. Implement variable bin sizes and numbers of basis functions in each bin based on femlist. Presently each bin must have equal size and equal number of basis functions.
4. Move 'params' syntax into a secondary layer, leave pure keywords in the input file.
5. Use scipy.sparse dia_matrix for constructing inflated KD and KC.
6. Try numba jit (parallel = True) with prange when constructing the KEO with (indptr,indices,data)->csr. Loop over bins, loop over points. Any gain vs scipy.sparse.block_diag? Consider cupyx.scipy.sparse for the same purpose.
7. Parallelize calc_vxi part in potmat (loop over grid points).