PECD code user guide
======================

The PECD package is developed and maintained by the Controlled Molecule Imaging group (CMI) at
the Center for Free-Electron Laser Science (CFEL), Hamburg, Germany.

For further reading see:

General usage
-------------

The basic usage so far includes the following steps:
1) Generation of electrostatic potential (ESP) from a quantum chemistry package (we use psi4) on a cartesian grid.
2) Run main code (propagate.py) with option     params['gen_adaptive_quads'] = TRUE
  This option generates and saves in file a list of degrees for adaptive angular quadratures (Lebedev for now). This list is potential and basis dependent. It is going to be read upon construction of the hamiltonian.
3) Generate field-free Hamiltonian with option: params['method'] = "static", store it in a file. Choose CSR or Numpy storing format.
4) Generate initial wavefunction by diagonalization of field-free Hamiltonian: store in a file.
5) Run params['method'] = "dynamic_direct" for time-propagation



.. comment
   Local Variables:
   coding: utf-8
   fill-column: 100
   End:
