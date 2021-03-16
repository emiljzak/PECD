PECD code user guide
======================

The PECD package is developed and maintained by the Controlled Molecule Imaging group (CMI) at
the Center for Free-Electron Laser Science (CFEL), Hamburg, Germany.

For further reading see:

General usage
-------------

The basic usage so far includes the following steps:
1) Generation of electrostatic potential (ESP) from a quantum chemistry package (we use psi4) on a cartesian grid.
2) Run main code (propagate.py) with option     params['gen_adaptive_quads'] = TRUE . This option generates and saves in file a list of degrees for adaptive angular quadratures (Lebedev for now). This list is potential and basis dependent. It is going to be read upon construction of the hamiltonian.
3) Generate field-free Hamiltonian with option: params['method'] = "static", store its eigenfunctions in a file. Choose CSR or Numpy storing format.
5) Run params['method'] = "dynamic_direct" for time-propagation. In first run you can save the field-free hamiltonian into file. In subsequent runs you can read both the initial wavefunction and the field-free hamiltonian.


Initial wavefunction
--------------------

The initial wavefunction of the active electron can be defined in following ways:
1. Manual entry
2. Projection of a 3D-grid representation of the wavefunction onto the spectral basis
3. eigenfunction of a given Hamiltonian (for example field-free)
4. Read spectral representation from file



.. comment
   Local Variables:
   coding: utf-8
   fill-column: 100
   End:
