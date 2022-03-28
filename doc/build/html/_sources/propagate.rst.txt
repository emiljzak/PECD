.. currentmodule:: pecd

Public API: CHIRALEX package
*****************************


Propagation of the wavepacket (:code:`propagate`)
-------------------------------------------------

.. autosummary::

	pecd.propagate.Propagator

.. autoclass:: pecd.propagate.Propagator

	.. automethod:: gen_timegrid
	.. automethod:: prop_wf


Hamiltonian module (:code:`hamiltonian`)
------------------------------------------

.. autosummary::

	pecd.hamiltonian.Hamiltonian

.. autoclass:: pecd.hamiltonian.Hamiltonian

	.. automethod:: build_dmat
	.. automethod:: build_keomat_blocks
	.. automethod:: build_jmat
	.. automethod:: gen_klist
	.. automethod:: calc_vxi
	.. automethod:: gen_tjmat
	.. automethod:: build_potmat_chiralium_anton
	.. automethod:: gen_3j_dip
	.. automethod:: map_tjmat_dip
	.. automethod:: build_intmat
	.. automethod:: calc_dipxi
	
Potential module (:code:`potential`)
-------------------------------------

.. autosummary::


Wavefunction module (:code:`wavefunction`)
------------------------------------------

.. autosummary::

	pecd.wavefunction.Map
	pecd.wavefunction.GridRad
	pecd.wavefunction.GridEuler
	pecd.wavefunction.Psi

.. autoclass:: pecd.wavefunction.Map

	.. automethod:: genmap_femlist
	.. automethod:: map_dvr_femlist_nat
	.. automethod:: gen_sphlist

.. autoclass:: pecd.wavefunction.GridRad

	.. automethod:: gen_grid

.. autoclass:: pecd.wavefunction.GridEuler

	.. automethod:: gen_euler_grid
	.. automethod:: gen_euler_grid_2D
	.. automethod:: read_euler_grid
	
.. autoclass:: pecd.wavefunction.Psi

	.. automethod:: project_psi_global



Electric field module (:code:`field`)
-------------------------------------

.. autosummary::

	pecd.field.Field

.. autoclass:: pecd.field.Field

	.. automethod:: gen_field


Analysis module (:code:`analyze`)
-------------------------------------

.. autosummary::

	pecd.analyze.analysis
	pecd.analyze.momentumfuncs
	
.. autoclass:: pecd.analyze.analysis

	.. automethod:: read_wavepacket
	.. automethod:: momentum_au_to_energy_ev
	.. automethod:: energy_ev_to_momentum_au
	
	
.. autoclass:: pecd.analyze.momentumfuncs

	.. automethod:: W2D
	.. automethod:: W2D_calc
	.. automethod:: W2D_plot
	.. automethod:: PES
	.. automethod:: PES_plot
	.. automethod:: W2Dav
	.. automethod:: W2Dav_calc
	.. automethod:: W2Dav_plot
	.. automethod:: PESav
	.. automethod:: legendre_expansion
	
	
	
	
	


