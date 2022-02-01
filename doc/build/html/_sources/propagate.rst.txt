.. currentmodule:: pecd

Public API: CHIRALEX package
*****************************


Hamiltonian module (:code:`hamiltonian`)
------------------------------------------

.. autosummary::

	pecd.hamiltonian.Hamiltonian

.. autoclass:: pecd.hamiltonian.Hamiltonian

	.. automethod:: gen_klist
	.. automethod:: calc_vxi


Propagation of the wavepacket (:code:`propagate`)
-------------------------------------------------

.. autosummary::

	pecd.propagate.Propagator

.. autoclass:: pecd.propagate.Propagator

	.. automethod:: gen_timegrid


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

.. autoclass:: pecd.analyze.analysis

	.. automethod:: read_wavepacket
