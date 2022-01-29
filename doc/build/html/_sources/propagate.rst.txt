.. currentmodule:: pecd

Public API: CHIRALEX package
*****************************


Hamiltonian module (:code:`hamiltonian`)
------------------------------------------

.. autosummary::

	pecd.hamiltonian.Hamiltonian


.. autoclass:: pecd.hamiltonian.Hamiltonian

	.. automethod:: gen_klist



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
	
.. autoclass:: pecd.wavefunction.Map

	.. automethod:: genmap_femlist
	.. automethod:: map_dvr_femlist_nat


.. autoclass:: pecd.wavefunction.GridRad

	.. automethod:: gen_grid



.. autoclass:: pecd.wavefunction.GridEuler

	.. automethod:: gen_euler_grid
	.. automethod:: gen_euler_grid_2D




Electric field module (:code:`field`)
-------------------------------------

.. autosummary::

pecd.field.Field


.. autoclass:: pecd.field.Field

.. automethod:: gen_field


