.. currentmodule:: pecd

Public API: CHIRALEX package
*****************************




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

.. autoclass:: pecd.wavefunction.Map

	.. automethod:: genmap_femlist
	.. automethod:: map_dvr_femlist_nat


.. autoclass:: pecd.wavefunction.GridRad

	.. automethod:: gen_grid







Electric field module (:code:`field`)
-------------------------------------

.. autosummary::

pecd.field.Field


.. autoclass:: pecd.field.Field

.. automethod:: gen_field


