Core Configuration
==================

This page covers only the core settings most users change first. For the complete configuration (citygen, assets_rp, traffic, map, user, scooter), see the :doc:`../additional_configuration/configuration` page.

Quick guidance
--------------

* Core settings: ``simworld`` is the main global block most users adjust first (e.g., seed, dt, UE manager path).
* Advanced settings: see :doc:`../additional_configuration/configuration` for the full parameter reference in ``default.yaml``.

Global Settings (simworld)
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Parameter
     - Type
     - Description
   * - ``seed``
     - int
     - Random seed for reproducibility. Default: ``42``
   * - ``dt``
     - float
     - Time step (delta time) for simulation in seconds. Default: ``0.1``
   * - ``ue_manager_path``
     - str
     - Unreal Engine blueprint path to the UE Manager class. Default: ``"/Game/TrafficSystem/UE_Manager.UE_Manager_C"``

More configuration
------------------

See the complete parameter reference at :doc:`../additional_configuration/configuration` when you need city generation, assets retrieval, traffic, map, user, or scooter settings.
